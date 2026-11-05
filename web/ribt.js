// ribt in the browser: lilt G2P (wasm) → acoustic encoder → durations →
// decoder → 1-step flow → Vocos vocoder, all on WebGPU, no server.
//
// The GPU programs are tang's own WGSL, exported per shape bucket by
// `ribt-web-export` (graphs/*.json); this file only uploads buffers and
// dispatches kernels in the exported order, so the math is the same code the
// native wgpu backend runs. Host steps (durations, alignment, noise, pitch
// shift) mirror `ribt-tts synth` line for line.
//
//   const tts = await RIBT.load({ base: "dist/", onProgress })
//   const r = await tts.synth("Hello there.", { onStage })
//   for await (const r of tts.speak(longText)) play(r.audio)

import initLilt, * as lilt from "./pkg/lilt_wasm.js";

const HOP = 256;

// ---------------------------------------------------------------------------
// host helpers

const F16 = (() => {
  // f16 bits → f32, exact (65536-entry table)
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

/// `Rng` of src/model.rs: xorshift64 + Box–Muller, bit-exact.
class Rng {
  constructor(seed) {
    this.s = (BigInt.asUintN(64, BigInt(seed) * 0x9e3779b97f4a7c15n)) | 1n;
  }
  nextU64() {
    let x = this.s;
    x ^= BigInt.asUintN(64, x << 13n);
    x ^= x >> 7n;
    x ^= BigInt.asUintN(64, x << 17n);
    this.s = x;
    return x;
  }
  uniform() {
    return Number(this.nextU64() >> 11n) / 2 ** 53;
  }
  normal() {
    const u = Math.max(this.uniform(), 1e-300), v = this.uniform();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
}

async function sha256hex(buf) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", buf));
  return [...d].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchBytes(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  if (!r.body || !onBytes) return new Uint8Array(await r.arrayBuffer());
  const len = +r.headers.get("content-length") || 0;
  const parts = [];
  let got = 0;
  for (const rd = r.body.getReader(); ; ) {
    const { done, value } = await rd.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onBytes(value.length);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const p of parts) out.set(p, o), (o += p.length);
  void len;
  return out;
}

// A file may be split into `parts` pieces (`file.0`, `file.1`, …) for hosts with a per-file size cap;
// the pieces are fetched in parallel and joined before the sha256 check.
async function fetchFile(base, f, onBytes) {
  if (!f.parts) return fetchBytes(base + f.file, onBytes);
  const ps = await Promise.all(Array.from({ length: f.parts }, (_, i) => fetchBytes(`${base}${f.file}.${i}`, onBytes)));
  const out = new Uint8Array(ps.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of ps) out.set(p, o), (o += p.length);
  return out;
}

const bucket = (v, bs) => bs.find((b) => b >= v);

// ---------------------------------------------------------------------------
// GPU: one exported graph

class Graph {
  constructor(rt, name, plan) {
    this.rt = rt;
    this.name = name;
    this.plan = plan;
  }

  async compile() {
    const { dev } = this.rt;
    const p = this.plan;
    const U = GPUBufferUsage;
    this.bufs = p.buffers.map((b) => {
      if (b.w) return this.rt.weight(b.w);
      if (b.c) return this.rt.constant(b.c[0], b.c[1]);
      return dev.createBuffer({ size: Math.max(4, b.len * 4), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    });
    this.data = {};
    p.buffers.forEach((b, i) => b.d && (this.data[b.d] = i));
    // one pipeline per distinct shader (binding count fixed by its source)
    const nreads = new Map();
    for (const k of p.kernels) nreads.set(k[0], k[1].length);
    const pipes = new Map();
    await Promise.all(
      [...nreads].map(async ([si, nr]) => {
        const entries = [];
        for (let j = 0; j < nr; j++) entries.push({ binding: j, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } });
        entries.push({ binding: nr, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
        entries.push({ binding: nr + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } });
        const bgl = dev.createBindGroupLayout({ entries });
        const module = this.rt.module(p.shaders[si]);
        const pipe = await dev.createComputePipelineAsync({
          layout: dev.createPipelineLayout({ bindGroupLayouts: [bgl] }),
          compute: { module, entryPoint: "main" },
        });
        pipes.set(si, { pipe, bgl });
      })
    );
    // uniforms: {points, 0, 0, 0} in one buffer, 256-byte aligned
    const ub = dev.createBuffer({ size: 256 * p.kernels.length, usage: U.UNIFORM | U.COPY_DST });
    const u = new Uint32Array(64 * p.kernels.length);
    p.kernels.forEach((k, i) => (u[64 * i] = k[3]));
    dev.queue.writeBuffer(ub, 0, u);
    this.steps = p.kernels.map(([si, reads, out, , count, wg], i) => {
      const { pipe, bgl } = pipes.get(si);
      const entries = reads.map((r, j) => ({ binding: j, resource: { buffer: this.bufs[r] } }));
      entries.push({ binding: reads.length, resource: { buffer: this.bufs[out] } });
      entries.push({ binding: reads.length + 1, resource: { buffer: ub, offset: 256 * i, size: 16 } });
      const groups = Math.ceil(count / wg);
      return { pipe, bind: dev.createBindGroup({ layout: bgl, entries }), x: Math.min(groups, 65535), y: Math.ceil(groups / 65535) };
    });
    this.kernels = p.kernels.length;
    this.plan = { buffers: p.buffers, outputs: p.outputs }; // drop shader text
    return this;
  }

  /// Write named data inputs, run every kernel in one pass, read the named outputs.
  async run(inputs, outs) {
    const { dev } = this.rt;
    for (const [k, v] of Object.entries(inputs)) {
      const i = this.data[k];
      if (i === undefined) continue; // a dead input (optimized away)
      if (v.length !== this.plan.buffers[i].len) throw new Error(`${this.name}: ${k} has ${v.length}, wants ${this.plan.buffers[i].len}`);
      dev.queue.writeBuffer(this.bufs[i], 0, v);
    }
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const s of this.steps) {
      pass.setPipeline(s.pipe);
      pass.setBindGroup(0, s.bind);
      pass.dispatchWorkgroups(s.x, s.y);
    }
    pass.end();
    const reads = outs.map((o) => {
      const i = this.plan.outputs[o];
      const n = this.plan.buffers[i].len * 4;
      const st = dev.createBuffer({ size: n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.copyBufferToBuffer(this.bufs[i], 0, st, 0, n);
      return st;
    });
    dev.queue.submit([enc.finish()]);
    const res = {};
    await Promise.all(
      reads.map(async (st, j) => {
        await st.mapAsync(GPUMapMode.READ);
        res[outs[j]] = new Float32Array(st.getMappedRange().slice(0));
        st.destroy();
      })
    );
    return res;
  }
}

// ---------------------------------------------------------------------------
// runtime

export class RIBT {
  static async load({ base = "dist/", onProgress = () => {}, warm = true } = {}) {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
    const t0 = performance.now();
    const rt = new RIBT();
    rt.base = base.endsWith("/") ? base : base + "/";
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const L = adapter.limits;
    rt.dev = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
        maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxBufferSize: L.maxBufferSize,
      },
    });
    rt.dev.lost.then((i) => console.error("WebGPU device lost:", i.message));
    rt.adapterInfo = adapter.info || {};
    const [man] = await Promise.all([fetch(rt.base + "manifest.json").then((r) => r.json()), initLilt()]);
    rt.m = man;
    lilt.set_accent(man.synth.accent);

    // weights + constants, verified
    const files = [...Object.values(man.weights), man.consts];
    const total = files.reduce((a, f) => a + f.bytes, 0);
    let got = 0;
    const blobs = {};
    await Promise.all(
      files.map(async (f) => {
        const b = await fetchFile(rt.base, f, (n) => onProgress({ stage: "download", got: (got += n), total }));
        const h = await sha256hex(b);
        if (h !== f.sha256) throw new Error(`${f.file}: sha256 ${h}, manifest says ${f.sha256}`);
        blobs[f.file] = b;
      })
    );
    rt.blobs = {};
    for (const [k, f] of Object.entries(man.weights)) rt.blobs[k] = blobs[f.file];
    rt.constBlob = blobs[man.consts.file];
    rt.wbufs = new Map();
    rt.cbufs = new Map();
    rt.modules = new Map();
    rt.graphs = new Map();
    rt.timings = { load_ms: performance.now() - t0 };
    if (warm) {
      // the smallest buckets now (first audio), the rest in the background
      const t1 = performance.now();
      onProgress({ stage: "compile" });
      await Promise.all([`enc_n${man.ns[0]}`, `dec_n${man.ns[0]}_t${man.ts[0]}`, `flow_t${man.ts[0]}`, `voc_t${man.ts[0]}`].map((g) => rt.graph(g)));
      rt.timings.compile_ms = performance.now() - t1;
    }
    rt.timings.ready_ms = performance.now() - t0;
    onProgress({ stage: "ready", ...rt.timings });
    return rt;
  }

  /// Background-compile every bucket (call after the first utterance).
  prewarm() {
    return Promise.all(this.m.graphs.map((g) => this.graph(g)));
  }

  module(code) {
    let m = this.modules.get(code);
    if (!m) this.modules.set(code, (m = this.dev.createShaderModule({ code })));
    return m;
  }

  weight(name) {
    let b = this.wbufs.get(name);
    if (b) return b;
    const [blob, off, n, dt] = this.m.index[name];
    const src = this.blobs[blob];
    const f = new Float32Array(n);
    if (dt === "f16") {
      const h = new Uint16Array(src.buffer, src.byteOffset + off, n);
      for (let i = 0; i < n; i++) f[i] = F16[h[i]];
    } else f.set(new Float32Array(src.buffer.slice(src.byteOffset + off, src.byteOffset + off + 4 * n)));
    b = this.dev.createBuffer({ size: Math.max(4, n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.dev.queue.writeBuffer(b, 0, f);
    this.wbufs.set(name, b);
    return b;
  }

  constant(off, n) {
    let b = this.cbufs.get(off);
    if (b) return b;
    b = this.dev.createBuffer({ size: Math.max(4, n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.dev.queue.writeBuffer(b, 0, this.constBlob.buffer, this.constBlob.byteOffset + off, n * 4);
    this.cbufs.set(off, b);
    return b;
  }

  graph(name) {
    let g = this.graphs.get(name);
    if (!g) {
      g = fetch(this.base + `graphs/${name}.json`)
        .then((r) => r.json())
        .then((p) => new Graph(this, name, p).compile());
      this.graphs.set(name, g);
    }
    return g;
  }

  /// lilt only: phonemes, model tokens and words.
  phonemize(text) {
    return JSON.parse(lilt.phonemize(text));
  }

  /// One chunk (≤ the largest buckets) → every stage. `onStage(name, partial)`
  /// fires as each stage lands, for the page to animate.
  async synth(text, opts = {}) {
    const m = this.m;
    const o = { ...m.synth, ...opts };
    const onStage = o.onStage || (() => {});
    const T = { start: performance.now() };
    const lap = (k) => (T[k] = performance.now());

    // 1. G2P
    const g2p = this.phonemize(text);
    const toks = [0, ...g2p.toks.map((t) => t[1]), 0];
    const phones = ["", ...g2p.toks.map((t) => t[0]), ""];
    const n = bucket(toks.length, m.ns);
    if (!n) throw new Error(`${toks.length} tokens: over the ${m.ns.at(-1)}-token bucket (use speak())`);
    lap("g2p");
    onStage("phonemes", { text, phonemes: g2p.ps, words: g2p.words, tokens: toks.map((id, i) => ({ ph: phones[i], id })) });

    // 2. encoder: token states, ln(d+1), pitch, energy, speaker
    const V = m.vocab;
    const onehot = new Float32Array(V * n), tmask = new Float32Array(n);
    toks.forEach((id, i) => ((onehot[id * n + i] = 1), (tmask[i] = 1)));
    const enc = await (await this.graph(`enc_n${n}`)).run(
      { onehot, tmask, refmel: new Float32Array(m.refmel), rw: new Float32Array(m.rw) },
      ["h", "ld", "pitch", "energy", "spk"]
    );
    const pace = 1 / o.rate;
    const dshift = (o.pitch_st * Math.LN2) / 12 / m.norm[1];
    const pitch = enc.pitch.map((v) => v + Math.fround(dshift));
    const durs = toks.map((_, i) => Math.max(0, Math.round((Math.exp(enc.ld[i]) - 1) * pace)));
    const frames = durs.reduce((a, b) => a + b, 0);
    lap("enc");
    const tokPitchHz = toks.map((_, i) => Math.exp(pitch[i] * m.norm[1] + m.norm[0]));
    onStage("durations", { durs, frames, seconds: (frames * HOP) / m.sr, tokenPitchHz: tokPitchHz });
    const t = bucket(frames, m.ts);
    if (!t) throw new Error(`${frames} frames: over the ${m.ts.at(-1)}-frame bucket (use speak())`);

    // 3. decoder: mel, F0, hidden
    const align = new Float32Array(n * t), fmask = new Float32Array(t);
    let pos = 0;
    durs.forEach((k, i) => {
      for (let f = pos; f < Math.min(pos + k, t); f++) align[i * t + f] = 1;
      pos += k;
    });
    fmask.fill(1, 0, frames);
    const dec = await (await this.graph(`dec_n${n}_t${t}`)).run(
      { h: enc.h, pitch, energy: enc.energy, tmask, align, fmask, spk_dec: enc.spk },
      ["mel", "f0", "hid"]
    );
    const f0 = new Float32Array(frames);
    for (let f = 0; f < frames; f++) f0[f] = dec.f0[t + f] > 0 ? Math.exp(dec.f0[f] * m.norm[1] + m.norm[0]) : 0;
    const C = m.n_mels;
    const trim = (x) => {
      const y = new Float32Array(C * frames);
      for (let c = 0; c < C; c++) y.set(x.subarray(c * t, c * t + frames), c * frames);
      return y;
    };
    lap("dec");
    onStage("decoder", { f0, mel: trim(dec.mel), frames, nMels: C });

    // 4. flow, 1 NFE shortcut: x1 = x0 + v(x0, t=0, d=1)
    const [fm, fs] = m.fnorm;
    const mu = new Float32Array(C * t), x0 = new Float32Array(C * t);
    const rng = new Rng(o.seed);
    for (let c = 0; c < C; c++)
      for (let f = 0; f < frames; f++) {
        mu[c * t + f] = (dec.mel[c * t + f] - fm) / fs;
        x0[c * t + f] = o.temp * rng.normal();
      }
    const abias = new Float32Array(t * t);
    const kmax = Math.max(frames, 1);
    for (let i = 0; i < t; i++) abias.fill(-1e4, i * t + kmax, (i + 1) * t);
    if (o.nfe !== 1 || !o.shortcut) throw new Error("this export runs the 1-step shortcut flow only");
    const fl = await (await this.graph(`flow_t${t}`)).run({ xt: x0, mu, hc: dec.hid, temb: new Float32Array(m.temb), fmask, abias }, ["v"]);
    const vin = new Float32Array(C * t).fill(m.mel_floor);
    for (let c = 0; c < C; c++)
      for (let f = 0; f < frames; f++) {
        const i = c * t + f;
        vin[i] = Math.fround(x0[i] + fl.v[i]) * fs + fm;
      }
    const mel = trim(vin);
    lap("flow");
    onStage("flow", { mel, frames, nMels: C });

    // 5. vocoder (padded to the bucket with silence, trimmed)
    const voc = await (await this.graph(`voc_t${t}`)).run({ mel: vin }, ["audio"]);
    const audio = voc.audio.slice(0, frames * HOP);
    lap("voc");
    const timings = {
      g2p_ms: T.g2p - T.start,
      enc_ms: T.enc - T.g2p,
      dec_ms: T.dec - T.enc,
      flow_ms: T.flow - T.dec,
      voc_ms: T.voc - T.flow,
      total_ms: T.voc - T.start,
    };
    timings.rtf = timings.total_ms / 1000 / (audio.length / m.sr);
    const r = {
      text,
      phonemes: g2p.ps,
      words: g2p.words,
      tokens: toks.map((id, i) => ({ ph: phones[i], id })),
      durs,
      frames,
      f0,
      melDecoder: trim(dec.mel),
      mel,
      nMels: C,
      audio,
      sr: m.sr,
      hop: HOP,
      buckets: { n, t },
      timings,
    };
    onStage("audio", r);
    return r;
  }

  /// Any length: sentences (then clauses, then word runs) synthesized in
  /// order; yields each chunk's result as soon as it is ready.
  async *speak(text, opts = {}) {
    for (const chunk of this.chunks(text)) yield* this._fit(chunk, opts);
  }

  async *_fit(chunk, opts) {
    try {
      yield await this.synth(chunk, opts);
    } catch (e) {
      if (!/bucket/.test(e.message)) throw e;
      const w = chunk.split(/\s+/);
      if (w.length < 2) throw e;
      const h = w.length >> 1;
      yield* this._fit(w.slice(0, h).join(" "), opts);
      yield* this._fit(w.slice(h).join(" "), opts);
    }
  }

  chunks(text) {
    const maxTok = this.m.ns.at(-1) - 2;
    const out = [];
    const sents = text.replace(/\s+/g, " ").trim().match(/[^.!?…]+[.!?…]*["”’)]*\s*|.+$/g) || [];
    for (let s of sents) {
      s = s.trim();
      if (!s) continue;
      if (this.phonemize(s).toks.length <= maxTok) {
        out.push(s);
        continue;
      }
      let cur = "";
      for (const part of s.split(/(?<=[,;:—])\s+/)) {
        const cand = cur ? cur + " " + part : part;
        if (cur && this.phonemize(cand).toks.length > maxTok) out.push(cur), (cur = part);
        else cur = cand;
      }
      if (cur) out.push(cur);
    }
    return out;
  }
}

export { Rng };
