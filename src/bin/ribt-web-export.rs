//! `ribt-web-export`: everything the browser runtime (web/) needs to speak
//! ribt with no server: per-bucket WGSL programs (kernels, buffer plan,
//! dispatch order) as JSON, weights-only fp16 blobs with sha256s, the
//! reference-voice tensors and the host constants.
//!
//!   ribt-web-export --rel WEIGHTS_DIR --out web/dist [--check web/parity/lines.tsv --native DIR]
//!
//! `--check` also runs the browser's pipeline natively (same buckets, same
//! fp16-rounded weights, padded vocoder) and reports its RMS against the
//! `ribt-tts synth` wavs in `--native`: the error budget the browser gets
//! before any WebGPU difference.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use ribt::acoustic::{self, Dims, TConfig};
use ribt::ckpt;
use ribt::flow::{self, FConfig, FDims};
use ribt::model::{self, Built, Config, Role, Rng};
use ribt::runtime::{optimize, Device, Program};
use ribt::{dsp, lilt};
use tang_compile::tensor::{precision, Format, TensorGraph, TensorId};

const HOP: usize = 256;
const SR: usize = 24000;
const REF_T: usize = 256;
/// Silence in the frontend's log-mel (ln 1e-5): vocoder padding.
const MEL_FLOOR: f32 = -11.512925;
/// Token buckets (multiples of the native 32) and frame buckets (of 64).
const NS: [usize; 3] = [64, 128, 192];
const TS: [usize; 3] = [256, 512, 1024];
/// Weights shipped as fp16 (`model.param` prefixes); the rest (pitch, energy,
/// duration and reference heads, ~1.6M params) stay f32: fp16 pitch/energy
/// heads alone move the waveform ~5% (rel RMS), these bodies ~1%.
const SHIP_F16: &str = "ac.enc,ac.dec,fl,voc";

fn is_f16(name: &str, set: &str) -> bool {
    set.split(',').any(|h| !h.is_empty() && name.starts_with(h))
}

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned())
}

fn home(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(r) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(r),
        None => PathBuf::from(p),
    }
}

fn refs(v: &[Vec<f32>]) -> Vec<&[f32]> {
    v.iter().map(|x| x.as_slice()).collect()
}

fn f16_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| precision::f16_from_bits(precision::f16_bits(x as f64)) as f32).collect()
}

fn sha256(path: &Path) -> String {
    let o = std::process::Command::new("shasum").args(["-a", "256"]).arg(path).output().expect("shasum");
    String::from_utf8_lossy(&o.stdout).split_whitespace().next().unwrap_or("").to_string()
}

fn js(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => write!(o, "\\u{:04x}", c as u32).unwrap(),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn nums(v: &[f32]) -> String {
    let s: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
    format!("[{}]", s.join(","))
}

/// Constant tables shared by every graph, deduplicated by content.
#[derive(Default)]
struct Consts {
    blob: Vec<u8>,
    at: HashMap<Vec<u8>, (usize, usize)>,
}

impl Consts {
    fn add(&mut self, v: &[f32]) -> (usize, usize) {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        if let Some(&r) = self.at.get(&bytes) {
            return r;
        }
        let r = (self.blob.len(), v.len());
        self.blob.extend(&bytes);
        self.at.insert(bytes, r);
        r
    }
}

/// One graph's WebGPU plan as JSON: buffers (what fills each: a weight by
/// name, a named data input, a constant range, or nothing), then kernels in
/// dispatch order. Binding order per kernel: reads (read-only storage), the
/// output (read-write), then a uniform `{points, 0, 0, 0}`; workgroups =
/// ceil(count / wg) split as (min(·, 65535), ceil(· / 65535)).
fn plan_json(b: &Built, specs: &[model::ParamSpec], prefix: &str, outs: &[(&str, TensorId)], consts: &mut Consts) -> (String, usize) {
    let ids: Vec<TensorId> = outs.iter().map(|o| o.1).collect();
    let (g, oids): (TensorGraph, Vec<TensorId>) = optimize(&b.g, &ids);
    let p = g.to_wgsl(&oids);
    let alias: HashMap<TensorId, TensorId> = p.aliases.iter().copied().collect();
    let storage = |mut t: TensorId| {
        while let Some(&s) = alias.get(&t) {
            t = s;
        }
        t
    };
    let slot: HashMap<TensorId, usize> = p.buffers.iter().enumerate().map(|(i, bs)| (bs.node, i)).collect();
    let mut s = String::from("{\"buffers\":[");
    for (i, bs) in p.buffers.iter().enumerate() {
        assert_eq!(bs.format, Format::F32, "f32 buffers only");
        if i > 0 {
            s.push(',');
        }
        write!(s, "{{\"len\":{}", bs.len).unwrap();
        if let Some(k) = bs.input {
            match &b.inputs[k].role {
                Role::Param(j) => write!(s, ",\"w\":{}", js(&format!("{prefix}.{}", specs[*j].name))).unwrap(),
                Role::Data(n) => write!(s, ",\"d\":{}", js(n)).unwrap(),
                Role::Const(v) => {
                    let (o, n) = consts.add(v);
                    write!(s, ",\"c\":[{o},{n}]").unwrap()
                }
                r => panic!("inference graph input {r:?}"),
            }
        }
        s.push('}');
    }
    // shaders deduplicated within the graph
    let mut srcs: Vec<&str> = Vec::new();
    let mut src_ix: HashMap<&str, usize> = HashMap::new();
    s.push_str("],\"kernels\":[");
    for (i, k) in p.kernels.iter().enumerate() {
        assert!(!k.source.starts_with("#include"), "MSL kernel in a WGSL plan");
        let si = *src_ix.entry(&k.source).or_insert_with(|| {
            srcs.push(&k.source);
            srcs.len() - 1
        });
        let reads: Vec<String> = k.reads.iter().map(|r| slot[&storage(*r)].to_string()).collect();
        if i > 0 {
            s.push(',');
        }
        write!(s, "[{si},[{}],{},{},{},{}]", reads.join(","), slot[&storage(k.node)], k.points, k.count, k.workgroup_size).unwrap();
    }
    s.push_str("],\"outputs\":{");
    for (i, ((name, _), id)) in outs.iter().zip(&oids).enumerate() {
        if i > 0 {
            s.push(',');
        }
        write!(s, "{}:{}", js(name), slot[&storage(*id)]).unwrap();
    }
    s.push_str("},\"shaders\":[");
    for (i, src) in srcs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&js(&web_wgsl(src)));
    }
    s.push_str("]}");
    (s, p.kernels.len())
}

/// tang's WGSL as browsers accept it: Tint reads `3.4028235e38` (f32::MAX
/// printed shortest) as an abstract float just above f32::MAX and rejects it;
/// the exact value is representable. (Upstream fix: print it exactly in
/// tang-compile's emitter.)
fn web_wgsl(src: &str) -> String {
    src.replace("3.4028235e38", "3.4028234663852886e38")
}

struct Models {
    tc: TConfig,
    fc: FConfig,
    vc: Config,
    ac: Vec<Vec<f32>>,
    fl: Vec<Vec<f32>>,
    voc: Vec<Vec<f32>>,
    /// f0 mu, sd, energy mu, sd
    norm: [f64; 4],
    /// flow m, s
    fnorm: [f64; 2],
}

fn load(rel: &Path, half: &str) -> Result<Models> {
    let tc = TConfig::by_name("base").unwrap();
    let fc = FConfig::by_name("base").unwrap();
    let vp = rel.join("vocoder_vocos_import.safetensors");
    let vc = if ckpt::has(&vp, "hsrc.w") || ckpt::has(&vp, "src.w") {
        return Err(anyhow!("vocoder with an F0 source: not supported by the web export"));
    } else {
        Config::v1()
    };
    let rd = |p: &str| -> Result<Vec<f64>> { Ok(std::fs::read_to_string(rel.join(p))?.split_whitespace().filter_map(|x| x.parse().ok()).collect()) };
    let n = rd("acoustic_norm.txt")?;
    let f = rd("flow_norm.txt")?;
    // `half`: comma-separated `model.param` prefixes whose weights are fp16-rounded
    let r = |k: &str, specs: Vec<model::ParamSpec>, v: Vec<Vec<f32>>| -> Vec<Vec<f32>> {
        v.into_iter().zip(&specs).map(|(x, s)| if is_f16(&format!("{k}.{}", s.name), half) { f16_round(&x) } else { x }).collect()
    };
    Ok(Models {
        ac: r("ac", tc.params(), ckpt::load_specs(&rel.join("acoustic_af_bon.safetensors"), &tc.params(), false)?.params),
        fl: r("fl", fc.params(), ckpt::load_specs(&rel.join("flow_student_af.safetensors"), &fc.params(), false)?.params),
        voc: r("voc", vc.params(), ckpt::load(&vp, &vc)?.params),
        tc,
        fc,
        vc,
        norm: [n[0], n[1], n[2], n[3]],
        fnorm: [f[0], f[1]],
    })
}

/// Reference mel `[n_mels, REF_T]` and pool weights `[REF_T]` exactly as `synth` builds them.
fn ref_tensors(dev: &Device, wav: &Path, n_mels: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let (x, sr) = ribt::data::read_wav(wav)?;
    if sr as usize != SR {
        return Err(anyhow!("{}: {sr} Hz", wav.display()));
    }
    let fr = x.len() / HOP;
    let off = if fr > REF_T + 30 { 30 } else { 0 };
    let len = (fr - off).min(REF_T);
    let vc = Config::v1();
    let fe = model::frontend_graph(&vc, 1, REF_T);
    let fin = fe.b.assemble(&[], None, &[]);
    let mut prog = Program::new(dev, &fe.b.g, &[fe.mel], &refs(&fin));
    let pad = vc.mel_pad();
    let clip = &x[off * HOP..(off + len) * HOP];
    let mut xp = vec![0.0f32; REF_T * HOP + 2 * pad];
    let p = dsp::reflect_pad(clip, pad);
    xp[..p.len()].copy_from_slice(&p);
    prog.write_input(dev, fe.b.input_index(fe.audio), &xp);
    prog.run(dev);
    let m = prog.read(dev, 0);
    let mut mel = vec![0.0f32; n_mels * REF_T];
    for c in 0..n_mels {
        mel[c * REF_T..c * REF_T + len].copy_from_slice(&m[c * REF_T..c * REF_T + len]);
    }
    let mut rw = vec![0.0f32; REF_T];
    rw[..len].iter_mut().for_each(|v| *v = 1.0 / len as f32);
    Ok((mel, rw))
}

fn enc_built(m: &Models, n: usize) -> (acoustic::EncGraph, Dims) {
    let d = Dims { batch: 1, n, t: 64, r: REF_T };
    (acoustic::enc_graph(&m.tc, d), d)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let rel = home(&arg(&a, "--rel").expect("--rel WEIGHTS_DIR (the release weights)"));
    let out = PathBuf::from(arg(&a, "--out").unwrap_or_else(|| "web/dist".into()));
    std::fs::create_dir_all(out.join("graphs"))?;
    lilt::accent::set("minnesota").map_err(|e| anyhow!(e))?;
    let dev = Device::new();
    let m = load(&rel, "")?;
    let (refmel, rw) = ref_tensors(&dev, &rel.join("ref_afheart.wav"), m.tc.n_mels)?;

    // --- weights: fp16 except the small prosody heads (SHIP_F16), one blob per
    // model; index `model.param` → [blob, byte offset, numel, "f16"|"f32"]
    let mut index = String::from("{");
    let mut files = String::new();
    for (prefix, specs, params) in [("ac", m.tc.params(), &m.ac), ("fl", m.fc.params(), &m.fl), ("voc", m.vc.params(), &m.voc)] {
        let mut blob: Vec<u8> = Vec::new();
        let (mut n16, mut n32) = (0usize, 0usize);
        for (s, p) in specs.iter().zip(params.iter()) {
            assert_eq!(s.numel(), p.len(), "{}", s.name);
            let name = format!("{prefix}.{}", s.name);
            let half = is_f16(&name, SHIP_F16);
            while blob.len() % 4 != 0 {
                blob.push(0);
            }
            if index.len() > 1 {
                index.push(',');
            }
            write!(index, "{}:[{},{},{},\"{}\"]", js(&name), js(prefix), blob.len(), p.len(), if half { "f16" } else { "f32" }).unwrap();
            for &x in p {
                if half {
                    blob.extend(precision::f16_bits(x as f64).to_le_bytes());
                } else {
                    blob.extend(x.to_le_bytes());
                }
            }
            if half { n16 += p.len() } else { n32 += p.len() }
        }
        let f = out.join(format!("{prefix}.bin"));
        std::fs::write(&f, &blob)?;
        if !files.is_empty() {
            files.push(',');
        }
        write!(files, "{}:{{\"file\":{},\"bytes\":{},\"sha256\":{}}}", js(prefix), js(&format!("{prefix}.bin")), blob.len(), js(&sha256(&f))).unwrap();
        eprintln!("{prefix}: {n16} params fp16 + {n32} f32, {:.1} MB", blob.len() as f64 / 1e6);
    }
    index.push('}');

    // --- graphs per bucket
    let mut consts = Consts::default();
    let mut graphs = String::new();
    let mut put = |name: String, json: (String, usize)| -> Result<()> {
        std::fs::write(out.join("graphs").join(format!("{name}.json")), &json.0)?;
        eprintln!("graph {name}: {} kernels, {:.0} KB", json.1, json.0.len() as f64 / 1e3);
        if !graphs.is_empty() {
            graphs.push(',');
        }
        graphs.push_str(&js(&name));
        Ok(())
    };
    let tspecs = m.tc.params();
    for &n in &NS {
        let (eg, _) = enc_built(&m, n);
        put(format!("enc_n{n}"), plan_json(&eg.b, &tspecs, "ac", &[("h", eg.enc.h), ("ld", eg.enc.ld), ("pitch", eg.enc.pitch), ("energy", eg.enc.energy), ("spk", eg.enc.spk_dec)], &mut consts))?;
        for &t in &TS {
            let dg = acoustic::dec_graph(&m.tc, Dims { batch: 1, n, t, r: REF_T });
            put(format!("dec_n{n}_t{t}"), plan_json(&dg.b, &tspecs, "ac", &[("mel", dg.mel), ("f0", dg.f0), ("hid", dg.hid)], &mut consts))?;
        }
    }
    let fspecs = m.fc.params();
    let vspecs = m.vc.params();
    for &t in &TS {
        let fg = flow::flow_graph(&m.fc, FDims { batch: 1, t }, false);
        put(format!("flow_t{t}"), plan_json(&fg.b, &fspecs, "fl", &[("v", fg.v)], &mut consts))?;
        let inf = model::infer_graph(&m.vc, 1, t);
        put(format!("voc_t{t}"), plan_json(&inf.b, &vspecs, "voc", &[("audio", inf.audio)], &mut consts))?;
    }
    std::fs::write(out.join("consts.f32.bin"), &consts.blob)?;
    eprintln!("consts: {:.1} MB", consts.blob.len() as f64 / 1e6);

    // --- manifest
    let temb = flow::time_feats(&m.fc, 0.0, 1.0); // nfe 1, shortcut: t = 0, d = 1
    let mut man = String::from("{\"name\":\"ribt\",\"sr\":24000,\"hop\":256,");
    write!(
        man,
        "\"ns\":{:?},\"ts\":{:?},\"vocab\":{},\"n_mels\":{},\"ref_t\":{REF_T},\"mel_floor\":{MEL_FLOOR},\"norm\":{:?},\"fnorm\":{:?},\"synth\":{{\"nfe\":1,\"shortcut\":true,\"temp\":0.3,\"seed\":0,\"pitch_st\":1.0,\"rate\":1.0,\"accent\":\"minnesota\"}},",
        NS, TS, m.tc.vocab, m.tc.n_mels, m.norm, m.fnorm
    )
    .unwrap();
    write!(man, "\"temb\":{},\"refmel\":{},\"rw\":{},", nums(&temb), nums(&refmel), nums(&rw)).unwrap();
    let cf = out.join("consts.f32.bin");
    write!(man, "\"consts\":{{\"file\":\"consts.f32.bin\",\"bytes\":{},\"sha256\":{}}},", consts.blob.len(), js(&sha256(&cf))).unwrap();
    write!(man, "\"weights\":{{{files}}},\"index\":{index},\"graphs\":[{graphs}]}}").unwrap();
    std::fs::write(out.join("manifest.json"), man)?;

    if let (Some(lines), Some(native)) = (arg(&a, "--check"), arg(&a, "--native")) {
        check(&dev, &rel, &refmel, &rw, &lines, Path::new(&native))?;
    }
    Ok(())
}

/// The browser's pipeline on the native device (same buckets, fp16 weights
/// as rounded on export, padded vocoder) vs `ribt-tts synth` output.
fn check(dev: &Device, rel: &Path, refmel: &[f32], rw: &[f32], lines: &str, native: &Path) -> Result<()> {
    for half in std::env::var("RIBT_HALF").unwrap_or_else(|_| format!(";{SHIP_F16}")).split(';') {
        let m = load(rel, half)?;
        let tc = &m.tc;
        for line in std::fs::read_to_string(lines)?.lines().filter(|l| !l.trim().is_empty()) {
            let (name, text) = line.split_once('\t').ok_or_else(|| anyhow!("bad line"))?;
            let (y, mel, durs) = pipeline(dev, &m, refmel, rw, text);
            if half == SHIP_F16 {
                // what the browser should reproduce up to WebGPU float differences
                let dir = Path::new(lines).parent().unwrap().join("fp16");
                std::fs::create_dir_all(&dir)?;
                let raw = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
                std::fs::write(dir.join(format!("{name}.audio.f32")), raw(&y))?;
                std::fs::write(dir.join(format!("{name}.mel.f32")), raw(&mel))?;
                std::fs::write(dir.join(format!("{name}.durs.txt")), durs.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" "))?;
            }
            let (want, _) = ribt::data::read_wav(&native.join(format!("{name}_ref.wav")))?;
            let (num, den) = y.iter().zip(&want).fold((0.0f64, 0.0f64), |(n, d), (a, b)| (n + ((a - b) as f64).powi(2), d + (*b as f64).powi(2)));
            eprintln!("check {name} ({}): {} vs {} samples, rel RMS {:.2e}", format!("fp16: {half}"), y.len(), want.len(), (num / den).sqrt());
            let _ = tc;
        }
    }
    Ok(())
}

fn bucket(v: usize, bs: &[usize]) -> usize {
    *bs.iter().find(|&&b| b >= v).unwrap_or_else(|| panic!("{v} over the largest bucket"))
}

/// Audio, valid mel `[n_mels, frames]` (after the flow) and token durations.
fn pipeline(dev: &Device, m: &Models, refmel: &[f32], rw: &[f32], text: &str) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    let tc = &m.tc;
    let (ps, _) = lilt::phonemize(text);
    let mut toks: Vec<usize> = vec![0];
    toks.extend(lilt::vocab::ids(&ps).iter().map(|&v| v as usize));
    toks.push(0);
    let n = bucket(toks.len(), &NS);
    let (eg, _) = enc_built(m, n);
    let ei = acoustic::assemble(&eg.b, &tc.params(), &m.ac, None, &[]);
    let mut enc = Program::new(dev, &eg.b.g, &[eg.enc.h, eg.enc.ld, eg.enc.pitch, eg.enc.energy, eg.enc.spk_dec], &refs(&ei));
    let mut oh = vec![0.0f32; tc.vocab * n];
    let mut tm = vec![0.0f32; n];
    for (i, &id) in toks.iter().enumerate() {
        oh[id * n + i] = 1.0;
        tm[i] = 1.0;
    }
    let dm = |b: &Built| -> HashMap<&'static str, usize> { b.inputs.iter().enumerate().filter_map(|(k, s)| if let Role::Data(n) = s.role { Some((n, k)) } else { None }).collect() };
    let em = dm(&eg.b);
    for (k, v) in [("onehot", &oh[..]), ("tmask", &tm), ("refmel", refmel), ("rw", rw)] {
        enc.write_input(dev, em[k], v);
    }
    enc.run(dev);
    let [h, ld, mut pp, ep, spk] = [0, 1, 2, 3, 4].map(|i| enc.read(dev, i));
    let d = (1.0 * std::f64::consts::LN_2 / 12.0 / m.norm[1]) as f32;
    pp.iter_mut().for_each(|v| *v += d);
    let durs: Vec<u32> = (0..toks.len()).map(|i| ((ld[i] as f64).exp() - 1.0).round().max(0.0) as u32).collect();
    let frames: usize = durs.iter().sum::<u32>() as usize;
    let t = bucket(frames, &TS);
    let dg = acoustic::dec_graph(tc, Dims { batch: 1, n, t, r: REF_T });
    let di = acoustic::assemble(&dg.b, &tc.params(), &m.ac, None, &[]);
    let mut dec = Program::new(dev, &dg.b.g, &[dg.mel, dg.f0, dg.hid], &refs(&di));
    let al = acoustic::align_matrix(&[&durs], n, t);
    let mut fm = vec![0.0f32; t];
    fm[..frames].iter_mut().for_each(|v| *v = 1.0);
    let dmp = dm(&dg.b);
    for (k, v) in [("h", &h[..]), ("pitch", &pp), ("energy", &ep), ("tmask", &tm), ("align", &al), ("fmask", &fm), ("spk_dec", &spk)] {
        dec.write_input(dev, dmp[k], v);
    }
    dec.run(dev);
    let (mel, hid) = (dec.read(dev, 0), dec.read(dev, 2));
    // flow, 1 NFE shortcut
    let fg = flow::flow_graph(&m.fc, FDims { batch: 1, t }, false);
    let fi = flow::assemble(&fg.b, &m.fc.params(), &m.fl, None, &[]);
    let mut fl = Program::new(dev, &fg.b.g, &[fg.v], &refs(&fi));
    let (fmn, fsd) = (m.fnorm[0], m.fnorm[1]);
    let nm = tc.n_mels;
    let mut mun = vec![0.0f32; nm * t];
    let mut x0 = vec![0.0f32; nm * t];
    let mut rng = Rng::new(0);
    for c in 0..nm {
        for f in 0..frames {
            mun[c * t + f] = ((mel[c * t + f] as f64 - fmn) / fsd) as f32;
            x0[c * t + f] = (0.3 * rng.normal()) as f32;
        }
    }
    let ab = flow::attn_bias(&[frames], t, None);
    let temb = flow::time_feats(&m.fc, 0.0, 1.0);
    let fmp = dm(&fg.b);
    for (k, v) in [("mu", &mun[..]), ("hc", &hid), ("fmask", &fm), ("abias", &ab), ("xt", &x0), ("temb", &temb)] {
        fl.write_input(dev, fmp[k], v);
    }
    fl.run(dev);
    let v = fl.read(dev, 0);
    // vocoder at bucket t: valid frames, MEL_FLOOR after
    let mut vin = vec![MEL_FLOOR; nm * t];
    for c in 0..nm {
        for f in 0..frames {
            vin[c * t + f] = ((x0[c * t + f] + v[c * t + f]) as f64 * fsd + fmn) as f32;
        }
    }
    let inf = model::infer_graph(&m.vc, 1, t);
    let vi = inf.b.assemble(&m.voc, None, &[]);
    let mut voc = Program::new(dev, &inf.b.g, &[inf.audio], &refs(&vi));
    voc.write_input(dev, inf.b.input_index(inf.mel), &vin);
    voc.run(dev);
    let mut y = voc.read(dev, 0);
    y.truncate(frames * HOP);
    let mel: Vec<f32> = (0..nm).flat_map(|c| vin[c * t..c * t + frames].to_vec()).collect();
    (y, mel, durs)
}
