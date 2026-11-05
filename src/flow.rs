//! V2 rung 2 (the ribt paper): a conditional flow-matching mel
//! decoder (OT-CFM, Matcha-TTS 2309.03199) on top of the frozen rung-1
//! acoustic model.
//!
//! The velocity net predicts `v(x_t, t, d | c)` over the normalized 100-bin
//! log-mel, frame layout `[B, C, T]`:
//! - conditioning `c`: rung 1's deterministic mel `μ` (normalized like `x`)
//!   and its decoder's last hidden state `[B, dd, T]` (length-regulated text
//!   states + pitch/energy + speaker, already mixed by 8 ConvNeXt blocks);
//! - time: host sinusoids of `t` (and of the shortcut step `d`, 0 for the
//!   teacher, see 2410.12557) → per-block linear AdaLN-zero shift/scale/gate.
//!   There is deliberately no shared time MLP: one embedding read by ~50
//!   modulation linears makes its gradient a 50-way sum that fuses into one
//!   kernel past Metal's 31-buffer limit (IR gap 22); reading the host
//!   features (data, no gradient) directly avoids the fan-in;
//! - body: ConvNeXt blocks (dw k7 → modulated LN → pw → GELU → pw, gated) with
//!   a multi-head self-attention block every `attn_every` blocks. Attention
//!   reads a host-built bias `[B, T, T]`: key padding, and optionally a
//!   block-causal chunk mask with lookahead (CosyVoice2-style streaming).
//!
//! The ODE (Euler), noise, `x_t`, targets and time features all live on the
//! host: each graph is one velocity evaluation (+ loss / AdamW in training).

use std::collections::HashMap;

use tang::Scalar;
use tang_compile::tensor::{self as ir_view, Monoid, TensorGraph, TensorId};

use crate::ir::{self, arg, lit, pick};
use crate::model::{self, Built, Config, ParamSpec, Role};

#[derive(Clone, Debug)]
pub struct FConfig {
    pub n_mels: usize,
    /// rung-1 decoder hidden width (conditioning)
    pub dc: usize,
    pub d: usize,
    pub ff: usize,
    pub k: usize,
    pub layers: usize,
    /// an attention block after every `attn_every` conv blocks (0: none)
    pub attn_every: usize,
    pub heads: usize,
    /// host time features (half t, half d)
    pub tf: usize,
}

impl FConfig {
    /// ~23M: d384, 12 ConvNeXt blocks (ff 1536, k7) + 4 attention blocks.
    pub fn base() -> Self {
        Self { n_mels: 100, dc: 384, d: 384, ff: 1536, k: 7, layers: 12, attn_every: 3, heads: 4, tf: 256 }
    }
    pub fn tiny() -> Self {
        Self { n_mels: 5, dc: 6, d: 8, ff: 12, k: 3, layers: 2, attn_every: 1, heads: 2, tf: 8 }
    }
    pub fn check() -> Self {
        Self { n_mels: 20, dc: 32, d: 64, ff: 128, k: 7, layers: 4, attn_every: 2, heads: 2, tf: 32 }
    }
    pub fn by_name(s: &str) -> Option<Self> {
        match s {
            "base" => Some(Self::base()),
            "tiny" => Some(Self::tiny()),
            "check" => Some(Self::check()),
            _ => None,
        }
    }

    fn blocks(&self) -> Vec<bool> {
        // true = attention block
        let mut v = Vec::new();
        for l in 0..self.layers {
            v.push(false);
            if self.attn_every > 0 && (l + 1) % self.attn_every == 0 {
                v.push(true);
            }
        }
        v
    }

    pub fn params(&self) -> Vec<ParamSpec> {
        let lin = |fan: usize| (1.0 / fan as f64).sqrt();
        let (d, te) = (self.d, self.tf);
        let mut v = vec![
            ParamSpec::normal("in.x.w", &[d, self.n_mels], lin(3 * self.n_mels)),
            ParamSpec::normal("in.mu.w", &[d, self.n_mels], lin(3 * self.n_mels)),
            ParamSpec::normal("in.h.w", &[d, self.dc], lin(3 * self.dc)),
            ParamSpec::zeros("in.b", &[d]),
        ];
        // shift, scale, gate stacked in one [3d, tf] linear: fewer parameter
        // tensors (AdamW's grad-norm partials fuse into one kernel that hits
        // Metal's 31-buffer limit past ~224 tensors)
        let modl = |v: &mut Vec<ParamSpec>, p: &dyn Fn(&str) -> String| {
            v.push(ParamSpec::zeros(&p("mod.w"), &[3 * d, te]));
            v.push(ParamSpec::zeros(&p("mod.b"), &[3 * d]));
            v.push(ParamSpec::ones(&p("ln.g"), &[d]));
            v.push(ParamSpec::zeros(&p("ln.b"), &[d]));
        };
        for (l, attn) in self.blocks().into_iter().enumerate() {
            let p = |s: &str| format!("b.{l}.{s}");
            modl(&mut v, &p);
            if attn {
                for w in ["q", "k", "v", "o"] {
                    v.push(ParamSpec::normal(&p(w), &[d, d], lin(d)));
                }
            } else {
                v.push(ParamSpec::normal(&p("dw.w"), &[d, self.k], lin(self.k)));
                v.push(ParamSpec::zeros(&p("dw.b"), &[d]));
                v.push(ParamSpec::normal(&p("pw1.w"), &[self.ff, d], lin(d)));
                v.push(ParamSpec::zeros(&p("pw1.b"), &[self.ff]));
                v.push(ParamSpec::normal(&p("pw2.w"), &[d, self.ff], lin(self.ff)));
                v.push(ParamSpec::zeros(&p("pw2.b"), &[d]));
            }
        }
        // final modulated LN (shift/scale only) and a zero output layer
        v.push(ParamSpec::zeros("out.mod.w", &[3 * d, te]));
        v.push(ParamSpec::zeros("out.mod.b", &[3 * d]));
        v.push(ParamSpec::ones("out.ln.g", &[d]));
        v.push(ParamSpec::zeros("out.ln.b", &[d]));
        v.push(ParamSpec::zeros("out.w", &[self.n_mels, d]));
        v.push(ParamSpec::zeros("out.b", &[self.n_mels]));
        v
    }

    pub fn n_params(&self) -> usize {
        self.params().iter().map(|p| p.numel()).sum()
    }
}

pub fn init_params(fc: &FConfig, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = model::Rng::new(seed);
    fc.params().iter().map(|s| s.init_data(&mut rng)).collect()
}

/// Host time features `[tf]`: sinusoids of 1000·t (first half) and 1000·d.
pub fn time_feats(fc: &FConfig, t: f64, d: f64) -> Vec<f32> {
    let h = fc.tf / 2;
    let mut out = vec![0.0f32; fc.tf];
    for (base, s) in [(0usize, t), (h, d)] {
        for i in 0..h / 2 {
            let w = 1.0 / 10000f64.powf(2.0 * i as f64 / h as f64);
            out[base + 2 * i] = (1000.0 * s * w).sin() as f32;
            out[base + 2 * i + 1] = (1000.0 * s * w).cos() as f32;
        }
    }
    out
}

/// Attention bias `[B, T, T]` (row = query, col = key): −1e4 for padded keys
/// (`len[b]`) and, with `chunk = Some((c, look))`, for keys more than `look`
/// chunks of `c` frames ahead of the query's chunk.
pub fn attn_bias(lens: &[usize], t: usize, chunk: Option<(usize, usize)>) -> Vec<f32> {
    let mut a = vec![0.0f32; lens.len() * t * t];
    for (b, &n) in lens.iter().enumerate() {
        for i in 0..t {
            for j in 0..t {
                let ok = j < n.max(1)
                    && match chunk {
                        Some((c, look)) => j / c <= i / c + look,
                        None => true,
                    };
                if !ok {
                    a[(b * t + i) * t + j] = -1e4;
                }
            }
        }
    }
    a
}

struct P {
    by_name: HashMap<String, TensorId>,
}

impl P {
    fn p(&self, n: &str) -> TensorId {
        *self.by_name.get(n).unwrap_or_else(|| panic!("no param {n}"))
    }
}

fn add(g: &mut TensorGraph, a: TensorId, b: TensorId) -> TensorId {
    g.map(&[a, b], |v| v[0] + v[1]).unwrap()
}

fn mask_mul(g: &mut TensorGraph, x: TensorId, m: TensorId) -> TensorId {
    let sh = ir::dims(g, x);
    g.comp(&sh, &[], vec![arg(x, pick(3, &[Some(0), Some(1), Some(2)])), arg(m, pick(3, &[Some(0), None, Some(2)]))], None, |v| vec![v[0] * v[1]])
        .unwrap()
}

/// Block modulation `W·e + b` → `[B, 3, d, 1]` (shift | scale | gate).
fn modv(g: &mut TensorGraph, p: &P, e: TensorId, pre: &str) -> TensorId {
    let y = ir::linear(g, e, p.p(&format!("{pre}.mod.w")));
    let y = ir::add_channel(g, y, p.p(&format!("{pre}.mod.b")));
    let y = ir::materialize(g, y);
    let [b, c, _] = ir::dims(g, y)[..] else { panic!() };
    g.reshape(y, &[b, 3, c / 3, 1]).unwrap()
}

/// Slice `k` of `m [B, 3, d, 1]` broadcast over time, in a rank-3 space
/// `(b, c, t)`: a constant index, always in range, so no guard. (A guarded
/// `k·d + c` channel offset into `[B, 3d, 1]` kept tang's scheduler from
/// tiling the pw2 / Wo GEMM that this epilogue follows: one thread per point,
/// 15–22 ms per block at B32 T384, ~75% of the forward.)
fn mod_view(_d: usize, k: usize) -> ir_view::View {
    ir::view(vec![ir::form(3, &[(0, 1)], 0), ir::form(3, &[], k as i64), ir::form(3, &[(1, 1)], 0), ir::form(3, &[], 0)], false)
}

/// `LN(x)·(1 + scale) + shift`, per-utterance scale/shift from `m`.
fn ada_ln(g: &mut TensorGraph, p: &P, x: TensorId, m: TensorId, pre: &str) -> TensorId {
    let y = ir::layer_norm(g, x, p.p(&format!("{pre}.ln.g")), p.p(&format!("{pre}.ln.b")), 1e-6);
    let dims = ir::dims(g, y);
    let d = dims[1];
    g.comp(&dims, &[], vec![arg(y, pick(3, &[Some(0), Some(1), Some(2)])), arg(m, mod_view(d, 1)), arg(m, mod_view(d, 0))], None, |v| vec![v[0] * (lit(1.0) + v[1]) + v[2]])
        .unwrap()
}

/// `x + gate·y`, gate = third slice of `m`.
fn gated_add(g: &mut TensorGraph, x: TensorId, y: TensorId, m: TensorId) -> TensorId {
    let dims = ir::dims(g, x);
    let id = pick(3, &[Some(0), Some(1), Some(2)]);
    g.comp(&dims, &[], vec![arg(x, id.clone()), arg(y, id), arg(m, mod_view(dims[1], 2))], None, |v| vec![v[0] + v[2] * v[1]])
        .unwrap()
}

/// Multi-head self-attention over frames, bias `ab [B, T, T]`.
fn mha(g: &mut TensorGraph, x: TensorId, p: &P, pre: &str, ab: TensorId, heads: usize) -> TensorId {
    let [b, d, n] = ir::dims(g, x)[..] else { panic!() };
    let dh = d / heads;
    let proj = |g: &mut TensorGraph, w: &str| {
        let y = ir::linear(g, x, p.p(&format!("{pre}.{w}")));
        g.reshape(y, &[b, heads, dh, n]).unwrap()
    };
    let q = proj(g, "q");
    let k = proj(g, "k");
    let v = proj(g, "v");
    let scale = 1.0 / (dh as f64).sqrt();
    let s = g
        .comp(&[b, heads, n, n], &[dh], vec![arg(q, pick(5, &[Some(0), Some(1), Some(4), Some(2)])), arg(k, pick(5, &[Some(0), Some(1), Some(4), Some(3)]))], Some(Monoid::sum()), |v| vec![v[0] * v[1]])
        .unwrap();
    let s = g
        .comp(&[b, heads, n, n], &[], vec![arg(s, pick(4, &[Some(0), Some(1), Some(2), Some(3)])), arg(ab, pick(4, &[Some(0), Some(2), Some(3)]))], None, |v| vec![v[0] * lit(scale) + v[1]])
        .unwrap();
    let m = g.comp(&[b, heads, n], &[n], vec![arg(s, pick(4, &[Some(0), Some(1), Some(2), Some(3)]))], Some(Monoid::max()), |v| vec![v[0]]).unwrap();
    let e = g
        .comp(&[b, heads, n, n], &[], vec![arg(s, pick(4, &[Some(0), Some(1), Some(2), Some(3)])), arg(m, pick(4, &[Some(0), Some(1), Some(2)]))], None, |v| vec![(v[0] - v[1]).exp()])
        .unwrap();
    let l = g.comp(&[b, heads, n], &[n], vec![arg(e, pick(4, &[Some(0), Some(1), Some(2), Some(3)]))], Some(Monoid::sum()), |v| vec![v[0]]).unwrap();
    let o = g
        .comp(&[b, heads, dh, n], &[n], vec![arg(e, pick(5, &[Some(0), Some(1), Some(3), Some(4)])), arg(v, pick(5, &[Some(0), Some(1), Some(2), Some(4)]))], Some(Monoid::sum()), |v| vec![v[0] * v[1]])
        .unwrap();
    let o = g
        .comp(&[b, heads, dh, n], &[], vec![arg(o, pick(4, &[Some(0), Some(1), Some(2), Some(3)])), arg(l, pick(4, &[Some(0), Some(1), Some(3)]))], None, |v| vec![v[0] / v[1]])
        .unwrap();
    let o = g.reshape(o, &[b, d, n]).unwrap();
    ir::linear(g, o, p.p(&format!("{pre}.o")))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FDims {
    pub batch: usize,
    pub t: usize,
}

pub struct FlowGraph {
    pub b: Built,
    pub fc: FConfig,
    pub params: Vec<TensorId>,
    pub xt: TensorId,
    pub mu: TensorId,
    pub hc: TensorId,
    pub temb: TensorId,
    pub fmask: TensorId,
    pub abias: TensorId,
    pub v: TensorId,
    /// training only: target velocity and per-frame weights; loss `[1]`
    pub ut: Option<TensorId>,
    pub fw: Option<TensorId>,
    pub loss: Option<TensorId>,
}

/// Velocity graph; with `train`, also the masked CFM loss
/// `Σ_{b,t} fw[b,t] · Σ_c (v − u)² / n_mels`.
pub fn flow_graph(fc: &FConfig, d: FDims, train: bool) -> FlowGraph {
    let mut b = Built::new_pub(&Config::tiny());
    let specs = fc.params();
    let ids: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| b.input(&s.shape, Role::Param(i))).collect();
    let p = P { by_name: specs.iter().zip(&ids).map(|(s, &id)| (s.name.clone(), id)).collect() };
    let (bb, t) = (d.batch, d.t);
    let xt = b.input(&[bb, fc.n_mels, t], Role::Data("xt"));
    let mu = b.input(&[bb, fc.n_mels, t], Role::Data("mu"));
    let hc = b.input(&[bb, fc.dc, t], Role::Data("hc"));
    let temb = b.input(&[bb, fc.tf, 1], Role::Data("temb"));
    let fmask = b.input(&[bb, 1, t], Role::Data("fmask"));
    let abias = b.input(&[bb, t, t], Role::Data("abias"));
    let (ut, fw) = if train {
        (Some(b.input(&[bb, fc.n_mels, t], Role::Data("ut"))), Some(b.input(&[bb, 1, t], Role::Data("fw"))))
    } else {
        (None, None)
    };
    let g = &mut b.g;
    // time features (data) feed every modulation linear directly
    let e = temb;
    // input projection
    let a1 = ir::linear(g, xt, p.p("in.x.w"));
    let a2 = ir::linear(g, mu, p.p("in.mu.w"));
    let a3 = ir::linear(g, hc, p.p("in.h.w"));
    let x = add(g, a1, a2);
    let x = add(g, x, a3);
    let x = ir::add_channel(g, x, p.p("in.b"));
    let x = mask_mul(g, x, fmask);
    let mut x = ir::materialize(g, x);
    for (l, attn) in fc.blocks().into_iter().enumerate() {
        let pre = format!("b.{l}");
        let q = |s: &str| p.p(&format!("{pre}.{s}"));
        let m = modv(g, &p, e, &pre);
        let y = if attn {
            let h = ada_ln(g, &p, x, m, &pre);
            mha(g, h, &p, &pre, abias, fc.heads)
        } else {
            let y = ir::dwconv(g, x, q("dw.w"), q("dw.b"));
            let y = ada_ln(g, &p, y, m, &pre);
            let y = ir::linear(g, y, q("pw1.w"));
            let y = ir::channel_map(g, y, &[q("pw1.b")], |x, b| ir::gelu(x + b[0]));
            let y = ir::linear(g, y, q("pw2.w"));
            ir::add_channel(g, y, q("pw2.b"))
        };
        let x2 = gated_add(g, x, y, m);
        let x2 = mask_mul(g, x2, fmask);
        x = ir::materialize(g, x2);
    }
    let mo = modv(g, &p, e, "out");
    let y = ada_ln(g, &p, x, mo, "out");
    let v = ir::linear(g, y, p.p("out.w"));
    let v = ir::add_channel(g, v, p.p("out.b"));
    let v = mask_mul(g, v, fmask);
    let loss = match (ut, fw) {
        (Some(ut), Some(fw)) => {
            let [_, c, _] = ir::dims(g, v)[..] else { panic!() };
            let inv = 1.0 / c as f64;
            // channels first, then everything (as acoustic::wsum)
            let s1 = g
                .comp(&[bb, 1, t], &[c], vec![arg(v, pick(4, &[Some(0), Some(3), Some(2)])), arg(ut, pick(4, &[Some(0), Some(3), Some(2)])), arg(fw, pick(4, &[Some(0), None, Some(2)]))], Some(Monoid::sum()), |a| {
                    let dd = a[0] - a[1];
                    vec![dd * dd * a[2] * lit(inv)]
                })
                .unwrap();
            Some(ir::sum_all2(g, s1, None, |v| v[0]))
        }
        _ => None,
    };
    FlowGraph { b, fc: fc.clone(), params: ids, xt, mu, hc, temb, fmask, abias, v, ut, fw, loss }
}

pub struct FlowTrain {
    pub fg: FlowGraph,
    pub m: Vec<TensorId>,
    pub v: Vec<TensorId>,
    pub hyper: TensorId,
    /// `[grad_norm, loss, p'…, m'…, v'…, partials…]`
    pub outputs: Vec<TensorId>,
}

pub fn train_graph(fc: &FConfig, d: FDims) -> FlowTrain {
    let mut fg = flow_graph(fc, d, true);
    let specs = fc.params();
    let m: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| fg.b.input(&s.shape, Role::AdamM(i))).collect();
    let v: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| fg.b.input(&s.shape, Role::AdamV(i))).collect();
    let hyper = fg.b.input(&[6], Role::Hyper);
    let loss = fg.loss.unwrap();
    let g = &mut fg.b.g;
    let grads = g.backward(loss, &fg.params).unwrap();
    let ad = model::adamw_betas(g, &specs, &fg.params, &grads, &m, &v, hyper, 0.9, 0.99);
    let mut outputs = vec![ad.grad_norm, loss];
    outputs.extend(&ad.new_p);
    outputs.extend(&ad.new_m);
    outputs.extend(&ad.new_v);
    outputs.extend(ad.partials);
    FlowTrain { fg, m, v, hyper, outputs }
}

pub fn assemble(b: &Built, specs: &[ParamSpec], params: &[Vec<f32>], adam: Option<(&[Vec<f32>], &[Vec<f32>])>, data: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
    crate::align::assemble(b, specs, params, adam, data)
}
