//! V2 acoustic model, first rung (the ribt paper): Kokoro tokens →
//! log-mel (+ frame F0 for the V1.2 vocoder's source input). FastSpeech2
//! shape, deterministic L1 baseline that the flow decoder will replace.
//!
//! - text: one-hot `[B, V, N]` × embedding + sinusoid positions, pre-LN
//!   transformer blocks (MHA + conv FFN), padded keys masked by a −1e4 bias;
//! - speaker: a reference encoder (conv × 2 + masked mean pool over a mel crop
//!   of *another* utterance of the speaker) → added to the encoder output and
//!   the decoder input; LibriTTS test-clean speakers are unseen, so a lookup
//!   table could not be scored there;
//! - variance adaptor: duration (ln(d+1)), token pitch and token energy
//!   predictors (conv-ReLU-LN × 2 → 1); pitch/energy enter through k3 convs
//!   (FastPitch), teacher-forced in training;
//! - length regulator: **no gather in the IR**, so the host builds the hard
//!   alignment `A[b,n,t]` from durations and the graph does
//!   `x[b,c,t] = Σ_n h[b,c,n]·A[b,n,t]` (a banded one-hot matmul);
//! - decoder: ConvNeXt blocks (dwconv k7 → LN → pw → GELU → pw, layer
//!   scale), final LN, linear → n_mels log-mel and 2 F0 channels
//!   (normalized ln F0, voicing logit).
//!
//! Layout `[B, C, X]` throughout (X = tokens N or frames T), as `ir.rs`.

use std::collections::HashMap;

use tang::Scalar;
use tang_compile::tensor::{Monoid, TensorGraph, TensorId};
use tang_expr::ExprId;

use crate::align::KOKORO_VOCAB;
use crate::ir::{self, arg, form, lit, pick, view};
use crate::model::{self, Built, Config, ParamSpec, Role};

#[derive(Clone, Debug)]
pub struct TConfig {
    pub vocab: usize,
    pub n_mels: usize,
    pub d: usize,
    pub heads: usize,
    pub enc_layers: usize,
    pub enc_ff: usize,
    pub enc_k: usize,
    pub var_h: usize,
    pub var_k: usize,
    pub ref_h: usize,
    pub dd: usize,
    pub dec_layers: usize,
    pub dec_ff: usize,
    pub dec_k: usize,
    /// loss weights: duration, pitch, energy, frame F0 (L1 + BCE)
    pub w_var: f64,
    pub w_f0: f64,
}

impl TConfig {
    /// ~19M params: enc d256 × 4 (2 heads, conv FFN 1024 k3), dec d384 × 8
    /// ConvNeXt (ff 1536, k7).
    pub fn base() -> Self {
        Self {
            vocab: KOKORO_VOCAB,
            n_mels: 100,
            d: 256,
            heads: 2,
            enc_layers: 4,
            enc_ff: 1024,
            enc_k: 3,
            var_h: 256,
            var_k: 3,
            ref_h: 256,
            dd: 384,
            dec_layers: 8,
            dec_ff: 1536,
            dec_k: 7,
            w_var: 1.0,
            w_f0: 1.0,
        }
    }

    /// For the f64 interpreter.
    pub fn tiny() -> Self {
        Self {
            vocab: 7,
            n_mels: 5,
            d: 8,
            heads: 2,
            enc_layers: 1,
            enc_ff: 12,
            enc_k: 3,
            var_h: 6,
            var_k: 3,
            ref_h: 6,
            dd: 10,
            dec_layers: 1,
            dec_ff: 14,
            dec_k: 3,
            w_var: 1.0,
            w_f0: 1.0,
        }
    }

    /// Big enough for the tiled GPU paths, small enough for the interpreter.
    pub fn check() -> Self {
        Self {
            vocab: 40,
            n_mels: 20,
            d: 64,
            heads: 2,
            enc_layers: 2,
            enc_ff: 128,
            enc_k: 3,
            var_h: 32,
            var_k: 3,
            ref_h: 32,
            dd: 64,
            dec_layers: 2,
            dec_ff: 128,
            dec_k: 7,
            w_var: 1.0,
            w_f0: 1.0,
        }
    }

    pub fn by_name(s: &str) -> Option<Self> {
        match s {
            "base" => Some(Self::base()),
            "tiny" => Some(Self::tiny()),
            "check" => Some(Self::check()),
            _ => None,
        }
    }

    pub fn params(&self) -> Vec<ParamSpec> {
        let lin = |fan: usize| (1.0 / fan as f64).sqrt();
        let he = |fan: usize| (2.0 / fan as f64).sqrt();
        let (d, dd) = (self.d, self.dd);
        let res = (2.0 * self.enc_layers as f64).sqrt();
        let mut v = vec![
            ParamSpec::normal("enc.emb", &[d, self.vocab], 0.5),
            ParamSpec::normal("ref.c1.w", &[self.ref_h, self.n_mels, 3], he(3 * self.n_mels)),
            ParamSpec::zeros("ref.c1.b", &[self.ref_h]),
            ParamSpec::normal("ref.c2.w", &[self.ref_h, self.ref_h, 3], he(3 * self.ref_h)),
            ParamSpec::zeros("ref.c2.b", &[self.ref_h]),
            ParamSpec::normal("ref.enc.w", &[d, self.ref_h], lin(self.ref_h)),
            ParamSpec::zeros("ref.enc.b", &[d]),
            ParamSpec::normal("ref.dec.w", &[dd, self.ref_h], lin(self.ref_h)),
            ParamSpec::zeros("ref.dec.b", &[dd]),
        ];
        for l in 0..self.enc_layers {
            let p = |s: &str| format!("enc.{l}.{s}");
            v.push(ParamSpec::ones(&p("ln1.g"), &[d]));
            v.push(ParamSpec::zeros(&p("ln1.b"), &[d]));
            for w in ["q", "k", "v"] {
                v.push(ParamSpec::normal(&p(w), &[d, d], lin(d)));
            }
            v.push(ParamSpec::normal(&p("o"), &[d, d], lin(d) / res));
            v.push(ParamSpec::ones(&p("ln2.g"), &[d]));
            v.push(ParamSpec::zeros(&p("ln2.b"), &[d]));
            v.push(ParamSpec::normal(&p("ff1.w"), &[self.enc_ff, d, self.enc_k], he(d * self.enc_k)));
            v.push(ParamSpec::zeros(&p("ff1.b"), &[self.enc_ff]));
            v.push(ParamSpec::normal(&p("ff2.w"), &[d, self.enc_ff, self.enc_k], lin(self.enc_ff * self.enc_k) / res));
            v.push(ParamSpec::zeros(&p("ff2.b"), &[d]));
        }
        v.push(ParamSpec::ones("enc.lnf.g", &[d]));
        v.push(ParamSpec::zeros("enc.lnf.b", &[d]));
        for pr in ["dur", "pitch", "energy"] {
            let p = |s: &str| format!("{pr}.{s}");
            let h = self.var_h;
            v.push(ParamSpec::normal(&p("c1.w"), &[h, d, self.var_k], he(d * self.var_k)));
            v.push(ParamSpec::zeros(&p("c1.b"), &[h]));
            v.push(ParamSpec::ones(&p("ln1.g"), &[h]));
            v.push(ParamSpec::zeros(&p("ln1.b"), &[h]));
            v.push(ParamSpec::normal(&p("c2.w"), &[h, h, self.var_k], he(h * self.var_k)));
            v.push(ParamSpec::zeros(&p("c2.b"), &[h]));
            v.push(ParamSpec::ones(&p("ln2.g"), &[h]));
            v.push(ParamSpec::zeros(&p("ln2.b"), &[h]));
            v.push(ParamSpec::normal(&p("out.w"), &[1, h], lin(h)));
            v.push(ParamSpec::zeros(&p("out.b"), &[1]));
        }
        for e in ["pemb", "eemb"] {
            v.push(ParamSpec::normal(&format!("{e}.w"), &[d, 1, 3], 0.3));
            v.push(ParamSpec::zeros(&format!("{e}.b"), &[d]));
        }
        v.push(ParamSpec::normal("dec.in.w", &[dd, d], lin(d)));
        v.push(ParamSpec::zeros("dec.in.b", &[dd]));
        for l in 0..self.dec_layers {
            let p = |s: &str| format!("dec.{l}.{s}");
            v.push(ParamSpec::normal(&p("dw.w"), &[dd, self.dec_k], lin(self.dec_k)));
            v.push(ParamSpec::zeros(&p("dw.b"), &[dd]));
            v.push(ParamSpec::ones(&p("ln.g"), &[dd]));
            v.push(ParamSpec::zeros(&p("ln.b"), &[dd]));
            v.push(ParamSpec::normal(&p("pw1.w"), &[self.dec_ff, dd], lin(dd)));
            v.push(ParamSpec::zeros(&p("pw1.b"), &[self.dec_ff]));
            v.push(ParamSpec::normal(&p("pw2.w"), &[dd, self.dec_ff], lin(self.dec_ff)));
            v.push(ParamSpec::zeros(&p("pw2.b"), &[dd]));
            v.push(ParamSpec::constant(&p("gamma"), &[dd], 0.1));
        }
        v.push(ParamSpec::ones("dec.lnf.g", &[dd]));
        v.push(ParamSpec::zeros("dec.lnf.b", &[dd]));
        v.push(ParamSpec::normal("mel.w", &[self.n_mels, dd], lin(dd)));
        v.push(ParamSpec::constant("mel.b", &[self.n_mels], -5.0));
        v.push(ParamSpec::normal("f0.w", &[2, dd], lin(dd)));
        v.push(ParamSpec::zeros("f0.b", &[2]));
        v
    }

    pub fn n_params(&self) -> usize {
        self.params().iter().map(|p| p.numel()).sum()
    }
}

pub fn init_params(tc: &TConfig, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = model::Rng::new(seed);
    tc.params().iter().map(|s| s.init_data(&mut rng)).collect()
}

/// Sinusoid positions `[C, X]` (Vaswani), row-major.
pub fn sinusoid(c: usize, x: usize) -> Vec<f64> {
    let mut out = vec![0.0; c * x];
    for i in 0..c / 2 {
        let w = 1.0 / 10000f64.powf(2.0 * i as f64 / c as f64);
        for t in 0..x {
            out[(2 * i) * x + t] = (t as f64 * w).sin();
            out[(2 * i + 1) * x + t] = (t as f64 * w).cos();
        }
    }
    out
}

// ---------------------------------------------------------------------------
// graph helpers

fn relu(x: ExprId) -> ExprId {
    ExprId::select(x, x, ExprId::ZERO)
}

/// `x[b,c,t] · m[b,0,t]`.
fn mask_mul(g: &mut TensorGraph, x: TensorId, m: TensorId) -> TensorId {
    let sh = ir::dims(g, x);
    g.comp(&sh, &[], vec![arg(x, pick(3, &[Some(0), Some(1), Some(2)])), arg(m, pick(3, &[Some(0), None, Some(2)]))], None, |v| vec![v[0] * v[1]])
        .unwrap()
}

/// `x[b,c,t] + v[b,c,0]` (a per-utterance vector broadcast over time).
fn add_bc(g: &mut TensorGraph, x: TensorId, v: TensorId) -> TensorId {
    let sh = ir::dims(g, x);
    g.comp(&sh, &[], vec![arg(x, pick(3, &[Some(0), Some(1), Some(2)])), arg(v, pick(3, &[Some(0), Some(1), None]))], None, |a| vec![a[0] + a[1]])
        .unwrap()
}

/// `x[b,c,t] + pe[c,t]`.
fn add_ct(g: &mut TensorGraph, x: TensorId, pe: TensorId) -> TensorId {
    let sh = ir::dims(g, x);
    g.comp(&sh, &[], vec![arg(x, pick(3, &[Some(0), Some(1), Some(2)])), arg(pe, pick(3, &[Some(1), Some(2)]))], None, |a| vec![a[0] + a[1]])
        .unwrap()
}

fn add(g: &mut TensorGraph, a: TensorId, b: TensorId) -> TensorId {
    g.map(&[a, b], |v| v[0] + v[1]).unwrap()
}

/// Per-head projection: a plain per-token linear `[B, d, N]` (its weight
/// gradient is the fast GEMM-shaped reduction), then a row-major reshape to
/// `[B, H, dh, N]`. Reading W through an `h·dh + c` view instead made each
/// weight gradient a 130 ms kernel (16 of them at B16).
fn head_proj(g: &mut TensorGraph, x: TensorId, w: TensorId, heads: usize) -> TensorId {
    let [b, d, n] = ir::dims(g, x)[..] else { panic!() };
    let y = ir::linear(g, x, w);
    g.reshape(y, &[b, heads, d / heads, n]).unwrap()
}

/// Multi-head self-attention over `x [B, d, N]`; `kb [B,1,N]` is the key
/// bias (0 valid, −1e4 padded).
fn mha(g: &mut TensorGraph, x: TensorId, wq: TensorId, wk: TensorId, wv: TensorId, wo: TensorId, kb: TensorId, heads: usize) -> TensorId {
    let [b, d, n] = ir::dims(g, x)[..] else { panic!() };
    let dh = d / heads;
    let q = head_proj(g, x, wq, heads);
    let k = head_proj(g, x, wk, heads);
    let v = head_proj(g, x, wv, heads);
    let scale = 1.0 / (dh as f64).sqrt();
    // s[b,h,i,j] = scale·Σ_c q[b,h,c,i]·k[b,h,c,j]; space (b, h, i, j, c)
    let s = g
        .comp(
            &[b, heads, n, n],
            &[dh],
            vec![arg(q, pick(5, &[Some(0), Some(1), Some(4), Some(2)])), arg(k, pick(5, &[Some(0), Some(1), Some(4), Some(3)]))],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let s = g
        .comp(
            &[b, heads, n, n],
            &[],
            vec![arg(s, pick(4, &[Some(0), Some(1), Some(2), Some(3)])), arg(kb, pick(4, &[Some(0), None, Some(3)]))],
            None,
            |v| vec![v[0] * lit(scale) + v[1]],
        )
        .unwrap();
    // row max (a stabilizer; its gradient cancels exactly)
    let m = g.comp(&[b, heads, n], &[n], vec![arg(s, pick(4, &[Some(0), Some(1), Some(2), Some(3)]))], Some(Monoid::max()), |v| vec![v[0]]).unwrap();
    let p = g
        .comp(
            &[b, heads, n, n],
            &[],
            vec![arg(s, pick(4, &[Some(0), Some(1), Some(2), Some(3)])), arg(m, pick(4, &[Some(0), Some(1), Some(2)]))],
            None,
            |v| vec![(v[0] - v[1]).exp()],
        )
        .unwrap();
    let l = g.comp(&[b, heads, n], &[n], vec![arg(p, pick(4, &[Some(0), Some(1), Some(2), Some(3)]))], Some(Monoid::sum()), |v| vec![v[0]]).unwrap();
    // o[b,h,c,i] = Σ_j p[b,h,i,j]·v[b,h,c,j]; space (b, h, c, i, j)
    let o = g
        .comp(
            &[b, heads, dh, n],
            &[n],
            vec![arg(p, pick(5, &[Some(0), Some(1), Some(3), Some(4)])), arg(v, pick(5, &[Some(0), Some(1), Some(2), Some(4)]))],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let o = g
        .comp(
            &[b, heads, dh, n],
            &[],
            vec![arg(o, pick(4, &[Some(0), Some(1), Some(2), Some(3)])), arg(l, pick(4, &[Some(0), Some(1), Some(3)]))],
            None,
            |v| vec![v[0] / v[1]],
        )
        .unwrap();
    // heads back to [B, d, N] (row-major: channel h·dh + c), then Wo
    let o = g.reshape(o, &[b, d, n]).unwrap();
    ir::linear(g, o, wo)
}

/// `x[b,c,t] = Σ_n h[b,c,n]·A[b,n,t]` (the length regulator).
pub fn expand(g: &mut TensorGraph, h: TensorId, a: TensorId) -> TensorId {
    let [b, c, n] = ir::dims(g, h)[..] else { panic!() };
    let t = ir::dims(g, a)[2];
    g.comp(&[b, c, t], &[n], vec![arg(h, pick(4, &[Some(0), Some(1), Some(3)])), arg(a, pick(4, &[Some(0), Some(3), Some(2)]))], Some(Monoid::sum()), |v| vec![v[0] * v[1]])
        .unwrap()
}

/// `Σ_{b,c,x} w[b,0,x]·f(a, b?)` in two stages (channels first), as `[1]`.
fn wsum(g: &mut TensorGraph, a: TensorId, b2: Option<TensorId>, w: TensorId, f: impl Fn(&[ExprId]) -> ExprId) -> TensorId {
    let [b, c, x] = ir::dims(g, a)[..] else { panic!() };
    // space (b, one, x, c)
    let mut args = vec![arg(a, pick(4, &[Some(0), Some(3), Some(2)]))];
    if let Some(b2) = b2 {
        args.push(arg(b2, pick(4, &[Some(0), Some(3), Some(2)])));
    }
    args.push(arg(w, pick(4, &[Some(0), None, Some(2)])));
    let s1 = g.comp(&[b, 1, x], &[c], args, Some(Monoid::sum()), |v| {
        let n = v.len();
        vec![f(&v[..n - 1]) * v[n - 1]]
    })
    .unwrap();
    ir::sum_all2(g, s1, None, |v| v[0])
}

/// Named parameters as graph inputs.
pub struct Params {
    pub ids: Vec<TensorId>,
    pub by_name: HashMap<String, TensorId>,
}

impl Params {
    fn new(b: &mut Built, specs: &[ParamSpec]) -> Self {
        let ids: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| b.input(&s.shape, Role::Param(i))).collect();
        let by_name = specs.iter().zip(&ids).map(|(s, &id)| (s.name.clone(), id)).collect();
        Self { ids, by_name }
    }
    fn p(&self, n: &str) -> TensorId {
        *self.by_name.get(n).unwrap_or_else(|| panic!("no param {n}"))
    }
}

pub struct EncOut {
    /// encoder output + speaker, masked `[B, d, N]`
    pub h: TensorId,
    /// predictions `[B, 1, N]`: ln(d+1), token pitch, token energy
    pub ld: TensorId,
    pub pitch: TensorId,
    pub energy: TensorId,
    /// speaker vectors `[B, d, 1]`, `[B, dd, 1]`
    pub spk_dec: TensorId,
}

struct EncIn {
    onehot: TensorId,
    tmask: TensorId,
    refmel: TensorId,
    rw: TensorId,
    pe: TensorId,
}

fn predictor(g: &mut TensorGraph, p: &Params, name: &str, h: TensorId, tmask: TensorId, ctx: Option<TensorId>) -> TensorId {
    let q = |s: &str| p.p(&format!("{name}.{s}"));
    // AdaLN-zero (§25): LN(x)·(1 + γ(c)) + β(c), γ/β linear in the context
    // state c with zero-initialised weights, so an untrained context path is
    // exactly the base model
    let modulate = |g: &mut TensorGraph, x: TensorId, j: usize| -> TensorId {
        let Some(c) = ctx else { return x };
        let ga = ir::linear(g, c, p.p(&format!("ctx.{name}.g{j}")));
        let be = ir::linear(g, c, p.p(&format!("ctx.{name}.b{j}")));
        g.map(&[x, ga, be], |v| v[0] * (lit(1.0) + v[1]) + v[2]).unwrap()
    };
    let x = ir::conv1d(g, h, q("c1.w"), q("c1.b"));
    let x = g.map(&[x], |v| relu(v[0])).unwrap();
    let x = ir::layer_norm(g, x, q("ln1.g"), q("ln1.b"), 1e-5);
    let x = modulate(g, x, 1);
    let x = ir::conv1d(g, x, q("c2.w"), q("c2.b"));
    let x = g.map(&[x], |v| relu(v[0])).unwrap();
    let x = ir::layer_norm(g, x, q("ln2.g"), q("ln2.b"), 1e-5);
    let x = modulate(g, x, 2);
    let x = ir::linear(g, x, q("out.w"));
    let x = ir::add_channel(g, x, q("out.b"));
    mask_mul(g, x, tmask)
}

fn encoder(g: &mut TensorGraph, tc: &TConfig, p: &Params, i: &EncIn, cx: Option<&CtxIn>) -> EncOut {
    // reference encoder → speaker vectors
    let r = g.map(&[i.refmel], |v| (v[0] + lit(5.0)) * lit(0.4)).unwrap();
    let r = ir::conv1d(g, r, p.p("ref.c1.w"), p.p("ref.c1.b"));
    let r = g.map(&[r], |v| relu(v[0])).unwrap();
    let r = ir::conv1d(g, r, p.p("ref.c2.w"), p.p("ref.c2.b"));
    let r = g.map(&[r], |v| relu(v[0])).unwrap();
    let [b, rh, rt] = ir::dims(g, r)[..] else { panic!() };
    // masked mean pool: rw[b,0,t] = mask / count
    let pooled = g
        .comp(&[b, rh, 1], &[rt], vec![arg(r, pick(4, &[Some(0), Some(1), Some(3)])), arg(i.rw, pick(4, &[Some(0), None, Some(3)]))], Some(Monoid::sum()), |v| vec![v[0] * v[1]])
        .unwrap();
    let se = ir::linear(g, pooled, p.p("ref.enc.w"));
    let se = ir::add_channel(g, se, p.p("ref.enc.b"));
    let sd = ir::linear(g, pooled, p.p("ref.dec.w"));
    let spk_dec = ir::add_channel(g, sd, p.p("ref.dec.b"));
    // text encoder
    let kb = g.map(&[i.tmask], |v| (v[0] - lit(1.0)) * lit(1e4)).unwrap();
    let x = ir::linear(g, i.onehot, p.p("enc.emb"));
    let mut x = add_ct(g, x, i.pe);
    for l in 0..tc.enc_layers {
        let q = |s: &str| p.p(&format!("enc.{l}.{s}"));
        let h = ir::layer_norm(g, x, q("ln1.g"), q("ln1.b"), 1e-5);
        let a = mha(g, h, q("q"), q("k"), q("v"), q("o"), kb, tc.heads);
        let x1 = add(g, x, a);
        let x1 = mask_mul(g, x1, i.tmask);
        let x1 = ir::materialize(g, x1);
        let h = ir::layer_norm(g, x1, q("ln2.g"), q("ln2.b"), 1e-5);
        let f = ir::conv1d(g, h, q("ff1.w"), q("ff1.b"));
        let f = g.map(&[f], |v| relu(v[0])).unwrap();
        let f = ir::conv1d(g, f, q("ff2.w"), q("ff2.b"));
        let x2 = add(g, x1, f);
        let x2 = mask_mul(g, x2, i.tmask);
        // a buffer per block: unbroken map chains fuse the backward into
        // kernels reading > 31 buffers (Metal's argument-table limit)
        x = ir::materialize(g, x2);
    }
    let h = ir::layer_norm(g, x, p.p("enc.lnf.g"), p.p("enc.lnf.b"), 1e-5);
    let h = add_bc(g, h, se);
    let h = mask_mul(g, h, i.tmask);
    let h = ir::materialize(g, h);
    let c = cx.map(|cx| ctx_state(g, p, h, i.onehot, i.tmask, cx));
    let ld = predictor(g, p, "dur", h, i.tmask, c);
    let pitch = predictor(g, p, "pitch", h, i.tmask, c);
    let energy = predictor(g, p, "energy", h, i.tmask, c);
    EncOut { h, ld, pitch, energy, spk_dec }
}

/// Decoder: token states + pitch/energy → frames (via `align`) → log-mel
/// `[B, n_mels, T]` and F0 channels `[B, 2, T]`.
fn decoder(
    g: &mut TensorGraph,
    tc: &TConfig,
    p: &Params,
    h: TensorId,
    pitch: TensorId,
    energy: TensorId,
    tmask: TensorId,
    align: TensorId,
    fmask: TensorId,
    spk_dec: TensorId,
) -> (TensorId, TensorId, TensorId) {
    let pe = ir::conv1d(g, pitch, p.p("pemb.w"), p.p("pemb.b"));
    let ee = ir::conv1d(g, energy, p.p("eemb.w"), p.p("eemb.b"));
    let x = add(g, h, pe);
    let x = add(g, x, ee);
    let x = mask_mul(g, x, tmask);
    let x = expand(g, x, align);
    let x = ir::linear(g, x, p.p("dec.in.w"));
    let x = ir::add_channel(g, x, p.p("dec.in.b"));
    let x = add_bc(g, x, spk_dec);
    let x = mask_mul(g, x, fmask);
    let mut x = ir::materialize(g, x);
    for l in 0..tc.dec_layers {
        let q = |s: &str| p.p(&format!("dec.{l}.{s}"));
        let y = ir::dwconv(g, x, q("dw.w"), q("dw.b"));
        let y = ir::layer_norm(g, y, q("ln.g"), q("ln.b"), 1e-6);
        let y = ir::linear(g, y, q("pw1.w"));
        let y = ir::channel_map(g, y, &[q("pw1.b")], |x, b| ir::gelu(x + b[0]));
        let y = ir::linear(g, y, q("pw2.w"));
        let y = ir::channel_map(g, y, &[q("pw2.b"), q("gamma")], |x, b| (x + b[0]) * b[1]);
        let x2 = add(g, x, y);
        let x2 = mask_mul(g, x2, fmask);
        x = ir::materialize(g, x2);
    }
    let x = ir::layer_norm(g, x, p.p("dec.lnf.g"), p.p("dec.lnf.b"), 1e-6);
    let mel = ir::linear(g, x, p.p("mel.w"));
    let mel = ir::add_channel(g, mel, p.p("mel.b"));
    let f0 = ir::linear(g, x, p.p("f0.w"));
    let f0 = ir::add_channel(g, f0, p.p("f0.b"));
    (mel, f0, x)
}

/// Shapes of one bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dims {
    pub batch: usize,
    pub n: usize,
    pub t: usize,
    /// reference crop frames
    pub r: usize,
}

pub struct Inputs {
    pub onehot: TensorId,
    pub tmask: TensorId,
    pub refmel: TensorId,
    pub rw: TensorId,
}

fn enc_inputs(b: &mut Built, tc: &TConfig, d: Dims) -> (Inputs, TensorId) {
    let onehot = b.input(&[d.batch, tc.vocab, d.n], Role::Data("onehot"));
    let tmask = b.input(&[d.batch, 1, d.n], Role::Data("tmask"));
    let refmel = b.input(&[d.batch, tc.n_mels, d.r], Role::Data("refmel"));
    let rw = b.input(&[d.batch, 1, d.r], Role::Data("rw"));
    let pe = b.constant(&[tc.d, d.n], sinusoid(tc.d, d.n));
    (Inputs { onehot, tmask, refmel, rw }, pe)
}

/// Forward + losses (no optimizer); the train graph adds AdamW.
pub struct TtsGraph {
    pub b: Built,
    pub tc: TConfig,
    pub dims: Dims,
    pub params: Vec<TensorId>,
    pub inp: Inputs,
    /// data inputs of the decoder / losses
    pub align: TensorId,
    pub fmask: TensorId,
    pub p_tok: TensorId,
    pub e_tok: TensorId,
    pub ld_tok: TensorId,
    pub tw: TensorId,
    pub mel: TensorId,
    pub lf0: TensorId,
    pub vuv: TensorId,
    pub fw: TensorId,
    pub vw: TensorId,
    pub enc: EncOut,
    pub mel_hat: TensorId,
    pub f0_hat: TensorId,
    /// `[total, mel L1, dur, pitch, energy, f0 L1, vuv BCE]`
    pub losses: Vec<TensorId>,
}

pub fn tts_graph(tc: &TConfig, d: Dims) -> TtsGraph {
    let mut b = Built::new_pub(&Config::tiny());
    let specs = tc.params();
    let p = Params::new(&mut b, &specs);
    let (inp, pe) = enc_inputs(&mut b, tc, d);
    let align = b.input(&[d.batch, d.n, d.t], Role::Data("align"));
    let fmask = b.input(&[d.batch, 1, d.t], Role::Data("fmask"));
    let p_tok = b.input(&[d.batch, 1, d.n], Role::Data("p_tok"));
    let e_tok = b.input(&[d.batch, 1, d.n], Role::Data("e_tok"));
    let ld_tok = b.input(&[d.batch, 1, d.n], Role::Data("ld_tok"));
    let tw = b.input(&[d.batch, 1, d.n], Role::Data("tw"));
    let mel = b.input(&[d.batch, tc.n_mels, d.t], Role::Data("mel"));
    let lf0 = b.input(&[d.batch, 1, d.t], Role::Data("lf0"));
    let vuv = b.input(&[d.batch, 1, d.t], Role::Data("vuv"));
    let fw = b.input(&[d.batch, 1, d.t], Role::Data("fw"));
    let vw = b.input(&[d.batch, 1, d.t], Role::Data("vw"));
    let g = &mut b.g;
    let ei = EncIn { onehot: inp.onehot, tmask: inp.tmask, refmel: inp.refmel, rw: inp.rw, pe };
    let enc = encoder(g, tc, &p, &ei, None);
    let (mel_hat, f0_hat, _) = decoder(g, tc, &p, enc.h, p_tok, e_tok, inp.tmask, align, fmask, enc.spk_dec);
    let inv_m = 1.0 / tc.n_mels as f64;
    let l_mel = wsum(g, mel_hat, Some(mel), fw, |v| ir::abs(v[0] - v[1]) * lit(inv_m));
    let l_dur = wsum(g, enc.ld, Some(ld_tok), tw, |v| (v[0] - v[1]) * (v[0] - v[1]));
    let l_p = wsum(g, enc.pitch, Some(p_tok), tw, |v| (v[0] - v[1]) * (v[0] - v[1]));
    let l_e = wsum(g, enc.energy, Some(e_tok), tw, |v| (v[0] - v[1]) * (v[0] - v[1]));
    // f0 channel 0 = normalized ln F0 (voiced frames), 1 = voicing logit
    let [bb, _, t] = ir::dims(g, f0_hat)[..] else { panic!() };
    let f0c = g.comp(&[bb, 1, t], &[], vec![arg(f0_hat, pick(3, &[Some(0), None, Some(2)]))], None, |v| vec![v[0]]).unwrap();
    let vz = g
        .comp(&[bb, 1, t], &[], vec![arg(f0_hat, view(vec![form(3, &[(0, 1)], 0), form(3, &[], 1), form(3, &[(2, 1)], 0)], false))], None, |v| vec![v[0]])
        .unwrap();
    let l_f0 = wsum(g, f0c, Some(lf0), vw, |v| ir::abs(v[0] - v[1]));
    // BCE with logits: softplus(z) − y·z, softplus stable both ways
    let l_vuv = wsum(g, vz, Some(vuv), fw, |v| {
        let z = v[0];
        let sp = ExprId::select(z, z + (lit(1.0) + (-z).exp()).ln(), (lit(1.0) + z.exp()).ln());
        sp - v[1] * z
    });
    let (wv, wf) = (tc.w_var, tc.w_f0);
    let total = g
        .map(&[l_mel, l_dur, l_p, l_e, l_f0, l_vuv], |v| v[0] + lit(wv) * (v[1] + v[2] + v[3]) + lit(wf) * (v[4] + v[5]))
        .unwrap();
    let losses = vec![total, l_mel, l_dur, l_p, l_e, l_f0, l_vuv];
    let params = p.ids.clone();
    TtsGraph { b, tc: tc.clone(), dims: d, params, inp, align, fmask, p_tok, e_tok, ld_tok, tw, mel, lf0, vuv, fw, vw, enc, mel_hat, f0_hat, losses }
}

pub struct TtsTrain {
    pub tg: TtsGraph,
    pub m: Vec<TensorId>,
    pub v: Vec<TensorId>,
    pub hyper: TensorId,
    pub grads: Vec<TensorId>,
    /// `[grad_norm, losses (7)…, p'…, m'…, v'…, partials…]`
    pub outputs: Vec<TensorId>,
}

pub const N_LOSS: usize = 7;

pub fn train_graph(tc: &TConfig, d: Dims) -> TtsTrain {
    let mut tg = tts_graph(tc, d);
    let specs = tc.params();
    let m: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| tg.b.input(&s.shape, Role::AdamM(i))).collect();
    let v: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| tg.b.input(&s.shape, Role::AdamV(i))).collect();
    let hyper = tg.b.input(&[6], Role::Hyper);
    let g = &mut tg.b.g;
    let grads = g.backward(tg.losses[0], &tg.params).unwrap();
    let ad = model::adamw_betas(g, &specs, &tg.params, &grads, &m, &v, hyper, 0.9, 0.98);
    let mut outputs = vec![ad.grad_norm];
    outputs.extend(&tg.losses);
    outputs.extend(&ad.new_p);
    outputs.extend(&ad.new_m);
    outputs.extend(&ad.new_v);
    outputs.extend(ad.partials);
    TtsTrain { tg, m, v, hyper, grads, outputs }
}

/// Inference, stage 1: tokens + reference → token states and predictions.
pub struct EncGraph {
    pub b: Built,
    pub params: Vec<TensorId>,
    pub inp: Inputs,
    pub enc: EncOut,
}

pub fn enc_graph(tc: &TConfig, d: Dims) -> EncGraph {
    let mut b = Built::new_pub(&Config::tiny());
    let p = Params::new(&mut b, &tc.params());
    let (inp, pe) = enc_inputs(&mut b, tc, d);
    let ei = EncIn { onehot: inp.onehot, tmask: inp.tmask, refmel: inp.refmel, rw: inp.rw, pe };
    let enc = encoder(&mut b.g, tc, &p, &ei, None);
    EncGraph { b, params: p.ids, inp, enc }
}

/// Inference, stage 2: token states (data) + pitch/energy + alignment → mel.
pub struct DecGraph {
    pub b: Built,
    pub params: Vec<TensorId>,
    pub h: TensorId,
    pub pitch: TensorId,
    pub energy: TensorId,
    pub tmask: TensorId,
    pub align: TensorId,
    pub fmask: TensorId,
    pub spk_dec: TensorId,
    pub mel: TensorId,
    pub f0: TensorId,
    /// last hidden state after the final LN `[B, dd, T]` (flow conditioning)
    pub hid: TensorId,
}

pub fn dec_graph(tc: &TConfig, d: Dims) -> DecGraph {
    let mut b = Built::new_pub(&Config::tiny());
    let p = Params::new(&mut b, &tc.params());
    let h = b.input(&[d.batch, tc.d, d.n], Role::Data("h"));
    let pitch = b.input(&[d.batch, 1, d.n], Role::Data("pitch"));
    let energy = b.input(&[d.batch, 1, d.n], Role::Data("energy"));
    let tmask = b.input(&[d.batch, 1, d.n], Role::Data("tmask"));
    let align = b.input(&[d.batch, d.n, d.t], Role::Data("align"));
    let fmask = b.input(&[d.batch, 1, d.t], Role::Data("fmask"));
    let spk_dec = b.input(&[d.batch, tc.dd, 1], Role::Data("spk_dec"));
    let (mel, f0, hid) = decoder(&mut b.g, tc, &p, h, pitch, energy, tmask, align, fmask, spk_dec);
    DecGraph { b, params: p.ids, h, pitch, energy, tmask, align, fmask, spk_dec, mel, f0, hid }
}

/// One vector per graph input of `b` (TTS params/state + named data).
pub fn assemble(b: &Built, specs: &[ParamSpec], params: &[Vec<f32>], adam: Option<(&[Vec<f32>], &[Vec<f32>])>, data: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
    crate::align::assemble(b, specs, params, adam, data)
}

/// Hard alignment `[B, N, T]` from per-token durations (frames).
pub fn align_matrix(durs: &[&[u32]], n: usize, t: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; durs.len() * n * t];
    for (b, d) in durs.iter().enumerate() {
        let mut pos = 0usize;
        for (i, &k) in d.iter().enumerate().take(n) {
            for f in pos..(pos + k as usize).min(t) {
                a[(b * n + i) * t + f] = 1.0;
            }
            pos += k as usize;
        }
    }
    a
}

/// Hard alignment `[B, N, T]` of a frame window: utterance frame
/// `starts[b] + j` → column `j` (training crops).
pub fn align_matrix_from(durs: &[&[u32]], starts: &[usize], n: usize, t: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; durs.len() * n * t];
    for (b, d) in durs.iter().enumerate() {
        let s0 = starts[b];
        let mut pos = 0usize;
        for (i, &k) in d.iter().enumerate().take(n) {
            for f in pos.max(s0)..(pos + k as usize).min(s0 + t) {
                a[(b * n + i) * t + f - s0] = 1.0;
            }
            pos += k as usize;
        }
    }
    a
}

// ---------------------------------------------------------------------------
// context-aware prosody (the ribt paper)
//
// A small context path conditions the three variance predictors on the
// dialogue: the previous turn's tokens (cross-attention from the current
// tokens: a word that was just said can be found and deaccented), the current
// sentence's role (none / answer / question / ack / correction) and per-token
// emphasis hints. It enters each predictor LN through zero-initialised AdaLN
// and is multiplied by a per-utterance gate (0 = reading, no context), so an
// untrained path — and any gate-0 row — is exactly the base model. Everything
// it reads is known before the turn starts (streaming-compatible).

/// Roles of the current sentence.
pub const CTX_ROLES: [&str; 5] = ["none", "answer", "question", "ack", "correction"];

#[derive(Clone, Debug)]
pub struct CtxConfig {
    pub dc: usize,
}

impl CtxConfig {
    pub fn base() -> Self {
        Self { dc: 64 }
    }
    pub fn tiny() -> Self {
        Self { dc: 6 }
    }
    /// Context parameters (appended after `TConfig::params` in ctx graphs).
    pub fn params(&self, tc: &TConfig) -> Vec<ParamSpec> {
        let lin = |fan: usize| (1.0 / fan as f64).sqrt();
        let he = |fan: usize| (2.0 / fan as f64).sqrt();
        let dc = self.dc;
        let mut v = vec![
            ParamSpec::normal("ctx.emb", &[dc, tc.vocab], 0.5),
            ParamSpec::normal("ctx.conv.w", &[dc, dc, 3], he(3 * dc)),
            ParamSpec::zeros("ctx.conv.b", &[dc]),
            ParamSpec::normal("ctx.q", &[dc, dc], lin(dc)),
            ParamSpec::normal("ctx.k", &[dc, dc], lin(dc)),
            ParamSpec::normal("ctx.v", &[dc, dc], lin(dc)),
            ParamSpec::normal("ctx.hw", &[dc, tc.d], lin(tc.d)),
            ParamSpec::normal("ctx.role", &[dc, CTX_ROLES.len()], 0.5),
            ParamSpec::normal("ctx.emph", &[dc, 1], 0.5),
            ParamSpec::normal("ctx.m1.w", &[dc, dc], lin(dc)),
            ParamSpec::zeros("ctx.m1.b", &[dc]),
        ];
        for pr in ["dur", "pitch", "energy"] {
            for j in 1..=2 {
                v.push(ParamSpec::zeros(&format!("ctx.{pr}.g{j}"), &[tc.var_h, dc]));
                v.push(ParamSpec::zeros(&format!("ctx.{pr}.b{j}"), &[tc.var_h, dc]));
            }
        }
        v
    }
}

/// Base + context parameter specs (the order of every ctx graph's params).
pub fn ctx_specs(tc: &TConfig, cc: &CtxConfig) -> Vec<ParamSpec> {
    let mut v = tc.params();
    v.extend(cc.params(tc));
    v
}

/// Context inputs: previous turn one-hot `[B, V, M]` + mask `[B, 1, M]`
/// (the turn's two pad tokens are always valid: an empty turn is just the
/// pads, the attention's sink), role one-hot `[B, R, 1]`, emphasis
/// `[B, 1, N]`, gate `[B, 1, 1]`.
pub struct CtxIn {
    pub prev: TensorId,
    pub pmask: TensorId,
    pub role: TensorId,
    pub emph: TensorId,
    pub gate: TensorId,
}

/// Shared token featurizer of both turns: embedding → conv k3 → ReLU, masked.
fn ctx_feats(g: &mut TensorGraph, p: &Params, onehot: TensorId, mask: TensorId) -> TensorId {
    let x = ir::linear(g, onehot, p.p("ctx.emb"));
    let x = ir::conv1d(g, x, p.p("ctx.conv.w"), p.p("ctx.conv.b"));
    let x = g.map(&[x], |v| relu(v[0])).unwrap();
    mask_mul(g, x, mask)
}

/// Context state `c [B, dc, N]` (gated, masked).
fn ctx_state(g: &mut TensorGraph, p: &Params, h: TensorId, onehot: TensorId, tmask: TensorId, cx: &CtxIn) -> TensorId {
    let fq = ctx_feats(g, p, onehot, tmask);
    let fk = ctx_feats(g, p, cx.prev, cx.pmask);
    let [b, dc, n] = ir::dims(g, fq)[..] else { panic!() };
    let m = ir::dims(g, fk)[2];
    let q = ir::linear(g, fq, p.p("ctx.q"));
    let k = ir::linear(g, fk, p.p("ctx.k"));
    let v = ir::linear(g, fk, p.p("ctx.v"));
    let scale = 1.0 / (dc as f64).sqrt();
    // s[b,i,j] = scale·Σ_c q[b,c,i]·k[b,c,j] + bias[b,j]; space (b, i, j, c)
    let s = g
        .comp(&[b, n, m], &[dc], vec![arg(q, pick(4, &[Some(0), Some(3), Some(1)])), arg(k, pick(4, &[Some(0), Some(3), Some(2)]))], Some(Monoid::sum()), |v| vec![v[0] * v[1]])
        .unwrap();
    let s = g
        .comp(&[b, n, m], &[], vec![arg(s, pick(3, &[Some(0), Some(1), Some(2)])), arg(cx.pmask, pick(3, &[Some(0), None, Some(2)]))], None, move |v| {
            vec![v[0] * lit(scale) + (v[1] - lit(1.0)) * lit(1e4)]
        })
        .unwrap();
    let mx = g.comp(&[b, n], &[m], vec![arg(s, pick(3, &[Some(0), Some(1), Some(2)]))], Some(Monoid::max()), |v| vec![v[0]]).unwrap();
    let e = g.comp(&[b, n, m], &[], vec![arg(s, pick(3, &[Some(0), Some(1), Some(2)])), arg(mx, pick(3, &[Some(0), Some(1)]))], None, |v| vec![(v[0] - v[1]).exp()]).unwrap();
    let l = g.comp(&[b, n], &[m], vec![arg(e, pick(3, &[Some(0), Some(1), Some(2)]))], Some(Monoid::sum()), |v| vec![v[0]]).unwrap();
    // o[b,c,i] = Σ_j e[b,i,j]·v[b,c,j] / l[b,i]; space (b, c, i, j)
    let o = g
        .comp(&[b, dc, n], &[m], vec![arg(e, pick(4, &[Some(0), Some(2), Some(3)])), arg(v, pick(4, &[Some(0), Some(1), Some(3)]))], Some(Monoid::sum()), |v| vec![v[0] * v[1]])
        .unwrap();
    let o = g.comp(&[b, dc, n], &[], vec![arg(o, pick(3, &[Some(0), Some(1), Some(2)])), arg(l, pick(3, &[Some(0), Some(2)]))], None, |v| vec![v[0] / v[1]]).unwrap();
    let hp = ir::linear(g, h, p.p("ctx.hw"));
    let rv = ir::linear(g, cx.role, p.p("ctx.role"));
    let ev = ir::linear(g, cx.emph, p.p("ctx.emph"));
    let x = add(g, o, hp);
    let x = add(g, x, fq);
    let x = add(g, x, ev);
    let x = add_bc(g, x, rv);
    let x = ir::materialize(g, x);
    let x = ir::linear(g, x, p.p("ctx.m1.w"));
    let x = ir::channel_map(g, x, &[p.p("ctx.m1.b")], |x, b| relu(x + b[0]));
    // gate (per utterance) and token mask
    let x = g
        .comp(&[b, dc, n], &[], vec![arg(x, pick(3, &[Some(0), Some(1), Some(2)])), arg(cx.gate, pick(3, &[Some(0), None, None]))], None, |v| vec![v[0] * v[1]])
        .unwrap();
    let x = mask_mul(g, x, tmask);
    ir::materialize(g, x)
}

fn ctx_inputs(b: &mut Built, tc: &TConfig, d: Dims, m: usize) -> CtxIn {
    CtxIn {
        prev: b.input(&[d.batch, tc.vocab, m], Role::Data("prev")),
        pmask: b.input(&[d.batch, 1, m], Role::Data("pmask")),
        role: b.input(&[d.batch, CTX_ROLES.len(), 1], Role::Data("role")),
        emph: b.input(&[d.batch, 1, d.n], Role::Data("emph")),
        gate: b.input(&[d.batch, 1, 1], Role::Data("gate")),
    }
}

/// Inference stage 1 with the context path (`m` = previous-turn tokens).
pub fn ctx_enc_graph(tc: &TConfig, cc: &CtxConfig, d: Dims, m: usize) -> EncGraph {
    let mut b = Built::new_pub(&Config::tiny());
    let p = Params::new(&mut b, &ctx_specs(tc, cc));
    let (inp, pe) = enc_inputs(&mut b, tc, d);
    let cx = ctx_inputs(&mut b, tc, d, m);
    let ei = EncIn { onehot: inp.onehot, tmask: inp.tmask, refmel: inp.refmel, rw: inp.rw, pe };
    let enc = encoder(&mut b.g, tc, &p, &ei, Some(&cx));
    EncGraph { b, params: p.ids, inp, enc }
}

pub const N_CTX_LOSS: usize = 4;

/// Context-path training: base params frozen (inputs, never updated), the
/// variance losses only, AdamW on the context params.
pub struct CtxTrain {
    pub b: Built,
    /// all params (base then ctx), and the trained subset's index offset
    pub params: Vec<TensorId>,
    pub n_base: usize,
    pub m: Vec<TensorId>,
    pub v: Vec<TensorId>,
    pub hyper: TensorId,
    /// `[grad_norm, total, dur, pitch, energy, ctx p'…, m'…, v'…, partials…]`
    pub outputs: Vec<TensorId>,
    /// forward-only losses `[total, dur, pitch, energy]` (held-out eval)
    pub losses: Vec<TensorId>,
}

pub fn ctx_train_graph(tc: &TConfig, cc: &CtxConfig, d: Dims, m: usize) -> CtxTrain {
    let mut b = Built::new_pub(&Config::tiny());
    let specs = ctx_specs(tc, cc);
    let n_base = tc.params().len();
    let p = Params::new(&mut b, &specs);
    let (inp, pe) = enc_inputs(&mut b, tc, d);
    let cx = ctx_inputs(&mut b, tc, d, m);
    let p_tok = b.input(&[d.batch, 1, d.n], Role::Data("p_tok"));
    let e_tok = b.input(&[d.batch, 1, d.n], Role::Data("e_tok"));
    let ld_tok = b.input(&[d.batch, 1, d.n], Role::Data("ld_tok"));
    let tw = b.input(&[d.batch, 1, d.n], Role::Data("tw"));
    let ctx_ids: Vec<TensorId> = p.ids[n_base..].to_vec();
    let mv: Vec<TensorId> = (n_base..specs.len()).map(|i| b.input(&specs[i].shape, Role::AdamM(i))).collect();
    let vv: Vec<TensorId> = (n_base..specs.len()).map(|i| b.input(&specs[i].shape, Role::AdamV(i))).collect();
    let hyper = b.input(&[6], Role::Hyper);
    let g = &mut b.g;
    let ei = EncIn { onehot: inp.onehot, tmask: inp.tmask, refmel: inp.refmel, rw: inp.rw, pe };
    let enc = encoder(g, tc, &p, &ei, Some(&cx));
    let l_dur = wsum(g, enc.ld, Some(ld_tok), tw, |v| (v[0] - v[1]) * (v[0] - v[1]));
    let l_p = wsum(g, enc.pitch, Some(p_tok), tw, |v| (v[0] - v[1]) * (v[0] - v[1]));
    let l_e = wsum(g, enc.energy, Some(e_tok), tw, |v| (v[0] - v[1]) * (v[0] - v[1]));
    let total = g.map(&[l_dur, l_p, l_e], |v| v[0] + v[1] + v[2]).unwrap();
    let grads = g.backward(total, &ctx_ids).unwrap();
    let ad = model::adamw_betas(g, &specs[n_base..], &ctx_ids, &grads, &mv, &vv, hyper, 0.9, 0.98);
    let losses = vec![total, l_dur, l_p, l_e];
    let mut outputs = vec![ad.grad_norm];
    outputs.extend(&losses);
    outputs.extend(&ad.new_p);
    outputs.extend(&ad.new_m);
    outputs.extend(&ad.new_v);
    outputs.extend(ad.partials);
    // last: predicted token pitch (diagnostics)
    outputs.push(enc.pitch);
    CtxTrain { b, params: p.ids, n_base, m: mv, v: vv, hyper, outputs, losses }
}
