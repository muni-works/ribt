//! §29.4 differentiable reward optimization (DiffRO, end to end in tang).
//!
//! - **Critic** (the differentiable reward): a small ConvNeXt over the
//!   flow-normalized log-mel `z [B, C, T]` → per-frame logits `[B, 1, T]`.
//!   It is trained (logistic loss, per frame, PatchGAN-style) to tell the
//!   teacher corpus' real mels (Kokoro af_heart audio, label 1) from the
//!   1-step student's mels under inference conditioning (label 0). No MOS
//!   judge is in the reward, so UTMOS, DNSMOS and NISQA all stay held out.
//! - **DiffRO step**: one graph = student velocity net (`flow::flow_graph`),
//!   `x₁ = x₀ + v(x₀, t 0, d 1)`, the frozen critic on `x₁`, and the loss
//!   `−mean critic logit + λ·‖x₁ − x₁ʳᵉᶠ‖²` (a trust region to the frozen
//!   student's own output for the same noise). Backward and AdamW on the
//!   student only; the critic's weights are `DParam` inputs (no update).

use std::collections::HashMap;

use tang::Scalar;
use tang_compile::tensor::{Monoid, TensorGraph, TensorId};
use tang_expr::ExprId;

use crate::flow::{self, FConfig, FDims};
use crate::ir::{self, arg, lit, pick};
use crate::model::{self, Built, Config, ParamSpec, Role};

#[derive(Clone, Debug)]
pub struct RConfig {
    pub n_mels: usize,
    pub d: usize,
    pub ff: usize,
    pub k_in: usize,
    pub k: usize,
    pub layers: usize,
}

impl RConfig {
    /// ~0.5M: conv k5 100→192, 4 ConvNeXt blocks (dw k7, ff 384). Receptive
    /// field 5 + 4·6 = 29 frames (0.31 s): texture, not prosody.
    pub fn base() -> Self {
        Self { n_mels: 100, d: 192, ff: 384, k_in: 5, k: 7, layers: 4 }
    }
    pub fn check() -> Self {
        Self { n_mels: 20, d: 16, ff: 32, k_in: 5, k: 7, layers: 2 }
    }
    pub fn by_name(s: &str) -> Option<Self> {
        match s {
            "base" => Some(Self::base()),
            "check" => Some(Self::check()),
            _ => None,
        }
    }
    pub fn params(&self) -> Vec<ParamSpec> {
        let lin = |fan: usize| (1.0 / fan as f64).sqrt();
        let d = self.d;
        let mut v = vec![ParamSpec::normal("in.w", &[d, self.n_mels, self.k_in], lin(self.n_mels * self.k_in)), ParamSpec::zeros("in.b", &[d])];
        for l in 0..self.layers {
            let p = |s: &str| format!("b.{l}.{s}");
            v.push(ParamSpec::normal(&p("dw.w"), &[d, self.k], lin(self.k)));
            v.push(ParamSpec::zeros(&p("dw.b"), &[d]));
            v.push(ParamSpec::ones(&p("ln.g"), &[d]));
            v.push(ParamSpec::zeros(&p("ln.b"), &[d]));
            v.push(ParamSpec::normal(&p("pw1.w"), &[self.ff, d], lin(d)));
            v.push(ParamSpec::zeros(&p("pw1.b"), &[self.ff]));
            v.push(ParamSpec::normal(&p("pw2.w"), &[d, self.ff], 0.1 * lin(self.ff)));
            v.push(ParamSpec::zeros(&p("pw2.b"), &[d]));
        }
        v.push(ParamSpec::ones("out.ln.g", &[d]));
        v.push(ParamSpec::zeros("out.ln.b", &[d]));
        v.push(ParamSpec::normal("out.w", &[1, d], lin(d)));
        v.push(ParamSpec::zeros("out.b", &[1]));
        v
    }
    pub fn n_params(&self) -> usize {
        self.params().iter().map(|p| p.numel()).sum()
    }
}

pub fn init_params(rc: &RConfig, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = model::Rng::new(seed);
    rc.params().iter().map(|s| s.init_data(&mut rng)).collect()
}

fn mask_mul(g: &mut TensorGraph, x: TensorId, m: TensorId) -> TensorId {
    let sh = ir::dims(g, x);
    g.comp(&sh, &[], vec![arg(x, pick(3, &[Some(0), Some(1), Some(2)])), arg(m, pick(3, &[Some(0), None, Some(2)]))], None, |v| vec![v[0] * v[1]])
        .unwrap()
}

/// Critic forward: per-frame logits `[B, 1, T]` (masked); `p(name)` → param id.
pub fn critic(g: &mut TensorGraph, rc: &RConfig, p: &dyn Fn(&str) -> TensorId, z: TensorId, fmask: TensorId) -> TensorId {
    let x = ir::conv1d(g, z, p("in.w"), p("in.b"));
    let x = mask_mul(g, x, fmask);
    let mut x = ir::materialize(g, x);
    for l in 0..rc.layers {
        let q = |s: &str| p(&format!("b.{l}.{s}"));
        let y = ir::dwconv(g, x, q("dw.w"), q("dw.b"));
        let y = ir::layer_norm(g, y, q("ln.g"), q("ln.b"), 1e-6);
        let y = ir::linear(g, y, q("pw1.w"));
        let y = ir::channel_map(g, y, &[q("pw1.b")], |x, b| ir::gelu(x + b[0]));
        let y = ir::linear(g, y, q("pw2.w"));
        let y = ir::add_channel(g, y, q("pw2.b"));
        let x2 = g.map(&[x, y], |v| v[0] + v[1]).unwrap();
        let x2 = mask_mul(g, x2, fmask);
        x = ir::materialize(g, x2);
    }
    let y = ir::layer_norm(g, x, p("out.ln.g"), p("out.ln.b"), 1e-6);
    let y = ir::linear(g, y, p("out.w"));
    let y = ir::add_channel(g, y, p("out.b"));
    mask_mul(g, y, fmask)
}

/// `Σ_{b,t} f(l[b,0,t], w[b,0,t], y[b,0,0]?)` → `[1]`. Space `(one, b, t)`.
fn wsum(g: &mut TensorGraph, l: TensorId, w: TensorId, y: Option<TensorId>, f: impl FnOnce(&[ExprId]) -> ExprId) -> TensorId {
    let [b, _, t] = ir::dims(g, l)[..] else { panic!() };
    let mut args = vec![arg(l, pick(3, &[Some(1), None, Some(2)])), arg(w, pick(3, &[Some(1), None, Some(2)]))];
    if let Some(y) = y {
        args.push(arg(y, pick(3, &[Some(1), None, None])));
    }
    g.comp(&[1], &[b, t], args, Some(Monoid::sum()), |v| vec![f(v)]).unwrap()
}

/// Stable softplus.
fn softplus(z: ExprId) -> ExprId {
    ExprId::select(z, z + (lit(1.0) + (-z).exp()).ln(), (lit(1.0) + z.exp()).ln())
}

pub struct CriticGraph {
    pub b: Built,
    pub params: Vec<TensorId>,
    pub logits: TensorId,
    /// `Σ fw·BCE(logit, y)` → `[1]` (fw = per-frame weights summing to 1)
    pub loss: TensorId,
}

/// Critic and its per-frame logistic loss; data `z`, `fmask`/`fw [B,1,T]`, `y [B,1,1]`.
pub fn critic_graph(rc: &RConfig, batch: usize, t: usize) -> CriticGraph {
    let mut b = Built::new_pub(&Config::tiny());
    let specs = rc.params();
    let ids: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| b.input(&s.shape, Role::Param(i))).collect();
    let z = b.input(&[batch, rc.n_mels, t], Role::Data("z"));
    let fmask = b.input(&[batch, 1, t], Role::Data("fmask"));
    let fw = b.input(&[batch, 1, t], Role::Data("fw"));
    let y = b.input(&[batch, 1, 1], Role::Data("y"));
    let by: HashMap<String, TensorId> = specs.iter().zip(&ids).map(|(s, &i)| (s.name.clone(), i)).collect();
    let g = &mut b.g;
    let logits = critic(g, rc, &|n| by[n], z, fmask);
    let logits = ir::materialize(g, logits);
    // BCE with logits: softplus(l) − y·l
    let loss = wsum(g, logits, fw, Some(y), |v| (softplus(v[0]) - v[2] * v[0]) * v[1]);
    CriticGraph { b, params: ids, logits, loss }
}

pub struct CriticTrain {
    pub cg: CriticGraph,
    pub m: Vec<TensorId>,
    pub v: Vec<TensorId>,
    pub hyper: TensorId,
    /// `[grad_norm, loss, logits, p'…, m'…, v'…, partials…]`
    pub outputs: Vec<TensorId>,
}

pub fn critic_train_graph(rc: &RConfig, batch: usize, t: usize) -> CriticTrain {
    let mut cg = critic_graph(rc, batch, t);
    let specs = rc.params();
    let m: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| cg.b.input(&s.shape, Role::AdamM(i))).collect();
    let v: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| cg.b.input(&s.shape, Role::AdamV(i))).collect();
    let hyper = cg.b.input(&[6], Role::Hyper);
    let g = &mut cg.b.g;
    let grads = g.backward(cg.loss, &cg.params).unwrap();
    let ad = model::adamw_betas(g, &specs, &cg.params, &grads, &m, &v, hyper, 0.5, 0.9);
    let mut outputs = vec![ad.grad_norm, cg.loss, cg.logits];
    outputs.extend(&ad.new_p);
    outputs.extend(&ad.new_m);
    outputs.extend(&ad.new_v);
    outputs.extend(ad.partials);
    CriticTrain { cg, m, v, hyper, outputs }
}

pub struct DiffroTrain {
    pub fg: flow::FlowGraph,
    /// critic weights (`Role::DParam`, frozen)
    pub cparams: Vec<TensorId>,
    pub m: Vec<TensorId>,
    pub v: Vec<TensorId>,
    pub hyper: TensorId,
    /// `[grad_norm, loss, reward, trust, p'…, m'…, v'…, partials…]`
    pub outputs: Vec<TensorId>,
}

/// One DiffRO step for the 1-step shortcut student (`xt` = `x₀`, `temb` =
/// time features of t 0, d 1). Data `x1ref [B,C,T]` (the frozen student's
/// x₁ for the same x₀) and `fw [B,1,T]` (per-frame weights summing to 1).
/// `lam` weights the trust region.
pub fn diffro_graph(fc: &FConfig, rc: &RConfig, d: FDims, lam: f64) -> DiffroTrain {
    let mut fg = flow::flow_graph(fc, d, false);
    let fspecs = fc.params();
    let rspecs = rc.params();
    fg.b.dspecs = rspecs.clone();
    let cparams: Vec<TensorId> = rspecs.iter().enumerate().map(|(i, s)| fg.b.input(&s.shape, Role::DParam(i))).collect();
    let x1ref = fg.b.input(&[d.batch, fc.n_mels, d.t], Role::Data("x1ref"));
    let fw = fg.b.input(&[d.batch, 1, d.t], Role::Data("fw"));
    let m: Vec<TensorId> = fspecs.iter().enumerate().map(|(i, s)| fg.b.input(&s.shape, Role::AdamM(i))).collect();
    let v: Vec<TensorId> = fspecs.iter().enumerate().map(|(i, s)| fg.b.input(&s.shape, Role::AdamV(i))).collect();
    let hyper = fg.b.input(&[6], Role::Hyper);
    let by: HashMap<String, TensorId> = rspecs.iter().zip(&cparams).map(|(s, &i)| (s.name.clone(), i)).collect();
    let (xt, vel, fmask) = (fg.xt, fg.v, fg.fmask);
    let g = &mut fg.b.g;
    let x1 = g.map(&[xt, vel], |a| a[0] + a[1]).unwrap();
    let x1 = ir::materialize(g, x1);
    let logits = critic(g, rc, &|n| by[n], x1, fmask);
    let reward = wsum(g, logits, fw, None, |v| v[0] * v[1]);
    let c = fc.n_mels;
    let inv = 1.0 / c as f64;
    let (bb, t) = (d.batch, d.t);
    let tr = g
        .comp(&[bb, 1, t], &[c], vec![arg(x1, pick(4, &[Some(0), Some(3), Some(2)])), arg(x1ref, pick(4, &[Some(0), Some(3), Some(2)])), arg(fw, pick(4, &[Some(0), None, Some(2)]))], Some(Monoid::sum()), |a| {
            let dd = a[0] - a[1];
            vec![dd * dd * a[2] * lit(inv)]
        })
        .unwrap();
    let trust = ir::sum_all2(g, tr, None, |v| v[0]);
    let loss = g.map(&[reward, trust], |v| v[1] * lit(lam) - v[0]).unwrap();
    let grads = g.backward(loss, &fg.params).unwrap();
    let ad = model::adamw_betas(g, &fspecs, &fg.params, &grads, &m, &v, hyper, 0.9, 0.99);
    let mut outputs = vec![ad.grad_norm, loss, reward, trust];
    outputs.extend(&ad.new_p);
    outputs.extend(&ad.new_m);
    outputs.extend(&ad.new_v);
    outputs.extend(ad.partials);
    DiffroTrain { fg, cparams, m, v, hyper, outputs }
}
