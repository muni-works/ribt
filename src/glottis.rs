//! §11.5: physics as the range operator.
//!
//! - [`phys_mag`]: an LF-style glottal source (F0 comb at frame rate,
//!   glottal formant from the open quotient, spectral-tilt lowpass,
//!   aspiration noise) times a per-frame mel-cepstral envelope, synthesized
//!   directly as an STFT magnitude `[B, bins, T]`.
//! - [`project`]: the exact mel-consistent projection. Given any positive
//!   candidate magnitude `x` and a target log-mel, returns
//!   `s = x ⊙ exp(F λ)` with `Fᵀ s = exp(mel)`: the KL (I-)projection of `x`
//!   onto the mel-consistent set. `s > 0` by construction (no `.abs()`
//!   leak); λ comes from Newton on the convex dual, whose Hessian
//!   `Fᵀ diag(s) F` is tridiagonal for HTK triangles and is solved by
//!   parallel cyclic reduction (log₂ M guarded-shift comps, no scan).
//! - [`pinv_range`]: RNDVoC's range `F⁺ e` (for the baseline), as `|F⁺e|`.
//! - [`fit_graph`]: analysis-by-synthesis, Adam on the per-frame params.

use tang::Scalar;
use tang_compile::tensor::{Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

use crate::dsp;
use crate::ir::{self, arg, form, lit, pick, view};
use crate::model::{Built, Config, Role};

/// Per-frame physical params: `[oq_logit, ln Fa, ln asp, c_0 … c_{Q−1}]`.
pub const P_OQ: usize = 0;
pub const P_TILT: usize = 1;
pub const P_ASP: usize = 2;
pub const P_CEP: usize = 3;
pub const NEWTON: usize = 12;
/// Comb peak width (Gaussian σ in bins ≈ Hann main lobe at −6 dB).
const COMB_SIGMA: f64 = 0.85;
pub const PRIOR_TILT: f64 = 8.0; // ln 3000 Hz
pub const PRIOR_ASP: f64 = -2.3; // ln 0.1

pub fn n_params(cfg: &Config) -> usize {
    P_CEP + cfg.n_cep
}

/// open quotient from its logit, in [0.3, 0.9]
pub fn oq_of(l: ExprId) -> ExprId {
    lit(0.3) + lit(0.6) / (lit(1.0) + (-l).exp())
}

/// Constant tables for the physics model and the projection.
pub struct PConsts {
    pub fb: TensorId,
    pub fb2: TensorId,
    pub fbn: TensorId,
    pub valid: TensorId,
    pub cep: TensorId,
    pub freqs: TensorId,
}

pub fn pconsts(b: &mut Built) -> PConsts {
    let cfg = b.cfg.clone();
    let (k, m) = (cfg.bins(), cfg.n_mels);
    let fb = dsp::mel_fbank(cfg.sr, cfg.n_fft, m);
    let fb2: Vec<f64> = fb.iter().map(|v| v * v).collect();
    let mut fbn = vec![0.0; k * m];
    for i in 0..k {
        for j in 0..m - 1 {
            fbn[i * m + j] = fb[i * m + j] * fb[i * m + j + 1];
        }
    }
    let mmax = 2595.0 * (1.0 + (cfg.sr as f64 / 2.0) / 700.0).log10();
    let q = cfg.n_cep.max(1);
    let mut cep = vec![0.0; q * k];
    for qq in 0..q {
        for i in 0..k {
            let f = i as f64 * cfg.sr as f64 / cfg.n_fft as f64;
            let mu = 2595.0 * (1.0 + f / 700.0).log10() / mmax;
            cep[qq * k + i] = (std::f64::consts::PI * qq as f64 * mu).cos();
        }
    }
    let df = cfg.sr as f64 / cfg.n_fft as f64;
    PConsts {
        fb: b.constant(&[k, m], fb),
        fb2: b.constant(&[k, m], fb2),
        fbn: b.constant(&[k, m], fbn),
        valid: b.constant(&[m], vec![1.0; m]),
        cep: b.constant(&[q, k], cep),
        freqs: b.constant(&[k], (0..k).map(|i| i as f64 * df).collect()),
    }
}

/// `y[b,m,t] = Σ_k W[k,m] · x[b,k,t]`.
fn to_bands(g: &mut TensorGraph, w: TensorId, x: TensorId) -> TensorId {
    let [b, k, t] = ir::dims(g, x)[..] else { panic!() };
    let m = ir::dims(g, w)[1];
    // space (b, m, t, k)
    g.comp(&[b, m, t], &[k], vec![arg(x, pick(4, &[Some(0), Some(3), Some(2)])), arg(w, pick(4, &[Some(3), Some(1)]))], Some(Monoid::sum()), |v| vec![v[0] * v[1]])
        .unwrap()
}

/// `y[b,k,t] = Σ_m W[k,m] · z[b,m,t]`.
fn to_bins(g: &mut TensorGraph, w: TensorId, z: TensorId) -> TensorId {
    let [b, m, t] = ir::dims(g, z)[..] else { panic!() };
    let k = ir::dims(g, w)[0];
    // space (b, k, t, m)
    g.comp(&[b, k, t], &[m], vec![arg(z, pick(4, &[Some(0), Some(3), Some(2)])), arg(w, pick(4, &[Some(1), Some(3)]))], Some(Monoid::sum()), |v| vec![v[0] * v[1]])
        .unwrap()
}

/// Tridiagonal solve per (b, t) over the band axis by parallel cyclic
/// reduction: `lo[m]·y[m−1] + d[m]·y[m] + up[m]·y[m+1] = r[m]`, where
/// `up[m]` couples m and m+1 (`up[M−1] = 0`) and `lo[m] = up[m−1]`.
pub fn pcr(g: &mut TensorGraph, valid: TensorId, up: TensorId, d: TensorId, r: TensorId, keep: &mut Vec<TensorId>) -> TensorId {
    let [_, m, _] = ir::dims(g, d)[..] else { panic!() };
    let sh = ir::dims(g, d);
    // lo[m] = up[m−1] (guarded: 0 at m = 0)
    let shv = |s: i64| view(vec![form(3, &[(0, 1)], 0), form(3, &[(1, 1)], s), form(3, &[(2, 1)], 0)], true);
    let mut lo = g.comp(&sh, &[], vec![arg(up, shv(-1))], None, |v| vec![v[0]]).unwrap();
    let (mut d, mut up, mut r) = (d, up, r);
    let mut s = 1i64;
    while (s as usize) < m {
        let vm = |o: i64| view(vec![form(3, &[(1, 1)], o)], true);
        let args = |x: [TensorId; 4]| {
            let mut a = vec![arg(lo, View::identity(3, 3)), arg(d, View::identity(3, 3)), arg(up, View::identity(3, 3)), arg(r, View::identity(3, 3))];
            for o in [-s, s] {
                for t in x {
                    a.push(arg(t, shv(o)));
                }
                a.push(arg(valid, vm(o)));
            }
            a
        };
        // v: 0 lo,1 d,2 up,3 r | 4 lo⁻,5 d⁻,6 up⁻,7 r⁻,8 valid⁻ | 9 lo⁺,10 d⁺,11 up⁺,12 r⁺,13 valid⁺
        let k = |v: &[ExprId]| (v[0] / (v[5] + lit(1.0) - v[8]), v[2] / (v[10] + lit(1.0) - v[13]));
        let q = [lo, d, up, r];
        let nlo = g.comp(&sh, &[], args(q), None, |v| { let (k1, _) = k(v); vec![-(k1 * v[4])] }).unwrap();
        let nd = g.comp(&sh, &[], args(q), None, |v| { let (k1, k2) = k(v); vec![v[1] - k1 * v[6] - k2 * v[9]] }).unwrap();
        let nup = g.comp(&sh, &[], args(q), None, |v| { let (_, k2) = k(v); vec![-(k2 * v[11])] }).unwrap();
        let nr = g.comp(&sh, &[], args(q), None, |v| { let (k1, k2) = k(v); vec![v[3] - k1 * v[7] - k2 * v[12]] }).unwrap();
        keep.extend([nlo, nd, nup, nr]);
        lo = nlo;
        d = nd;
        up = nup;
        r = nr;
        s *= 2;
    }
    g.map(&[r, d], |v| v[0] / v[1]).unwrap()
}

/// Exact mel-consistent projection of a positive candidate `x` `[B,bins,T]`
/// onto `{s : Fᵀ s = exp(mel)}` (`mel` `[B,M,T]`, already floor-clipped).
pub fn project(g: &mut TensorGraph, pc: &PConsts, x: TensorId, mel: TensorId) -> TensorId {
    project_keep(g, pc, x, mel, &mut Vec::new())
}

/// [`project`], also listing its intermediates in `keep`: a train graph
/// takes gradients w.r.t. them so they are materialized (otherwise the
/// backward fuses the PCR chain into kernels past Metal's 31 buffers).
pub fn project_keep(g: &mut TensorGraph, pc: &PConsts, x: TensorId, mel: TensorId, keep: &mut Vec<TensorId>) -> TensorId {
    let (e, mut lam) = dual_init(g, pc, x, mel);
    for _ in 0..NEWTON {
        lam = newton_step(g, pc, x, e, lam, false, keep);
    }
    s_of(g, pc, x, lam)
}

/// [`project`] with the implicit-function-theorem gradient: the `NEWTON`
/// steps run as in [`project_keep`], the converged λ is detached, and one
/// more Newton step is taken from it with a detached Hessian. At the
/// solution that step's derivative is `−H⁻¹ ∂r`, the derivative of the
/// solve, so the backward is one step's (one PCR chain) whatever `NEWTON`
/// is. Only that step's intermediates go in `keep`. The value is
/// [`project`]'s after `NEWTON + 1` steps; the gradient is exact once the
/// iteration has converged (last step inside the ±4 clamp, exp below its
/// cap).
pub fn project_implicit(g: &mut TensorGraph, pc: &PConsts, x: TensorId, mel: TensorId, keep: &mut Vec<TensorId>) -> TensorId {
    let (e, mut lam) = dual_init(g, pc, x, mel);
    for _ in 0..NEWTON {
        lam = newton_step(g, pc, x, e, lam, false, &mut Vec::new());
    }
    let fixed = g.detach(lam).unwrap();
    let lam = newton_step(g, pc, x, e, fixed, true, keep);
    s_of(g, pc, x, lam)
}

/// `(exp(mel), λ₀)`: the target and the dual start `mel − ln Fᵀx`.
fn dual_init(g: &mut TensorGraph, pc: &PConsts, x: TensorId, mel: TensorId) -> (TensorId, TensorId) {
    let e = g.map(&[mel], |v| v[0].exp()).unwrap();
    let fx = to_bands(g, pc.fb, x);
    let lam = g.map(&[mel, fx], |v| v[0] - (v[1] + lit(1e-30)).ln()).unwrap();
    (e, lam)
}

/// `s = x ⊙ exp(F λ)`, the exponent capped at 80.
fn s_of(g: &mut TensorGraph, pc: &PConsts, x: TensorId, lam: TensorId) -> TensorId {
    let fl = to_bins(g, pc.fb, lam);
    g.map(&[x, fl], |v| {
        let z = ExprId::select(v[1] - lit(80.0), lit(80.0), v[1]);
        v[0] * z.exp()
    })
    .unwrap()
}

/// One Newton step on the dual, `λ + clamp(H⁻¹(e − Fᵀs), ±4)` with
/// `H = Fᵀ diag(s) F`; `detach_h` stops the gradient through `H`.
fn newton_step(g: &mut TensorGraph, pc: &PConsts, x: TensorId, e: TensorId, lam: TensorId, detach_h: bool, keep: &mut Vec<TensorId>) -> TensorId {
    let s = s_of(g, pc, x, lam);
    let fs = to_bands(g, pc.fb, s);
    let mut d = to_bands(g, pc.fb2, s);
    let mut up = to_bands(g, pc.fbn, s);
    if detach_h {
        d = g.detach(d).unwrap();
        up = g.detach(up).unwrap();
    }
    let r = g.map(&[e, fs], |v| v[0] - v[1]).unwrap();
    keep.push(s);
    if !detach_h {
        keep.push(lam);
    }
    let step = pcr(g, pc.valid, up, d, r, keep);
    g.map(&[lam, step], |v| {
        let c = ExprId::select(v[1] - lit(4.0), lit(4.0), ExprId::select(lit(-4.0) - v[1], lit(-4.0), v[1]));
        v[0] + c
    })
    .unwrap()
}

/// RNDVoC's range `|F⁺ e|` (+ε): `z = (FᵀF)⁻¹ e`, `r = F z`.
pub fn pinv_range(g: &mut TensorGraph, pc: &PConsts, mel: TensorId) -> TensorId {
    let [b, _, t] = ir::dims(g, mel)[..] else { panic!() };
    let k = ir::dims(g, pc.fb)[0];
    let ones = g.comp(&[b, k, t], &[], vec![arg(pc.valid, view(vec![form(3, &[], 0)], false))], None, |v| vec![v[0]]).unwrap();
    let d = to_bands(g, pc.fb2, ones);
    let up = to_bands(g, pc.fbn, ones);
    let e = g.map(&[mel], |v| v[0].exp()).unwrap();
    let z = pcr(g, pc.valid, up, d, e, &mut Vec::new());
    let r = to_bins(g, pc.fb, z);
    g.map(&[r], |v| (v[0] * v[0] + lit(1e-14)).sqrt()).unwrap()
}

/// Physical magnitude `[B, bins, T]` from params `th` `[B, P, T]` and
/// `f0` `[B, 1, T]` (Hz, < 40 = unvoiced).
pub fn phys_mag(g: &mut TensorGraph, cfg: &Config, pc: &PConsts, th: TensorId, f0: TensorId) -> TensorId {
    let [b, _, t] = ir::dims(g, th)[..] else { panic!() };
    let (k, q) = (cfg.bins(), cfg.n_cep);
    // ln H[b,k,t] = Σ_q c_q[b,t]·cep[q,k]; space (b, k, t, q)
    let logh = g
        .comp(
            &[b, k, t],
            &[q],
            vec![arg(th, view(vec![form(4, &[(0, 1)], 0), form(4, &[(3, 1)], P_CEP as i64), form(4, &[(2, 1)], 0)], false)), arg(pc.cep, pick(4, &[Some(3), Some(1)]))],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let ch = |c: usize| view(vec![form(3, &[(0, 1)], 0), form(3, &[], c as i64), form(3, &[(2, 1)], 0)], false);
    let df = cfg.sr as f64 / cfg.n_fft as f64;
    g.comp(
        &[b, k, t],
        &[],
        vec![arg(th, ch(P_OQ)), arg(th, ch(P_TILT)), arg(th, ch(P_ASP)), arg(f0, ch(0)), arg(pc.freqs, pick(3, &[Some(1)])), arg(logh, View::identity(3, 3))],
        None,
        |v| {
            let (oq, fa, an, f0, f, lh) = (oq_of(v[0]), v[1].exp(), v[2].exp(), v[3], v[4], v[5]);
            let voiced = v[3] - lit(dsp::F0_MIN);
            let f0s = ExprId::select(voiced, f0, lit(100.0));
            let fg = ExprId::select(voiced, f0s / (lit(2.0) * oq), lit(200.0));
            let u = f / fg;
            let gl = u / (lit(1.0) + u * u) / (lit(1.0) + (f / fa) * (f / fa)).sqrt();
            let kap = (f0s / lit(2.0 * std::f64::consts::PI * COMB_SIGMA * df)) * (f0s / lit(2.0 * std::f64::consts::PI * COMB_SIGMA * df));
            let comb = (kap * ((lit(2.0 * std::f64::consts::PI) * f / f0s).cos() - lit(1.0))).exp();
            let comb = ExprId::select(voiced, comb, lit(0.0));
            vec![lh.exp() * gl * (comb * comb + an * an).sqrt() + lit(1e-7)]
        },
    )
    .unwrap()
}

/// `ln max(Fᵀ s, 1e-5)`.
pub fn log_mel_of(g: &mut TensorGraph, pc: &PConsts, s: TensorId) -> TensorId {
    let m = to_bands(g, pc.fb, s);
    g.map(&[m], |v| ExprId::select(v[0] - lit(1e-5), v[0], lit(1e-5)).ln()).unwrap()
}

/// `max |a − b|` as `[1]`.
pub fn max_abs_diff(g: &mut TensorGraph, a: TensorId, b: TensorId) -> TensorId {
    let sh = ir::dims(g, a);
    let r = sh.len();
    let picks: Vec<Option<usize>> = (1..=r).map(Some).collect();
    g.comp(&[1], &sh, vec![arg(a, pick(r + 1, &picks)), arg(b, pick(r + 1, &picks))], Some(Monoid::max()), |v| vec![ir::abs(v[0] - v[1])]).unwrap()
}

/// Analysis-by-synthesis: one Adam step on the per-frame params.
pub struct FitGraph {
    pub b: Built,
    pub th: TensorId,
    pub m: TensorId,
    pub v: TensorId,
    pub mel: TensorId,
    pub f0: TensorId,
    pub hyper: TensorId,
    /// `[loss, mel_rmse]`, then th′, m′, v′
    pub outputs: Vec<TensorId>,
}

pub fn fit_graph(cfg: &Config, batch: usize, frames: usize) -> FitGraph {
    let mut b = Built::new_pub(cfg);
    let np = n_params(cfg);
    let pc = pconsts(&mut b);
    let th = b.input(&[batch, np, frames], Role::Data("phys"));
    let m = b.input(&[batch, np, frames], Role::Data("m"));
    let v = b.input(&[batch, np, frames], Role::Data("v"));
    let mel = b.input(&[batch, cfg.n_mels, frames], Role::Data("mel"));
    let f0 = b.input(&[batch, 1, frames], Role::Data("f0hz"));
    let hyper = b.input(&[6], Role::Hyper);
    let g = &mut b.g;
    let x = phys_mag(g, cfg, &pc, th, f0);
    let lm = log_mel_of(g, &pc, x);
    let n = (batch * cfg.n_mels * frames) as f64;
    let sse = ir::sum_all2(g, lm, Some(mel), |v| (v[0] - v[1]) * (v[0] - v[1]));
    // weak priors: identify the source params only where the mel says so
    let nf = (batch * frames) as f64;
    let ch = |c: usize| view(vec![form(3, &[(1, 1)], 0), form(3, &[], c as i64), form(3, &[(2, 1)], 0)], false);
    let prior = g
        .comp(&[1], &[batch, frames], vec![arg(th, ch(P_OQ)), arg(th, ch(P_TILT)), arg(th, ch(P_ASP))], Some(Monoid::sum()), |v| {
            vec![v[0] * v[0] + (v[1] - lit(PRIOR_TILT)) * (v[1] - lit(PRIOR_TILT)) + (v[2] - lit(PRIOR_ASP)) * (v[2] - lit(PRIOR_ASP))]
        })
        .unwrap();
    let loss = g.map(&[sse, prior], |v| v[0] * lit(1.0 / n) + v[1] * lit(1e-3 / nf)).unwrap();
    let rmse = g.map(&[sse], |v| (v[0] * lit(1.0 / n)).sqrt()).unwrap();
    // gradient scaled to per-frame (frames are independent problems)
    let gr = g.backward(loss, &[th]).unwrap()[0];
    let gr = g.map(&[gr], |v| v[0] * lit(n / cfg.n_mels as f64)).unwrap();
    let id = View::identity(3, 3);
    let hy = |k: i64| view(vec![form(3, &[], k)], false);
    let m2 = g.comp(&[batch, np, frames], &[], vec![arg(m, id.clone()), arg(gr, id.clone())], None, |x| vec![x[0] * lit(0.9) + x[1] * lit(0.1)]).unwrap();
    let v2 = g.comp(&[batch, np, frames], &[], vec![arg(v, id.clone()), arg(gr, id.clone())], None, |x| vec![x[0] * lit(0.999) + x[1] * x[1] * lit(0.001)]).unwrap();
    let th2 = g
        .comp(&[batch, np, frames], &[], vec![arg(th, id.clone()), arg(m2, id.clone()), arg(v2, id), arg(hyper, hy(0)), arg(hyper, hy(1)), arg(hyper, hy(2))], None, |x| {
            vec![x[0] - x[3] * (x[1] / x[4]) / ((x[2] / x[5]).sqrt() + lit(1e-8))]
        })
        .unwrap();
    let outputs = vec![loss, rmse, th2, m2, v2];
    FitGraph { b, th, m, v, mel, f0, hyper, outputs }
}

/// Host init for the fit: c₀ from each frame's mean log-mel level, the
/// source params at their priors.
pub fn init_params(cfg: &Config, mel: &[f32], batch: usize, frames: usize) -> Vec<f32> {
    let np = n_params(cfg);
    let mut th = vec![0.0f32; batch * np * frames];
    for bb in 0..batch {
        for t in 0..frames {
            let mean = (0..cfg.n_mels).map(|m| mel[(bb * cfg.n_mels + m) * frames + t]).sum::<f32>() / cfg.n_mels as f32;
            let at = |c: usize| (bb * np + c) * frames + t;
            th[at(P_TILT)] = PRIOR_TILT as f32;
            th[at(P_ASP)] = PRIOR_ASP as f32;
            th[at(P_CEP)] = mean + 3.0;
        }
    }
    th
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Rng;

    /// Builds `L = Σ w·project(phys(θ)·exp(a), mel)` (+ the pinv range),
    /// checks exactness and the derived gradient vs central FD, all in the
    /// f64 interpreter.
    #[test]
    fn project_exact_and_grads() {
        check(false, 1e-5);
    }

    /// The same checks on [`project_implicit`]; then its gradients equal
    /// the ones unrolled through every Newton step, with a smaller backward.
    /// On this case the two agree to 4e-7 of each gradient's scale (the
    /// unrolled one is itself 5e-7 from FD), not to rounding, and one
    /// near-zero coordinate puts the implicit FD figure at 2e-5.
    #[test]
    fn implicit_project_matches_unrolled() {
        let (gi, ni) = check(true, 1e-4);
        let (gu, nu) = check(false, 1e-5);
        let mut worst = 0.0f64;
        for (a, b) in gi.iter().zip(&gu) {
            let sc = b.iter().fold(1e-3f64, |m, v| m.max(v.abs()));
            for (u, v) in a.iter().zip(b) {
                worst = worst.max((u - v).abs() / sc);
            }
        }
        println!("implicit vs unrolled gradients (f64): worst rel {worst:.2e}; backward nodes {ni} vs {nu}");
        assert!(worst < 1e-5, "{worst}");
        assert!(nu > 3 * ni);
    }

    /// Returns the gradients w.r.t. `[θ, a, mel]` and the live node count
    /// the backward adds to the forward.
    fn check(implicit: bool, fd_tol: f64) -> (Vec<Vec<f64>>, usize) {
        let project = |g: &mut TensorGraph, pc: &PConsts, x: TensorId, mel: TensorId| {
            if implicit { project_implicit(g, pc, x, mel, &mut Vec::new()) } else { super::project(g, pc, x, mel) }
        };
        let cfg = Config::by_name("checkf0phys").unwrap();
        let (k, m, t, np) = (cfg.bins(), cfg.n_mels, 3usize, n_params(&cfg));
        let mut b = Built::new_pub(&cfg);
        let pc = pconsts(&mut b);
        let th = b.input(&[1, np, t], Role::Data("phys"));
        let f0 = b.input(&[1, 1, t], Role::Data("f0hz"));
        let a = b.input(&[1, k, t], Role::Data("a"));
        let mel = b.input(&[1, m, t], Role::Data("mel"));
        let w = b.input(&[1, k, t], Role::Data("w"));
        let g = &mut b.g;
        let x = phys_mag(g, &cfg, &pc, th, f0);
        let cand = g.map(&[x, a], |v| v[0] * v[1].exp()).unwrap();
        let s = project(g, &pc, cand, mel);
        let r = pinv_range(g, &pc, mel);
        let s2 = project(g, &pc, r, mel);
        let lm = log_mel_of(g, &pc, s);
        let lm2 = log_mel_of(g, &pc, s2);
        let e1 = max_abs_diff(g, lm, mel);
        let e2 = max_abs_diff(g, lm2, mel);
        let l1 = ir::sum_all2(g, s, Some(w), |v| v[0] * v[1]);
        let l2 = ir::sum_all2(g, s2, Some(w), |v| v[0] * v[1]);
        let loss = g.map(&[l1, l2], |v| v[0] + v[1]).unwrap();
        let wrt = [th, a, mel];
        let grads = g.backward(loss, &wrt).unwrap();
        // live nodes of loss + gradients, minus the forward's
        let mut lg = vec![loss];
        lg.extend(&grads);
        let added = g.eliminate_dead(&lg).0.nodes().len() - g.eliminate_dead(&[loss]).0.nodes().len();
        let mut rng = Rng::new(3);
        let mut thv = vec![0.0f64; np * t];
        for tt in 0..t {
            thv[P_OQ * t + tt] = 0.3 * rng.normal();
            thv[P_TILT * t + tt] = 7.5 + 0.3 * rng.normal();
            thv[P_ASP * t + tt] = -2.0 + 0.5 * rng.normal();
            for q in 0..cfg.n_cep {
                thv[(P_CEP + q) * t + tt] = if q == 0 { -1.0 } else { 0.3 * rng.normal() / (q as f64) };
            }
        }
        let f0v = vec![130.0, 0.0, 211.0];
        let av: Vec<f64> = (0..k * t).map(|_| 0.3 * rng.normal()).collect();
        // a feasible target: the mel of a random positive spectrum
        let fbh = dsp::mel_fbank(cfg.sr, cfg.n_fft, m);
        let ys: Vec<f64> = (0..k * t).map(|_| (0.5 * rng.normal() - 2.0).exp()).collect();
        let melv: Vec<f64> = (0..m * t).map(|i| { let (mm, tt) = (i / t, i % t); (0..k).map(|kk| fbh[kk * m + mm] * ys[kk * t + tt]).sum::<f64>().ln() }).collect();
        let wv: Vec<f64> = (0..k * t).map(|_| rng.normal()).collect();
        let mut inputs: Vec<Vec<f64>> = b.inputs.iter().map(|s| match &s.role {
            Role::Const(v) => v.iter().map(|&x| x as f64).collect(),
            _ => vec![],
        }).collect();
        let set = |inputs: &mut Vec<Vec<f64>>, id: TensorId, v: &[f64]| {
            let i = b.input_index(id);
            inputs[i] = v.to_vec();
        };
        // constants were rounded to f32 in Role::Const; recompute exactly
        set(&mut inputs, th, &thv);
        set(&mut inputs, f0, &f0v);
        set(&mut inputs, a, &av);
        set(&mut inputs, mel, &melv);
        set(&mut inputs, w, &wv);
        let mut outs = vec![loss, e1, e2];
        outs.extend(&grads);
        let r0 = b.g.eval(&inputs, &outs);
        println!("exactness (f64): physics {:.2e}, pinv {:.2e}", r0[1][0], r0[2][0]);
        assert!(r0[1][0] < 1e-9 && r0[2][0] < 1e-9);
        let mut worst = 0.0f64;
        for (gi, &id) in wrt.iter().enumerate() {
            let idx = b.input_index(id);
            let n = inputs[idx].len();
            for j in (0..n).step_by((n / 12).max(1)) {
                let h = 1e-6;
                let mut ip = inputs.clone();
                ip[idx][j] += h;
                let mut im = inputs.clone();
                im[idx][j] -= h;
                let fd = (b.g.eval(&ip, &[loss])[0][0] - b.g.eval(&im, &[loss])[0][0]) / (2.0 * h);
                let an = r0[3 + gi][j];
                let e = (fd - an).abs() / (fd.abs().max(an.abs()).max(1e-3));
                worst = worst.max(e);
            }
        }
        println!("FD vs derived backward (f64, implicit {implicit}): worst rel {worst:.2e}");
        assert!(worst < fd_tol, "{worst}");
        (r0[3..].to_vec(), added)
    }
}
