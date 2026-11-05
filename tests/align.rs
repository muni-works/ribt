//! V2 aligner checks (docs §13): the host forward-sum/binarization gradient
//! vs finite differences, the beta-binomial prior vs a direct pmf, MAS vs
//! brute force, and the full aligner gradient (IR graph through the host
//! loss) vs finite differences in the f64 interpreter.

use ribt::align::{self, AConfig};
use ribt::model::Rng;

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

#[test]
fn prior_rows_are_beta_binomial() {
    let (n, t) = (6, 11);
    let lp = align::log_prior(n, t, 1.0);
    for ti in 0..t {
        let i = (ti + 1) as f64;
        let (a, b) = (i, t as f64 + 1.0 - i);
        let nn = (n - 1) as f64;
        let lb = |x: f64, y: f64| align::lgamma(x) + align::lgamma(y) - align::lgamma(x + y);
        let mut s = 0.0;
        for k in 0..n {
            let kf = k as f64;
            let lc = align::lgamma(nn + 1.0) - align::lgamma(kf + 1.0) - align::lgamma(nn - kf + 1.0);
            let want = lc + lb(kf + a, nn - kf + b) - lb(a, b);
            assert!((lp[ti * n + k] - want).abs() < 1e-10, "t{ti} k{k}: {} vs {want}", lp[ti * n + k]);
            s += want.exp();
        }
        assert!((s - 1.0).abs() < 1e-10);
    }
    assert!((align::lgamma(5.0) - 24f64.ln()).abs() < 1e-13);
    assert!((align::lgamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-13);
}

/// All monotonic paths, enumerated: log Z and the Viterbi path.
fn brute(lp: &[f64], n: usize, t: usize) -> (f64, Vec<u32>) {
    let mut best = (f64::NEG_INFINITY, vec![]);
    let mut terms = vec![];
    // choose which of frames 1..t advance (exactly n−1 of them)
    for mask in 0u32..(1 << (t - 1)) {
        if mask.count_ones() as usize != n - 1 {
            continue;
        }
        let mut ni = 0usize;
        let mut path = vec![0u32];
        let mut s = lp[0];
        for ti in 1..t {
            if mask >> (ti - 1) & 1 == 1 {
                ni += 1;
            }
            path.push(ni as u32);
            s += lp[ti * n + ni];
        }
        terms.push(s);
        if s > best.0 {
            best = (s, path);
        }
    }
    let m = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (m + terms.iter().map(|x| (x - m).exp()).sum::<f64>().ln(), best.1)
}

#[test]
fn forward_sum_and_mas_match_brute_force() {
    let mut rng = Rng::new(3);
    for &(n, t) in &[(1usize, 4usize), (3, 3), (3, 8), (4, 10)] {
        let z: Vec<f64> = (0..n * t).map(|_| rng.normal() * 2.0).collect();
        let prior = align::log_prior(n, t, 1.0);
        let lp = align::log_attn(&|ni, ti| z[ni * t + ti], n, t, &prior);
        let (lz, gamma) = align::forward_sum(&lp, n, t);
        let (bz, bpath) = brute(&lp, n, t);
        assert!((lz - bz).abs() < 1e-10, "{n}x{t}: {lz} vs {bz}");
        assert_eq!(align::mas(&lp, n, t), bpath);
        for ti in 0..t {
            let s: f64 = (0..n).map(|ni| gamma[ti * n + ni]).sum();
            assert!((s - 1.0).abs() < 1e-10);
        }
    }
}

#[test]
fn host_gradient_matches_finite_differences() {
    let mut rng = Rng::new(7);
    let (n, t) = (5, 13);
    let z: Vec<f64> = (0..n * t).map(|_| rng.normal()).collect();
    let prior = align::log_prior(n, t, 1.0);
    for &w_bin in &[0.0, 1.0] {
        let f = |z: &[f64]| {
            let o = align::utt_host(&|ni, ti| z[ni * t + ti], n, t, &prior, w_bin, None);
            o.loss_fs + w_bin * o.loss_bin
        };
        let o = align::utt_host(&|ni, ti| z[ni * t + ti], n, t, &prior, w_bin, None);
        let mut worst: f64 = 0.0;
        for j in 0..n * t {
            let h = 1e-6;
            let mut p = z.clone();
            p[j] += h;
            let mut m = z.clone();
            m[j] -= h;
            let fd = (f(&p) - f(&m)) / (2.0 * h);
            let an = o.grad[j];
            worst = worst.max((fd - an).abs() / fd.abs().max(an.abs()).max(1e-6));
        }
        eprintln!("host w_bin {w_bin}: L_fs {:.5} L_bin {:.5} worst rel {worst:.2e}", o.loss_fs, o.loss_bin);
        assert!(worst < 1e-5, "{worst}");
    }
}

/// The whole aligner: θ → IR graph → z → host loss. Analytic gradient is
/// the IR backward of the surrogate Σ z ⊙ G(z); FD perturbs θ through both.
#[test]
fn aligner_gradient_matches_finite_differences() {
    let ac = AConfig::tiny();
    let (bsz, nmax, tmax) = (2, 4, 9);
    let lens = [(4usize, 9usize), (3, 7)];
    let mut tg = align::train_graph(&ac, bsz, nmax, tmax);
    let specs = ac.params();
    let mut rng = Rng::new(11);
    let params: Vec<Vec<f32>> = align::init_params(&ac, 5).into_iter().map(|p| p.iter().map(|v| v + (0.2 * rng.normal()) as f32).collect()).collect();
    let toks: Vec<Vec<u16>> = vec![vec![0, 3, 5, 0], vec![0, 6, 0]];
    let rows: Vec<&[u16]> = toks.iter().map(|r| r.as_slice()).collect();
    let oh = align::onehot(&rows, ac.vocab, nmax);
    let mel: Vec<f32> = (0..bsz * ac.n_mels * tmax).map(|_| rng.normal() as f32).collect();
    for &(w_bin, blank) in &[(0.0, None), (0.5, None), (0.0, Some(-1.0)), (0.5, Some(-1.0))] {
        let loss_of = |inputs: &[Vec<f64>], g: &tang_compile::tensor::TensorGraph, z: tang_compile::tensor::TensorId| {
            let zv: Vec<f32> = g.eval(inputs, &[z])[0].iter().map(|&v| v as f32).collect();
            // f64 path for the host: re-evaluate per utterance from f64 z
            let zz = &g.eval(inputs, &[z])[0];
            let mut l = 0.0;
            for (b, &(n, t)) in lens.iter().enumerate() {
                let prior = align::log_prior(n, t, 1.0);
                let o = align::utt_host(&|ni, ti| zz[(b * nmax + ni) * tmax + ti], n, t, &prior, w_bin, blank);
                l += (o.loss_fs + w_bin * o.loss_bin) / bsz as f64;
            }
            (l, zv)
        };
        let base: Vec<Vec<f64>> = align::assemble(&tg.ag.b, &specs, &params, None, &[("onehot", &oh), ("mel", &mel)]).iter().map(|v| f64s(v)).collect();
        let (l0, _) = loss_of(&base, &tg.ag.b.g, tg.ag.z);
        // host G in f64, fed as data
        let zz = tg.ag.b.g.eval(&base, &[tg.ag.z])[0].clone();
        let mut gz = vec![0.0f64; bsz * nmax * tmax];
        for (b, &(n, t)) in lens.iter().enumerate() {
            let prior = align::log_prior(n, t, 1.0);
            let o = align::utt_host(&|ni, ti| zz[(b * nmax + ni) * tmax + ti], n, t, &prior, w_bin, blank);
            for ni in 0..n {
                for ti in 0..t {
                    gz[(b * nmax + ni) * tmax + ti] = o.grad[ni * t + ti] / bsz as f64;
                }
            }
        }
        let mut inp = base.clone();
        let k_gz = tg.ag.b.input_index(tg.gz);
        inp[k_gz] = gz;
        let grads = tg.grads.clone();
        let an = tg.ag.b.g.eval(&inp, &grads);
        let mut worst: f64 = 0.0;
        for (i, &pid) in tg.ag.params.clone().iter().enumerate() {
            let k = tg.ag.b.input_index(pid);
            for _ in 0..3 {
                let j = rng.below(base[k].len());
                let h = 1e-6;
                let mut p = base.clone();
                p[k][j] += h;
                let mut m = base.clone();
                m[k][j] -= h;
                let fd = (loss_of(&p, &tg.ag.b.g, tg.ag.z).0 - loss_of(&m, &tg.ag.b.g, tg.ag.z).0) / (2.0 * h);
                let a = an[i][j];
                let rel = (fd - a).abs() / fd.abs().max(a.abs()).max(1e-6);
                if rel > 1e-4 {
                    eprintln!("  {}[{j}]: analytic {a:.6e} fd {fd:.6e} rel {rel:.1e}", specs[i].name);
                }
                worst = worst.max(rel);
            }
        }
        eprintln!("aligner w_bin {w_bin} blank {blank:?}: loss {l0:.6}, {} tensors × 3 coords, worst rel {worst:.2e}", specs.len());
        assert!(worst < 1e-4, "{worst}");
    }
    let _ = &mut tg;
}

fn zf(z: &[f64], t: usize) -> impl Fn(usize, usize) -> f64 + '_ {
    move |ni, ti| z[ni * t + ti]
}

#[test]
fn ctc_forward_sum_gradient_and_limit() {
    let mut rng = Rng::new(9);
    let (n, t) = (4, 11);
    let z: Vec<f64> = (0..n * t).map(|_| rng.normal()).collect();
    let prior = align::log_prior(n, t, 1.0);
    for &blank in &[-1.0, 0.5] {
        let (_, g) = align::forward_sum_ctc(&zf(&z, t), n, t, &prior, blank);
        let f = |z: &[f64]| -align::forward_sum_ctc(&zf(z, t), n, t, &prior, blank).0;
        let mut worst: f64 = 0.0;
        for ti in 0..t {
            for ni in 0..n {
                let h = 1e-6;
                let mut p = z.clone();
                p[ni * t + ti] += h;
                let mut m = z.clone();
                m[ni * t + ti] -= h;
                let fd = (f(&p) - f(&m)) / (2.0 * h);
                let an = g[ti * n + ni];
                worst = worst.max((fd - an).abs() / fd.abs().max(an.abs()).max(1e-6));
            }
        }
        eprintln!("ctc blank {blank}: worst rel {worst:.2e}");
        assert!(worst < 1e-5, "{worst}");
    }
    // blank → −∞ recovers the blank-free forward-sum
    let lp = align::log_attn(&zf(&z, t), n, t, &prior);
    let (want, _) = align::forward_sum(&lp, n, t);
    let (got, _) = align::forward_sum_ctc(&zf(&z, t), n, t, &prior, -200.0);
    assert!((want - got).abs() < 1e-9, "{want} vs {got}");
    let o = align::utt_host(&zf(&z, t), n, t, &prior, 0.3, Some(-1.0));
    let fo = |z: &[f64]| {
        let o = align::utt_host(&zf(z, t), n, t, &prior, 0.3, Some(-1.0));
        o.loss_fs + 0.3 * o.loss_bin
    };
    let mut worst: f64 = 0.0;
    for j in 0..n * t {
        let h = 1e-6;
        let mut p = z.clone();
        p[j] += h;
        let mut m = z.clone();
        m[j] -= h;
        let fd = (fo(&p) - fo(&m)) / (2.0 * h);
        worst = worst.max((fd - o.grad[j]).abs() / fd.abs().max(o.grad[j].abs()).max(1e-6));
    }
    eprintln!("utt_host ctc + bin: worst rel {worst:.2e}");
    assert!(worst < 1e-5);
}
