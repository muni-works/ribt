//! V1.1 checks through tang's f64 interpreter on tiny configs: the
//! generator step's gradient (V1 losses + adversarial + feature matching,
//! with the F0 source) and the discriminator step's gradient vs central
//! finite differences, and the YIN estimator on a synthetic voice.

use ribt::dsp;
use ribt::gan::{self, DConfig, Weights};
use ribt::model::{Config, ParamSpec, Rng};
use tang_compile::tensor::{TensorGraph, TensorId};

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn signal(rng: &mut Rng, n: usize, ph: f64) -> Vec<f32> {
    (0..n).map(|i| {
        let t = i as f64;
        (0.5 * (0.31 * t + ph).sin() + 0.3 * (1.7 * t + 0.002 * t * t).sin() + 0.1 * rng.normal()) as f32
    }).collect()
}

/// Analytic gradient of `loss` w.r.t. `wrt` vs central differences at 2
/// random coordinates per tensor; returns the worst relative error.
fn fd_check(g: &mut TensorGraph, base: &[Vec<f64>], loss: TensorId, wrt: &[(TensorId, usize, String)], rng: &mut Rng) -> f64 {
    let ids: Vec<TensorId> = wrt.iter().map(|w| w.0).collect();
    let grads = g.backward(loss, &ids).unwrap();
    let mut outs = vec![loss];
    outs.extend(&grads);
    let res = g.eval(base, &outs);
    eprintln!("loss {:.6}", res[0][0]);
    let mut worst: f64 = 0.0;
    for (wi, (_, k, name)) in wrt.iter().enumerate() {
        let n = base[*k].len();
        for _ in 0..2 {
            let j = rng.below(n);
            let h = 1e-6;
            let mut p = base.to_vec();
            p[*k][j] += h;
            let mut m = base.to_vec();
            m[*k][j] -= h;
            let fd = (g.eval(&p, &[loss])[0][0] - g.eval(&m, &[loss])[0][0]) / (2.0 * h);
            let an = res[1 + wi][j];
            let rel = (fd - an).abs() / fd.abs().max(an.abs()).max(1e-4);
            if rel > 1e-3 {
                eprintln!("  {name}[{j}]: analytic {an:.6e} fd {fd:.6e} rel {rel:.1e}");
            }
            worst = worst.max(rel);
        }
    }
    eprintln!("{} tensors, worst rel {worst:.2e}", wrt.len());
    worst
}

fn perturbed(specs: &[ParamSpec], seed: u64, rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut r2 = Rng::new(seed);
    specs.iter().map(|s| s.init_data(&mut r2).iter().map(|v| v + (0.1 * rng.normal()) as f32).collect()).collect()
}

/// D params at their weight-norm init, perturbed (gains and biases too).
fn perturbed_d(d: &DConfig, seed: u64, rng: &mut Rng) -> Vec<Vec<f32>> {
    gan::init_dparams(d, seed).into_iter().map(|p| p.iter().map(|v| v + (0.1 * rng.normal()) as f32).collect()).collect()
}

#[test]
fn gen_step_gradient() {
    let cfg = Config::by_name("tinyf0").unwrap();
    let d = DConfig::tiny();
    let (batch, frames) = (2, 8);
    let n = frames * cfg.hop;
    let mut gg = gan::gen_graph(&cfg, &d, &Weights::default(), batch, frames);
    let mut rng = Rng::new(21);
    let params = perturbed(&cfg.params(), 7, &mut rng);
    let dparams = perturbed_d(&d, 8, &mut rng);
    let (mut xp, mut x) = (Vec::new(), Vec::new());
    for i in 0..batch {
        let s = signal(&mut rng, n, i as f64);
        xp.extend(dsp::reflect_pad(&s, cfg.mel_pad()));
        x.extend(s);
    }
    let f0: Vec<f32> = (0..batch * frames).map(|i| if i % 3 == 0 { 0.0 } else { 300.0 + 40.0 * i as f32 }).collect();
    let f0 = dsp::source_feats(&f0, batch, frames, cfg.sr, cfg.n_fft, cfg.n_mels);
    let b = &gg.lg.b;
    let mut base: Vec<Vec<f64>> = b.assemble_gd(&params, None, &dparams, None, &[("audio_pad", &xp), ("audio", &x), ("src", &f0)]).iter().map(|v| f64s(v)).collect();
    // adversarial + FM terms on (hyper[5] is their ramp scale)
    base[b.input_index(gg.hyper)][5] = 1.0;
    let specs = cfg.params();
    let wrt: Vec<(TensorId, usize, String)> = gg.lg.params.iter().zip(&specs).map(|(&t, s)| (t, b.input_index(t), s.name.clone())).collect();
    let loss = gg.outputs[0];
    let worst = fd_check(&mut gg.lg.b.g, &base, loss, &wrt, &mut rng);
    assert!(worst < 1e-3, "{worst}");
}

#[test]
fn disc_step_gradient() {
    disc_grad(DConfig::tiny());
    let mut d = DConfig::tiny();
    d.wn = false;
    d.bias = false;
    disc_grad(d);
}

fn disc_grad(d: DConfig) {
    let cfg = Config::tiny();
    let (batch, frames) = (2, 8);
    let n = frames * cfg.hop;
    let mut dg = gan::disc_graph(&cfg, &d, &Weights::default(), batch, frames);
    let mut rng = Rng::new(22);
    let dparams = perturbed_d(&d, 9, &mut rng);
    let x: Vec<f32> = (0..batch).flat_map(|i| signal(&mut rng, n, i as f64)).collect();
    let y: Vec<f32> = (0..batch).flat_map(|i| signal(&mut rng, n, 2.0 + i as f64)).map(|v| 0.7 * v).collect();
    let b = &dg.b;
    let base: Vec<Vec<f64>> = b.assemble_gd(&[], None, &dparams, None, &[("audio", &x), ("fake", &y)]).iter().map(|v| f64s(v)).collect();
    let wrt: Vec<(TensorId, usize, String)> = dg.dparams.iter().zip(&b.dspecs).map(|(&t, s)| (t, b.input_index(t), s.name.clone())).collect();
    let loss = dg.outputs[0];
    let worst = fd_check(&mut dg.b.g, &base, loss, &wrt, &mut rng);
    assert!(worst < 1e-3, "{worst}");
}

#[test]
fn yin_tracks_a_harmonic_voice() {
    let (sr, n_fft, hop, frames) = (24000, 1024, 256, 40);
    let pad = (n_fft - hop) / 2;
    let n = frames * hop;
    let f = |i: usize| 120.0 + 60.0 * i as f64 / n as f64; // 120 → 180 Hz glide
    let mut ph = 0.0;
    let x: Vec<f32> = (0..n)
        .map(|i| {
            ph += 2.0 * std::f64::consts::PI * f(i) / sr as f64;
            (1..8).map(|h| (h as f64 * ph).sin() / h as f64).sum::<f64>() as f32 * 0.3
        })
        .collect();
    let xp = dsp::reflect_pad(&x, pad);
    let est = dsp::yin_f0(&xp, sr, n_fft, hop, frames);
    let mut worst: f64 = 0.0;
    for t in 2..frames - 2 {
        let want = f(t * hop + hop / 2);
        worst = worst.max((est[t] as f64 - want).abs() / want);
    }
    eprintln!("yin worst rel err {worst:.3} ({:?})", &est[..6]);
    assert!(worst < 0.03, "{worst}");
    let silent = dsp::yin_f0(&vec![0.0; xp.len()], sr, n_fft, hop, frames);
    assert!(silent.iter().all(|&v| v == 0.0));
}

/// The optimizer's passes (f64 interpreter on the optimized graph) vs the
/// raw graph, on the D step: localizes GPU/interpreter disagreements to a pass.
#[test]
#[ignore]
fn disc_passes_preserve_semantics() {
    let cfg = Config::by_name("checkf0").unwrap();
    let d = DConfig::by_name(&std::env::var("DCFG").unwrap_or("check".into())).unwrap();
    let (batch, frames) = (2, 16);
    let n = frames * cfg.hop;
    let dg = gan::disc_graph(&cfg, &d, &Weights::default(), batch, frames);
    let mut rng = Rng::new(4);
    let dparams = perturbed_d(&d, 2, &mut rng);
    let x: Vec<f32> = (0..batch).flat_map(|i| signal(&mut rng, n, i as f64)).collect();
    let y: Vec<f32> = x.iter().map(|v| 0.8 * v + 0.01 * rng.normal() as f32).collect();
    let b = &dg.b;
    let mut base: Vec<Vec<f64>> = b.assemble_gd(&[], None, &dparams, None, &[("audio", &x), ("fake", &y)]).iter().map(|v| f64s(v)).collect();
    base[b.input_index(dg.hyper)] = vec![1e-3, 0.2, 0.1, 0.01, 1e9, 1.0];
    let want = b.g.eval(&base, &dg.outputs);
    let passes: Vec<(&str, fn(&TensorGraph, &[TensorId]) -> (TensorGraph, Vec<TensorId>))> = vec![
        ("substitute_maps", |g, o| g.substitute_maps(o)),
        ("eliminate_common", |g, o| g.eliminate_common(o)),
        ("merge_siblings", |g, o| g.merge_siblings(o)),
        ("relayout", |g, o| g.relayout(o)),
        ("merge_reduce_dims", |g, o| g.merge_reduce_dims(o)),
    ];
    let (mut g, mut outs) = (b.g.clone(), dg.outputs.clone());
    for (name, p) in passes {
        let (g2, o2) = p(&g, &outs);
        g = g2;
        outs = o2;
        let got = g.eval(&base, &outs);
        let mut worst = (0.0f64, 0usize);
        for (i, (a, w)) in got.iter().zip(&want).enumerate() {
            let sc = w.iter().fold(1e-9f64, |m, v| m.max(v.abs()));
            let e = a.iter().zip(w).map(|(a, w)| (a - w).abs()).fold(0.0, f64::max) / sc;
            if e > worst.0 {
                worst = (e, i);
            }
        }
        eprintln!("after {name}: worst rel {:.2e} at output {}", worst.0, worst.1);
    }
}
