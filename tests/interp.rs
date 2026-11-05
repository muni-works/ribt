//! CPU-only checks on the tiny config through tang's f64 interpreter:
//! shapes, iSTFT(STFT(x)) round trip, the mel frontend against an f64
//! reference, and the full training graph's gradient vs finite differences.

use ribt::dsp;
use ribt::model::{self, Config, Rng};

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn signal(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            (0.5 * (0.31 * t).sin() + 0.3 * (1.7 * t + 0.002 * t * t).sin() + 0.1 * rng.normal()) as f32
        })
        .collect()
}

#[test]
fn shapes() {
    let cfg = Config::v1();
    let inf = model::infer_graph(&cfg, 1, 94);
    let sh: Vec<usize> = inf.b.g.shape(inf.audio).iter().map(|d| d.n()).collect();
    assert_eq!(sh, vec![1, 94 * 256]);
    let n = cfg.n_params();
    eprintln!("v1 params: {n}");
    assert!((10_000_000..=15_000_000).contains(&n), "{n}");
    let fe = model::frontend_graph(&cfg, 2, 64);
    let sh: Vec<usize> = fe.b.g.shape(fe.mel).iter().map(|d| d.n()).collect();
    assert_eq!(sh, vec![2, 100, 64]);
    let lg = model::loss_graph(&Config::tiny(), 2, 8);
    for t in [lg.losses.total, lg.losses.mrstft, lg.losses.mel_l1] {
        assert_eq!(lg.b.g.shape(t).iter().map(|d| d.n()).collect::<Vec<_>>(), vec![1]);
    }
}

#[test]
fn istft_inverts_stft() {
    let cfg = Config::tiny();
    let frames = 12;
    let n = frames * cfg.hop;
    let mut rng = Rng::new(3);
    let x = signal(&mut rng, n);
    let (re, im, f) = dsp::stft_same_ref(&f64s(&x), cfg.n_fft, cfg.hop);
    assert_eq!(f, frames);
    let ig = model::istft_graph(&cfg, 1, frames);
    let re32: Vec<f32> = re.iter().map(|&v| v as f32).collect();
    let im32: Vec<f32> = im.iter().map(|&v| v as f32).collect();
    let inputs: Vec<Vec<f64>> = ig
        .b
        .assemble(&[], None, &[("re", &re32), ("im", &im32)])
        .iter()
        .map(|v| f64s(v))
        .collect();
    let y = &ig.b.g.eval(&inputs, &[ig.audio])[0];
    let err = y.iter().zip(&x).map(|(a, &b)| (a - b as f64).abs()).fold(0.0, f64::max);
    eprintln!("istft(stft(x)) max err {err:.2e}");
    assert!(err < 1e-5, "{err}");
}

#[test]
fn mel_matches_reference() {
    let cfg = Config::tiny();
    let frames = 10;
    let mut rng = Rng::new(5);
    let x = signal(&mut rng, frames * cfg.hop);
    let xp = dsp::reflect_pad(&x, cfg.mel_pad());
    let fe = model::frontend_graph(&cfg, 1, frames);
    let inputs: Vec<Vec<f64>> = fe.b.assemble(&[], None, &[("audio_pad", &xp)]).iter().map(|v| f64s(v)).collect();
    let got = &fe.b.g.eval(&inputs, &[fe.mel])[0];
    let want = dsp::log_mel_ref(&f64s(&xp), cfg.sr, cfg.n_fft, cfg.hop, cfg.n_mels);
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("mel vs f64 ref max err {err:.2e}");
    assert!(err < 1e-6, "{err}");
}

#[test]
fn gradient_matches_finite_differences() {
    let cfg = Config::tiny();
    let (batch, frames) = (2, 8);
    let n = frames * cfg.hop;
    let mut lg = model::loss_graph(&cfg, batch, frames);
    let mut rng = Rng::new(11);
    let mut params = model::init_params(&cfg, 7);
    // non-trivial biases / norms so every path carries gradient
    for p in params.iter_mut() {
        for v in p.iter_mut() {
            *v += (0.1 * rng.normal()) as f32;
        }
    }
    let mut xp = Vec::new();
    let mut x = Vec::new();
    for _ in 0..batch {
        let s = signal(&mut rng, n);
        xp.extend(dsp::reflect_pad(&s, cfg.mel_pad()));
        x.extend(s);
    }
    let base: Vec<Vec<f64>> = lg
        .b
        .assemble(&params, None, &[("audio_pad", &xp), ("audio", &x)])
        .iter()
        .map(|v| f64s(v))
        .collect();
    let total = lg.losses.total;
    let grads = lg.b.g.backward(total, &lg.params.clone()).unwrap();
    let mut outs = vec![total];
    outs.extend(&grads);
    let res = lg.b.g.eval(&base, &outs);
    eprintln!("tiny loss {:.6}", res[0][0]);
    let specs = cfg.params();
    let mut worst: f64 = 0.0;
    let mut checked = 0;
    for (pi, spec) in specs.iter().enumerate() {
        let k = lg.b.input_index(lg.params[pi]);
        for _ in 0..2 {
            let j = rng.below(spec.numel());
            let h = 1e-6;
            let mut plus = base.clone();
            plus[k][j] += h;
            let mut minus = base.clone();
            minus[k][j] -= h;
            let lp = lg.b.g.eval(&plus, &[total])[0][0];
            let lm = lg.b.g.eval(&minus, &[total])[0][0];
            let fd = (lp - lm) / (2.0 * h);
            let an = res[1 + pi][j];
            let rel = (fd - an).abs() / (fd.abs().max(an.abs()).max(1e-4));
            if rel > 1e-3 {
                eprintln!("  {}[{j}]: analytic {an:.6e} fd {fd:.6e} rel {rel:.1e}", spec.name);
            }
            worst = worst.max(rel);
            checked += 1;
        }
    }
    eprintln!("grad check: {checked} coords, worst rel err {worst:.2e}");
    assert!(worst < 1e-3, "{worst}");
}

/// §29.3: the APNet phase losses are 0 at the true phase, invariant to 2π
/// wraps, and a constant offset c costs only the IP term (1 − cos c).
#[test]
fn phase_losses_anti_wrapping() {
    phase_case(false);
    phase_case(true);
}

fn phase_case(weighted: bool) {
    let cfg = Config::tiny();
    let frames = 8;
    let mut rng = Rng::new(9);
    let x = signal(&mut rng, frames * cfg.hop);
    let xp = dsp::reflect_pad(&x, cfg.mel_pad());
    let mut b = model::Built::new(&cfg);
    let (c, _) = model::consts(&mut b, &[cfg.n_fft], frames);
    let ap = b.input(&[1, xp.len()], model::Role::Data("audio_pad"));
    let ph = b.input(&[1, cfg.bins(), frames], model::Role::Data("ph"));
    let (re, im) = model::stft_ri(&mut b.g, ap, cfg.n_fft, cfg.hop, c.bases(cfg.n_fft));
    let l = model::phase_losses(&mut b.g, ph, re, im, weighted);
    // the training graph differentiates this w.r.t. φ̂ (shape errors show up here)
    let tot = b.g.map(&l, |v| v[0] + v[1] + v[2]).unwrap();
    let dph = b.g.backward(tot, &[ph]).unwrap()[0];
    let k = cfg.bins();
    let run = |b: &model::Built, p: &[f32]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let xp32: Vec<f32> = xp.iter().map(|&v| v as f32).collect();
        let inputs: Vec<Vec<f64>> = b.assemble(&[], None, &[("audio_pad", &xp32), ("ph", p)]).iter().map(|v| f64s(v)).collect();
        let o = b.g.eval(&inputs, &[re, im, l[0], l[1], l[2], dph]);
        let gmax = o[5].iter().fold(0.0f64, |a, v| a.max(v.abs()));
        (o[0].clone(), o[1].clone(), vec![o[2][0], o[3][0], o[4][0], gmax])
    };
    let (r, i, _) = run(&b, &vec![0.0; k * frames]);
    // true phase, [bins, T] from [T, bins]; add random 2π wraps
    let phase = |off: f64, wrap: bool, rng: &mut Rng| -> Vec<f32> {
        let mut p = vec![0f32; k * frames];
        for t in 0..frames {
            for f in 0..k {
                let w = if wrap { 2.0 * std::f64::consts::PI * ((rng.uniform() * 5.0).floor() - 2.0) } else { 0.0 };
                p[f * frames + t] = (i[t * k + f].atan2(r[t * k + f]) + off + w) as f32;
            }
        }
        p
    };
    let (_, _, l0) = run(&b, &phase(0.0, true, &mut rng));
    eprintln!("true phase (wrapped): {l0:?}");
    assert!(l0.iter().all(|&v| v.abs() < 1e-5), "{l0:?}"); // incl. max |∂L/∂φ̂| = 0 at the optimum
    let (_, _, l1) = run(&b, &phase(0.3, false, &mut rng));
    eprintln!("offset 0.3: {l1:?}");
    assert!((l1[0] - (1.0 - 0.3f64.cos())).abs() < 1e-4 && l1[1].abs() < 1e-5 && l1[2].abs() < 1e-5 && l1[3] > 0.0, "{l1:?}");
}
