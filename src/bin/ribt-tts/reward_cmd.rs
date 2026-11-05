//! `ribt-tts diffro` (the ribt paper): differentiable reward
//! optimization of the 1-step flow student, end to end in tang.
//!
//!   ribt-tts diffro --ac-run R [--ac-ckpt C] --student S.safetensors --run DIR --phon … --durs … --root ROOT
//!                   [--held H.phon.tsv] [--batch 16] [--t 384] [--temp 0.3] [--seed 0]
//!                   [--critic-steps 2000 --critic-lr 2e-4]      phase A (skipped when DIR/critic.safetensors exists)
//!                   [--steps 500 --lr 1e-5 --lam 1.0 --warmup 50 --total 2000]   phase B, resumable (DIR/last.safetensors)
//!                   [--eval-every 100] [--keep-every 250]
//!
//! Phase A trains the critic (`ribt::reward`) on real crops (the corpus mel,
//! label 1) against the frozen student's x₁ on the same crops (label 0), both
//! under inference conditioning (predicted pitch/energy, GT durations, noise
//! temperature `--temp`). Phase B maximizes the frozen critic's mean logit on
//! the student's x₁ with a trust region `λ·‖x₁ − x₁(frozen student)‖²`.
//! Students are written as `DIR/student_stepN.safetensors` (+ flow_norm.txt),
//! loadable by `synth --flow`.

use super::*;
use flow_cmd::{cond_mix, lens_of, noise, FNorm, Vel};
use ribt::flow::{self, FConfig, FDims};
use ribt::reward::{self, RConfig};

pub fn diffro(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let fc = FConfig::base();
    let rc = RConfig::by_name(&a.opt("--rcfg").unwrap_or_else(|| "base".into())).ok_or_else(|| anyhow!("rcfg"))?;
    let (fspecs, rspecs) = (fc.params(), rc.params());
    let (ac_params, nm, _) = load_run(&PathBuf::from(a.need("--ac-run")?), &tc, a.opt("--ac-ckpt"))?;
    let spath = PathBuf::from(a.need("--student")?);
    let fnm = FNorm::load(&spath)?;
    let s0 = ckpt::load_specs(&spath, &fspecs, false)?.params;
    let run = PathBuf::from(a.need("--run")?);
    std::fs::create_dir_all(&run)?;
    std::fs::write(run.join("flow_norm.txt"), format!("{} {}\n", fnm.m, fnm.s))?;
    let root = PathBuf::from(a.need("--root")?);
    let batch: usize = a.get("--batch", 16);
    let tmax: usize = a.get("--t", 384);
    let temp: f64 = a.get("--temp", 0.3);
    let seed: u64 = a.get("--seed", 0);
    let nmax: usize = 192;
    let keep = |n: usize, t: usize| n <= nmax && t <= 1024 && t >= 32;
    let d = Dims { batch, n: nmax, t: tmax, r: REF_T };
    let fd = FDims { batch, t: tmax };
    let (c, t) = (tc.n_mels, tmax);
    let durs = read_durs(&a.need("--durs")?)?;
    let data = load(&a.need("--phon")?, &durs, &root, &keep)?;
    let spk = by_speaker(&data);
    let held: Vec<Utt> = match a.opt("--held") {
        Some(h) => load(&h, &durs, &root, &keep)?.into_iter().take(4 * batch).collect(),
        None => Vec::new(),
    };
    let held_spk = by_speaker(&held);
    eprintln!("diffro: {} train utts, {} held; critic {:.2}M params; temp {temp}", data.len(), held.len(), rc.n_params() as f64 / 1e6);
    let dev = Device::new();
    let mut fe = Frontend::new(&dev, batch, tmax + 2 * CTX);
    let mut fe_ref = Frontend::new(&dev, batch, REF_T);
    let mut tts = Tts::new(&dev, &tc, &ac_params, d);
    let mut frozen = Vel::new(&dev, &fc, &s0, fd);
    let temb1: Vec<f32> = (0..batch).flat_map(|_| flow::time_feats(&fc, 0.0, 1.0)).collect();

    // a batch under inference conditioning: (real z, x0, frozen x1, mu, hc, fmask, abias)
    struct B {
        real: Vec<f32>,
        x0: Vec<f32>,
        x1: Vec<f32>,
        mu: Vec<f32>,
        hc: Vec<f32>,
        fm: Vec<f32>,
        ab: Vec<f32>,
    }
    let mut make = |rng: &mut Rng, pool: &[Utt], pspk: &HashMap<String, Vec<usize>>, frozen: &mut Vel, tts: &mut Tts| -> B {
        let us: Vec<&Utt> = (0..batch).map(|_| &pool[rng.below(pool.len())]).collect();
        let (hb, mel, refmel) = prep_crops(&dev, &tc, d, &nm, rng, true, &us, pool, pspk, &mut fe, &mut fe_ref);
        let fm = hb.get("fmask").to_vec();
        let lens = lens_of(&fm, batch, t);
        let (mu, hc, _, _) = cond_mix(&dev, tts, &hb, &refmel, &vec![true; batch]);
        let mu = fnm.norm(&mu, &fm, batch, t);
        let real = fnm.norm(&mel, &fm, batch, t);
        let ab = flow::attn_bias(&lens, t, None);
        frozen.set_cond(&dev, &mu, &hc, &fm, &ab);
        let x0 = noise(rng, &fm, batch, c, t, temp);
        let (x1, _) = frozen.sample(&dev, &x0, 1, true);
        B { real, x0, x1, mu, hc, fm, ab }
    };
    let fw_of = |fm: &[f32]| -> Vec<f32> {
        let n: f32 = fm.iter().sum::<f32>().max(1.0);
        fm.iter().map(|m| m / n).collect()
    };

    // ---- phase A: critic
    let cpath = run.join("critic.safetensors");
    let cparams = if cpath.exists() {
        eprintln!("critic: loading {}", cpath.display());
        ckpt::load_specs(&cpath, &rspecs, false)?.params
    } else {
        let steps: usize = a.get("--critic-steps", 2000);
        let lr: f64 = a.get("--critic-lr", 2e-4);
        let ct = reward::critic_train_graph(&rc, 2 * batch, t);
        let init = reward::init_params(&rc, seed);
        let inp = ribt::flow::assemble(&ct.cg.b, &rspecs, &init, None, &[]);
        let mut prog = Program::new(&dev, &ct.cg.b.g, &ct.outputs, &refs(&inp));
        let b = &ct.cg.b;
        let dm = input_map(b);
        let k_h = b.input_index(ct.hyper);
        let np = rspecs.len();
        let k_p: Vec<usize> = ct.cg.params.iter().map(|&x| b.input_index(x)).collect();
        let k_m: Vec<usize> = ct.m.iter().map(|&x| b.input_index(x)).collect();
        let k_v: Vec<usize> = ct.v.iter().map(|&x| b.input_index(x)).collect();
        // held: forward-only critic graph for accuracy
        let eg = reward::critic_graph(&rc, 2 * batch, t);
        let einp = ribt::flow::assemble(&eg.b, &rspecs, &init, None, &[]);
        let mut eprog = Program::new(&dev, &eg.b.g, &[eg.logits, eg.loss], &refs(&einp));
        let em = input_map(&eg.b);
        let ek_p: Vec<usize> = eg.params.iter().map(|&x| eg.b.input_index(x)).collect();
        let y: Vec<f32> = (0..2 * batch).map(|i| if i < batch { 1.0 } else { 0.0 }).collect();
        let mut log = std::fs::File::create(run.join("critic.tsv"))?;
        writeln!(log, "step\tloss\tgrad_norm\theld_loss\theld_acc\theld_gap\tms")?;
        let mut rng = Rng::new(31 + seed);
        let t0 = Instant::now();
        let held_eval = |mk: &mut dyn FnMut(&mut Rng, &[Utt], &HashMap<String, Vec<usize>>, &mut Vel, &mut Tts) -> B, prog: &Program, eprog: &mut Program, frozen: &mut Vel, tts: &mut Tts| -> (f64, f64, f64) {
            if held.len() < batch {
                return (f64::NAN, f64::NAN, f64::NAN);
            }
            for (&ek, &k) in ek_p.iter().zip(&k_p) {
                eprog.write_input(&dev, ek, &prog.read_input(&dev, k).unwrap());
            }
            let mut hr = Rng::new(555);
            let (mut l, mut acc, mut gap, mut n) = (0.0, 0.0, 0.0, 0.0);
            for _ in 0..4 {
                let bt = mk(&mut hr, &held, &held_spk, frozen, tts);
                let z = [bt.real.clone(), bt.x1.clone()].concat();
                let fm = [bt.fm.clone(), bt.fm.clone()].concat();
                for (k, v) in [("z", &z), ("fmask", &fm), ("fw", &fw_of(&fm)), ("y", &y)] {
                    eprog.write_input(&dev, em[k], v);
                }
                eprog.run(&dev);
                let lg = eprog.read(&dev, 0);
                l += eprog.read(&dev, 1)[0] as f64;
                // per-utterance mean logit: real should exceed fake
                let mean = |i: usize| -> f64 {
                    let (s, m): (f64, f64) = (0..t).fold((0.0, 0.0), |(s, m), j| (s + (lg[i * t + j] * fm[i * t + j]) as f64, m + fm[i * t + j] as f64));
                    s / m.max(1.0)
                };
                for i in 0..batch {
                    let (r, f) = (mean(i), mean(batch + i));
                    acc += ((r > 0.0) as u8 as f64 + (f < 0.0) as u8 as f64) / 2.0;
                    gap += r - f;
                    n += 1.0;
                }
            }
            (l / 4.0, acc / n, gap / n)
        };
        let h = held_eval(&mut make, &prog, &mut eprog, &mut frozen, &mut tts);
        eprintln!("critic step 0: held loss {:.4} acc {:.3} gap {:.3}", h.0, h.1, h.2);
        for step in 1..=steps {
            let ts = Instant::now();
            let bt = make(&mut rng, &data, &spk, &mut frozen, &mut tts);
            let z = [bt.real, bt.x1].concat();
            let fm = [bt.fm.clone(), bt.fm].concat();
            for (k, v) in [("z", &z), ("fmask", &fm), ("fw", &fw_of(&fm)), ("y", &y)] {
                prog.write_input(&dev, dm[k], v);
            }
            let lr_s = lr_at(step - 1, steps, lr, 100);
            let ti = step as i32;
            prog.write_input(&dev, k_h, &[lr_s as f32, (1.0 - 0.5f64.powi(ti)) as f32, (1.0 - 0.9f64.powi(ti)) as f32, 0.0, 1.0, 0.0]);
            prog.run(&dev);
            let (gn, l) = (prog.read(&dev, 0)[0], prog.read(&dev, 1)[0]);
            if !(gn.is_finite() && l.is_finite()) {
                return Err(anyhow!("critic step {step}: non-finite"));
            }
            for i in 0..np {
                prog.swap_input_output(k_p[i], 3 + i);
                prog.swap_input_output(k_m[i], 3 + np + i);
                prog.swap_input_output(k_v[i], 3 + 2 * np + i);
            }
            let ms = ts.elapsed().as_secs_f64() * 1e3;
            let h = if step % 250 == 0 || step == steps { held_eval(&mut make, &prog, &mut eprog, &mut frozen, &mut tts) } else { (f64::NAN, f64::NAN, f64::NAN) };
            writeln!(log, "{step}\t{l:.5}\t{gn:.4}\t{:.5}\t{:.4}\t{:.4}\t{ms:.0}", h.0, h.1, h.2)?;
            if step % 100 == 0 {
                eprintln!("critic step {step:>5} loss {l:.4} |g| {gn:.3} {ms:.0} ms; held loss {:.4} acc {:.3} gap {:.3} ({:.0} s)", h.0, h.1, h.2, t0.elapsed().as_secs_f64());
                log.flush()?;
            }
        }
        let p: Vec<Vec<f32>> = k_p.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect();
        ckpt::save_specs(&cpath, &rspecs, &Checkpoint { params: p.clone(), adam: None, step: steps })?;
        p
    };

    // ---- phase B: DiffRO on the student
    let steps: usize = a.get("--steps", 0);
    if steps == 0 {
        return Ok(());
    }
    let lr: f64 = a.get("--lr", 1e-5);
    let lam: f64 = a.get("--lam", 1.0);
    let warmup: usize = a.get("--warmup", 50);
    let total: usize = a.get("--total", 2000);
    let eval_every: usize = a.get("--eval-every", 100);
    let keep_every: usize = a.get("--keep-every", 250);
    let lpath = run.join("last.safetensors");
    let ck = if lpath.exists() { ckpt::load_specs(&lpath, &fspecs, false)? } else { Checkpoint { params: s0.clone(), adam: None, step: 0 } };
    let dt = reward::diffro_graph(&fc, &rc, fd, lam);
    let zeros: Vec<Vec<f32>> = fspecs.iter().map(|s| vec![0.0; s.numel()]).collect();
    let (am, av) = match &ck.adam {
        Some((m, v)) => (m.clone(), v.clone()),
        None => (zeros.clone(), zeros.clone()),
    };
    let inp = dt.fg.b.assemble_gd(&ck.params, Some((&am, &av)), &cparams, None, &[]);
    let tcomp = Instant::now();
    let mut prog = Program::new(&dev, &dt.fg.b.g, &dt.outputs, &refs(&inp));
    drop(inp);
    eprintln!("diffro program: {} kernels, compile {:.1} s; lam {lam} lr {lr}", prog.kernels(), tcomp.elapsed().as_secs_f64());
    let b = &dt.fg.b;
    let dm = input_map(b);
    let k_h = b.input_index(dt.hyper);
    let np = fspecs.len();
    let k_p: Vec<usize> = dt.fg.params.iter().map(|&x| b.input_index(x)).collect();
    let k_m: Vec<usize> = dt.m.iter().map(|&x| b.input_index(x)).collect();
    let k_v: Vec<usize> = dt.v.iter().map(|&x| b.input_index(x)).collect();
    let o = 4;
    // held reward of the current student: the critic forward on its x1
    let cg = reward::critic_graph(&rc, batch, t);
    let cinp = ribt::flow::assemble(&cg.b, &rspecs, &cparams, None, &[]);
    let mut cprog = Program::new(&dev, &cg.b.g, &[cg.logits], &refs(&cinp));
    let cm = input_map(&cg.b);
    let mut cur = Vel::new(&dev, &fc, &ck.params, fd);
    let new_log = !run.join("diffro.tsv").exists();
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(run.join("diffro.tsv"))?;
    if new_log {
        writeln!(log, "step\tloss\treward\ttrust\tgrad_norm\tlr\theld_reward\theld_reward_frozen\theld_real\theld_trust\tms")?;
    }
    let held_eval = |mk: &mut dyn FnMut(&mut Rng, &[Utt], &HashMap<String, Vec<usize>>, &mut Vel, &mut Tts) -> B, prog: &Program, cur: &mut Vel, frozen: &mut Vel, tts: &mut Tts, cprog: &mut Program| -> [f64; 4] {
        if held.len() < batch {
            return [f64::NAN; 4];
        }
        let p: Vec<Vec<f32>> = k_p.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect();
        cur.set_params(&dev, &p);
        let mut hr = Rng::new(555);
        let mut acc = [0.0f64; 4];
        for _ in 0..4 {
            let bt = mk(&mut hr, &held, &held_spk, frozen, tts);
            cur.set_cond(&dev, &bt.mu, &bt.hc, &bt.fm, &bt.ab);
            let (x1, _) = cur.sample(&dev, &bt.x0, 1, true);
            let fw = fw_of(&bt.fm);
            let mut score = |z: &[f32]| -> f64 {
                cprog.write_input(&dev, cm["z"], z);
                cprog.write_input(&dev, cm["fmask"], &bt.fm);
                cprog.run(&dev);
                cprog.read(&dev, 0).iter().zip(&fw).map(|(l, w)| (l * w) as f64).sum()
            };
            acc[0] += score(&x1);
            acc[1] += score(&bt.x1);
            acc[2] += score(&bt.real);
            let mut tr = 0.0;
            for bi in 0..batch {
                for ci in 0..c {
                    for ti in 0..t {
                        let i = (bi * c + ci) * t + ti;
                        tr += ((x1[i] - bt.x1[i]) as f64).powi(2) * fw[bi * t + ti] as f64 / c as f64;
                    }
                }
            }
            acc[3] += tr;
        }
        acc.map(|v| v / 4.0)
    };
    let mut step = ck.step;
    let start = step;
    let mut rng = Rng::new(7000 + step as u64 + seed);
    let save = |prog: &Program, step: usize, path: &Path, adam: bool| -> Result<()> {
        let rd = |ks: &[usize]| -> Vec<Vec<f32>> { ks.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect() };
        ckpt::save_specs(path, &fspecs, &Checkpoint { params: rd(&k_p), adam: if adam { Some((rd(&k_m), rd(&k_v))) } else { None }, step })
    };
    if step == 0 {
        let h = held_eval(&mut make, &prog, &mut cur, &mut frozen, &mut tts, &mut cprog);
        writeln!(log, "0\t\t\t\t\t\t{:.4}\t{:.4}\t{:.4}\t{:.6}\t", h[0], h[1], h[2], h[3])?;
        eprintln!("diffro step 0: held reward {:.4} (frozen {:.4}, real {:.4})", h[0], h[1], h[2]);
    }
    while step < start + steps {
        let ts = Instant::now();
        let bt = make(&mut rng, &data, &spk, &mut frozen, &mut tts);
        let fw = fw_of(&bt.fm);
        for (k, v) in [("xt", &bt.x0), ("temb", &temb1), ("mu", &bt.mu), ("hc", &bt.hc), ("fmask", &bt.fm), ("abias", &bt.ab), ("x1ref", &bt.x1), ("fw", &fw)] {
            prog.write_input(&dev, dm[k], v);
        }
        let lr_s = lr_at(step, total, lr, warmup);
        let ti = (step + 1) as i32;
        prog.write_input(&dev, k_h, &[lr_s as f32, (1.0 - 0.9f64.powi(ti)) as f32, (1.0 - 0.99f64.powi(ti)) as f32, 0.0, 1.0, 0.0]);
        prog.run(&dev);
        let r: Vec<f32> = (0..4).map(|k| prog.read(&dev, k)[0]).collect();
        if !r.iter().all(|v| v.is_finite()) {
            return Err(anyhow!("diffro step {}: non-finite {r:?}", step + 1));
        }
        for i in 0..np {
            prog.swap_input_output(k_p[i], o + i);
            prog.swap_input_output(k_m[i], o + np + i);
            prog.swap_input_output(k_v[i], o + 2 * np + i);
        }
        step += 1;
        let ms = ts.elapsed().as_secs_f64() * 1e3;
        let h = if step % eval_every == 0 { held_eval(&mut make, &prog, &mut cur, &mut frozen, &mut tts, &mut cprog) } else { [f64::NAN; 4] };
        writeln!(log, "{step}\t{:.5}\t{:.4}\t{:.6}\t{:.4}\t{lr_s:.2e}\t{:.4}\t{:.4}\t{:.4}\t{:.6}\t{ms:.0}", r[1], r[2], r[3], r[0], h[0], h[1], h[2], h[3])?;
        if step % 25 == 0 {
            eprintln!("diffro step {step:>5} loss {:.4} reward {:.4} trust {:.5} |g| {:.3} {ms:.0} ms; held reward {:.4} (frozen {:.4}, real {:.4}) trust {:.5}", r[1], r[2], r[3], r[0], h[0], h[1], h[2], h[3]);
            log.flush()?;
        }
        if keep_every > 0 && step % keep_every == 0 {
            save(&prog, step, &run.join(format!("student_step{step:05}.safetensors")), false)?;
        }
    }
    save(&prog, step, &lpath, true)?;
    save(&prog, step, &run.join(format!("student_step{step:05}.safetensors")), false)?;
    Ok(())
}

/// `ribt-tts diffro-check`: one DiffRO step (check cfgs) on the GPU vs the f64
/// interpreter, and the derived student gradient vs central differences of
/// the interpreter's loss.
pub fn diffro_check(a: &Args) -> Result<()> {
    let fc = FConfig::by_name("check").unwrap();
    let rc = RConfig::check();
    let (fspecs, rspecs) = (fc.params(), rc.params());
    let d = FDims { batch: a.get("--batch", 2), t: a.get("--t", 32) };
    let lam = 0.7;
    let mut rng = Rng::new(5);
    let jit = |rng: &mut Rng, v: Vec<Vec<f32>>, s: f64| -> Vec<Vec<f32>> { v.into_iter().map(|p| p.iter().map(|x| x + (s * rng.normal()) as f32).collect()).collect() };
    let sp = jit(&mut rng, flow::init_params(&fc, 1), 0.05);
    let cp = jit(&mut rng, reward::init_params(&rc, 2), 0.05);
    let (b, t, c) = (d.batch, d.t, fc.n_mels);
    let lens: Vec<usize> = (0..b).map(|i| t - 5 * i).collect();
    let mut fm = vec![0.0f32; b * t];
    for (bi, &n) in lens.iter().enumerate() {
        fm[bi * t..bi * t + n].iter_mut().for_each(|v| *v = 1.0);
    }
    let nfr: f32 = fm.iter().sum();
    let temb: Vec<f32> = (0..b).flat_map(|_| flow::time_feats(&fc, 0.0, 1.0)).collect();
    let data: Vec<(&str, Vec<f32>)> = vec![
        ("xt", noise(&mut rng, &fm, b, c, t, 0.3)),
        ("mu", noise(&mut rng, &fm, b, c, t, 1.0)),
        ("hc", noise(&mut rng, &fm, b, fc.dc, t, 1.0)),
        ("x1ref", noise(&mut rng, &fm, b, c, t, 0.5)),
        ("temb", temb),
        ("abias", flow::attn_bias(&lens, t, None)),
        ("fw", fm.iter().map(|m| m / nfr).collect()),
        ("fmask", fm),
    ];
    let dr: Vec<(&str, &[f32])> = data.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let dt = reward::diffro_graph(&fc, &rc, d, lam);
    let zeros: Vec<Vec<f32>> = fspecs.iter().map(|s| vec![0.0; s.numel()]).collect();
    let mut inp = dt.fg.b.assemble_gd(&sp, Some((&zeros, &zeros)), &cp, None, &dr);
    // lr 0, bias corrections 1, clip huge: m' = 0.1·g
    inp[dt.fg.b.input_index(dt.hyper)] = vec![0.0, 1.0, 1.0, 0.0, 1e9, 0.0];
    let _ = &rspecs;
    let f64s = |inp: &[Vec<f32>]| -> Vec<Vec<f64>> { inp.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect() };
    let want = dt.fg.b.g.eval(&f64s(&inp), &dt.outputs);
    let dev = Device::new();
    let mut prog = Program::new(&dev, &dt.fg.b.g, &dt.outputs, &refs(&inp));
    prog.run(&dev);
    let np = fspecs.len();
    let rel = |got: &[f32], want: &[f64]| -> f64 {
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
        got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((*g as f64 - w).abs())) / scale
    };
    let mut gerr = 0.0f64;
    for i in 0..np {
        gerr = gerr.max(rel(&prog.read(&dev, 4 + np + i), &want[4 + np + i]));
    }
    eprintln!("{} vs interpreter: loss {} vs {} (reward {}, trust {}); grads worst rel {gerr:.2e}", ribt::runtime::BACKEND, prog.read(&dev, 1)[0], want[1][0], want[2][0], want[3][0]);
    // central differences on the interpreter
    let lossg = dt.fg.b.g.clone();
    let mut fd_err = 0.0f64;
    for name in ["out.w", "b.0.pw1.w", "in.x.w", "b.2.q"] {
        let pi = fspecs.iter().position(|s| s.name == name).ok_or_else(|| anyhow!("{name}"))?;
        let k = dt.fg.b.input_index(dt.fg.params[pi]);
        let gvec = &want[4 + np + pi];
        for j in [0usize, 7, 31] {
            let mut x = f64s(&inp);
            let h = 1e-4;
            x[k][j] += h;
            let lp = lossg.eval(&x, &[dt.outputs[1]])[0][0];
            x[k][j] -= 2.0 * h;
            let lm = lossg.eval(&x, &[dt.outputs[1]])[0][0];
            let fdv = (lp - lm) / (2.0 * h);
            let an = gvec[j] / 0.1;
            let e = (fdv - an).abs() / (fdv.abs().max(an.abs()).max(1e-6));
            fd_err = fd_err.max(e);
            eprintln!("  d loss / d {name}[{j}]: derived {an:.6e}  central diff {fdv:.6e}  rel {e:.1e}");
        }
    }
    if gerr > 1e-3 || fd_err > 1e-3 {
        return Err(anyhow!("diffro check failed (gpu {gerr:.2e}, fd {fd_err:.2e})"));
    }
    eprintln!("diffro-check ok (gpu {gerr:.2e}, fd {fd_err:.2e})");
    Ok(())
}
