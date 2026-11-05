//! `hear-heads`: rung-1 heads for crates/hear (docs/hear.md §10), trained on frozen-encoder frames
//! dumped by `hear dump`. Run GPU work under `~/.cache/ribt-gpu-lock.sh`.
//!
//!   hear-heads train --data DIR[,DIR..] --run RUN [--steps N] [--total N] [--batch 64] [--t 512]
//!                    [--hidden 256] [--lr 2e-3] [--w-turn 1]        (touch RUN/STOP to save + exit)
//!
//! DIR holds `feats.f16` + `index.tsv` (from `hear dump`) and `examples.tsv` (scripts/heads_data.py).
//! One shared hidden layer over the 512-d frames, two outputs:
//! - phoneme CTC over lilt/Kokoro ids 0..177, id 0 (pad, never a target) as the blank, at `--up`
//!   (default 3) sub-frames per 80 ms frame: w2 is [up·178, H], rows u·178.. are sub-frame u. At 80 ms
//!   one label per frame is too slow for phonemes (~12–15/s plus a blank between repeats);
//!   most utterances made CTC infeasible in the first run (h256, up = 1);
//! - turn hazard h_t = σ(z_t), discrete-time survival loss L = −log h_{T*} − Σ_{t<T*} log(1 − h_t).
//! Both losses are host-side (the IR has no scan; same cut as ribt-align): the host computes
//! G = ∂L/∂z, the train program's objective is the surrogate Σ z ⊙ G, and its derived backward is
//! ∂L/∂θ. Saves RUN/last.safetensors with hear's names (heads.w1 [H,512], heads.b1, heads.w2
//! [178,H], heads.b2, heads.w3 [1,H], heads.b3), which `hear` merges into its graph.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use ribt::ckpt::{self, Checkpoint};
use ribt::ir;
use ribt::model::{self, Built, Config, ParamSpec, Rng, Role};
use ribt::runtime::{Device, Program};
use tang::Scalar;
use tang_compile::tensor::TensorId;

const D: usize = 512;
const K: usize = 178;

struct Args(Vec<String>);
impl Args {
    fn opt(&self, k: &str) -> Option<String> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1).cloned())
    }
    fn get<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        self.opt(k).and_then(|v| v.parse().ok()).unwrap_or(d)
    }
}

fn specs(h: usize, up: usize) -> Vec<ParamSpec> {
    vec![
        ParamSpec::normal("heads.w1", &[h, D], 1.0 / (D as f64).sqrt()),
        ParamSpec::zeros("heads.b1", &[h]),
        ParamSpec::normal("heads.w2", &[up * K, h], 1.0 / (h as f64).sqrt()),
        ParamSpec::zeros("heads.b2", &[up * K]),
        ParamSpec::normal("heads.w3", &[1, h], 0.1 / (h as f64).sqrt()),
        ParamSpec::constant("heads.b3", &[1], -4.0),
    ]
}

pub struct Ex {
    off: usize,
    t: usize,
    phon: Vec<usize>,
    tstar: usize,
}

fn load(dir: &str) -> Result<(Vec<u16>, Vec<Ex>)> {
    let d = PathBuf::from(dir);
    let raw = std::fs::read(d.join("feats.f16"))?;
    let feats: Vec<u16> = raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    let mut idx = std::collections::HashMap::new();
    for l in std::fs::read_to_string(d.join("index.tsv"))?.lines() {
        let f: Vec<&str> = l.split('\t').collect();
        idx.insert(f[0].to_string(), (f[1].parse::<usize>()?, f[2].parse::<usize>()?));
    }
    let mut ex = vec![];
    for l in std::fs::read_to_string(d.join("examples.tsv"))?.lines() {
        let f: Vec<&str> = l.split('\t').collect();
        let Some(&(off, t)) = idx.get(f[0]) else { continue };
        let phon = f[2].split_whitespace().map(|v| v.parse().unwrap()).collect();
        ex.push(Ex { off, t, phon, tstar: f[3].parse()? });
    }
    Ok((feats, ex))
}

fn f16(b: u16) -> f32 {
    let s = if b & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 10) & 0x1f) as i32;
    let m = (b & 0x3ff) as f32;
    s * match e {
        0 => m * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// CTC (blank 0) over log-probs `lp[t*K + k]`: (−log p(y|x), ∂L/∂z for the logits z behind lp).
/// None when the target cannot fit in `t` frames.
pub fn ctc(lp: &[f64], t: usize, y: &[usize]) -> Option<(f64, Vec<f64>)> {
    let s = 2 * y.len() + 1;
    let lab = |i: usize| if i % 2 == 0 { 0 } else { y[i / 2] };
    let need = y.len() + y.windows(2).filter(|w| w[0] == w[1]).count();
    if need > t {
        return None;
    }
    let ni = f64::NEG_INFINITY;
    let mut a = vec![ni; t * s];
    let mut b = vec![ni; t * s];
    a[0] = lp[0];
    if s > 1 {
        a[1] = lp[lab(1)];
    }
    for ti in 1..t {
        for i in 0..s {
            let mut v = a[(ti - 1) * s + i];
            if i >= 1 {
                v = lse(v, a[(ti - 1) * s + i - 1]);
            }
            if i >= 2 && lab(i) != 0 && lab(i) != lab(i - 2) {
                v = lse(v, a[(ti - 1) * s + i - 2]);
            }
            a[ti * s + i] = v + lp[ti * K + lab(i)];
        }
    }
    b[(t - 1) * s + s - 1] = lp[(t - 1) * K + lab(s - 1)];
    if s > 1 {
        b[(t - 1) * s + s - 2] = lp[(t - 1) * K + lab(s - 2)];
    }
    for ti in (0..t - 1).rev() {
        for i in 0..s {
            let mut v = b[(ti + 1) * s + i];
            if i + 1 < s {
                v = lse(v, b[(ti + 1) * s + i + 1]);
            }
            if i + 2 < s && lab(i) != 0 && lab(i) != lab(i + 2) {
                v = lse(v, b[(ti + 1) * s + i + 2]);
            }
            b[ti * s + i] = v + lp[ti * K + lab(i)];
        }
    }
    let ll = lse(a[(t - 1) * s + s - 1], if s > 1 { a[(t - 1) * s + s - 2] } else { ni });
    if !ll.is_finite() {
        return None;
    }
    let mut g = vec![0.0; t * K];
    for ti in 0..t {
        // occupancy γ_t(k) = Σ_{i: lab(i)=k} α β / (p · y_t(k)); grad = softmax − γ
        let mut occ = vec![ni; K];
        for i in 0..s {
            let k = lab(i);
            occ[k] = lse(occ[k], a[ti * s + i] + b[ti * s + i]);
        }
        for k in 0..K {
            let post = if occ[k] == ni { 0.0 } else { (occ[k] - lp[ti * K + k] - ll).exp() };
            g[ti * K + k] = lp[ti * K + k].exp() - post;
        }
    }
    Some((-ll, g))
}

/// Survival loss on hazard logits z (frames 0..=tstar used): (loss, ∂L/∂z).
pub fn survival(z: &[f64], tstar: usize) -> (f64, Vec<f64>) {
    let mut g = vec![0.0; z.len()];
    let mut l = 0.0;
    for (t, &v) in z.iter().enumerate().take(tstar + 1) {
        let h = 1.0 / (1.0 + (-v).exp());
        let y = (t == tstar) as u8 as f64;
        // −[y log h + (1−y) log(1−h)], stable
        l += v.max(0.0) - v * y + (-v.abs()).exp().ln_1p();
        g[t] = h - y;
    }
    (l, g)
}

struct Graph {
    b: Built,
    params: Vec<TensorId>,
    x: TensorId,
    zp: TensorId,
    zt: TensorId,
    gp: TensorId,
    gt: TensorId,
    m: Vec<TensorId>,
    v: Vec<TensorId>,
    hyper: TensorId,
    outputs: Vec<TensorId>,
}

fn graph(sp: &[ParamSpec], h: usize, batch: usize, t: usize) -> Graph {
    let mut b = Built::new_pub(&Config::tiny());
    let params: Vec<TensorId> = sp.iter().enumerate().map(|(i, s)| b.input(&s.shape, Role::Param(i))).collect();
    let x = b.input(&[batch, D, t], Role::Data("x"));
    let gp = b.input(&[batch, sp[2].shape[0], t], Role::Data("gp"));
    let gt = b.input(&[batch, 1, t], Role::Data("gt"));
    let m: Vec<TensorId> = sp.iter().enumerate().map(|(i, s)| b.input(&s.shape, Role::AdamM(i))).collect();
    let v: Vec<TensorId> = sp.iter().enumerate().map(|(i, s)| b.input(&s.shape, Role::AdamV(i))).collect();
    let hyper = b.input(&[6], Role::Hyper);
    let g = &mut b.g;
    let _ = h;
    let hd = ir::linear(g, x, params[0]);
    let hd = ir::channel_map(g, hd, &[params[1]], |x, p| {
        let y = x + p[0];
        tang_expr::ExprId::select(y, y, tang_expr::ExprId::ZERO)
    });
    let zp = ir::linear(g, hd, params[2]);
    let zp = ir::add_channel(g, zp, params[3]);
    let zt = ir::linear(g, hd, params[4]);
    let zt = ir::add_channel(g, zt, params[5]);
    let s1 = ir::sum_all2(g, zp, Some(gp), |v| v[0] * v[1]);
    let s2 = ir::sum_all2(g, zt, Some(gt), |v| v[0] * v[1]);
    let sur = g.map(&[s1, s2], |v| v[0] + v[1]).unwrap();
    let grads = g.backward(sur, &params).unwrap();
    let ad = model::adamw_betas(g, sp, &params, &grads, &m, &v, hyper, 0.9, 0.999);
    let mut outputs = vec![ad.grad_norm, zp, zt];
    outputs.extend(&ad.new_p);
    outputs.extend(&ad.new_m);
    outputs.extend(&ad.new_v);
    outputs.extend(ad.partials);
    Graph { b, params, x, zp, zt, gp, gt, m, v, hyper, outputs }
}

fn main() -> Result<()> {
    let a = Args(std::env::args().skip(1).collect());
    if a.0.first().map(String::as_str) != Some("train") {
        return Err(anyhow!("usage: hear-heads train --data DIR --run RUN (see src/bin/hear-heads.rs)"));
    }
    let run = PathBuf::from(a.opt("--run").ok_or_else(|| anyhow!("--run"))?);
    std::fs::create_dir_all(&run)?;
    let hdim: usize = a.get("--hidden", 256);
    let batch: usize = a.get("--batch", 64);
    let tmax: usize = a.get("--t", 512);
    let steps: usize = a.get("--steps", 1000);
    let total: usize = a.get("--total", 6000);
    let peak: f64 = a.get("--lr", 2e-3);
    let w_turn: f64 = a.get("--w-turn", 1.0);
    let up: usize = a.get("--up", 3);
    let sp = specs(hdim, up);
    let ku = up * K;
    let np = sp.len();
    let t0 = Instant::now();
    let mut feats = vec![];
    let mut data = vec![];
    for d in a.opt("--data").ok_or_else(|| anyhow!("--data"))?.split(',') {
        let (f, mut e) = load(d)?;
        let base = feats.len() / D;
        for x in &mut e {
            x.off += base;
        }
        feats.extend(f);
        data.extend(e);
    }
    data.retain(|e| e.t <= tmax && e.tstar < e.t);
    eprintln!("{} examples ≤ {tmax} frames, {:.1} h, loaded in {:.0} s", data.len(), data.iter().map(|e| e.t).sum::<usize>() as f64 * 0.08 / 3600.0, t0.elapsed().as_secs_f64());

    let ck_path = run.join("last.safetensors");
    let ck = if ck_path.exists() {
        ckpt::load_specs(&ck_path, &sp, false)?
    } else {
        let mut rng = Rng::new(0);
        Checkpoint { params: sp.iter().map(|s| s.init_data(&mut rng)).collect(), adam: None, step: 0 }
    };
    let dev = Device::new();
    eprintln!("backend {} ({}), start step {}", ribt::runtime::BACKEND, dev.name(), ck.step);
    let gr = graph(&sp, hdim, batch, tmax);
    let adam = ck.adam.as_ref().map(|(m, v)| (m.as_slice(), v.as_slice()));
    let inputs: Vec<Vec<f32>> = gr
        .b
        .inputs
        .iter()
        .map(|i| match &i.role {
            Role::Param(k) => ck.params[*k].clone(),
            Role::AdamM(k) => adam.map_or_else(|| vec![0.0; sp[*k].numel()], |a| a.0[*k].clone()),
            Role::AdamV(k) => adam.map_or_else(|| vec![0.0; sp[*k].numel()], |a| a.1[*k].clone()),
            Role::Hyper => vec![0.0; 6],
            _ => vec![0.0; gr.b.g.shape(i.id).iter().map(|d| d.n()).product()],
        })
        .collect();
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let mut prog = Program::new(&dev, &gr.b.g, &gr.outputs, &refs);
    drop(inputs);
    let ix = |t: TensorId| gr.b.input_index(t);
    let (kx, kgp, kgt, kh) = (ix(gr.x), ix(gr.gp), ix(gr.gt), ix(gr.hyper));
    let kp: Vec<usize> = gr.params.iter().map(|&t| ix(t)).collect();
    let km: Vec<usize> = gr.m.iter().map(|&t| ix(t)).collect();
    let kv: Vec<usize> = gr.v.iter().map(|&t| ix(t)).collect();
    let (op, om, ov) = (3, 3 + np, 3 + 2 * np);
    let _ = (gr.zp, gr.zt);
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(run.join("log.tsv"))?;
    let save = |prog: &Program, step: usize| -> Result<()> {
        let rd = |ks: &[usize]| -> Vec<Vec<f32>> { ks.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect() };
        ckpt::save_specs(&ck_path, &sp, &Checkpoint { params: rd(&kp), adam: Some((rd(&km), rd(&kv))), step })
    };
    // Each step runs the program twice: once with G = 0 and lr = 0 and no input/output swap
    // (a pure forward: z for this batch; params and moments are untouched), then with the host G.
    let mut rng = Rng::new(1000 + ck.step as u64);
    let mut step = ck.step;
    let start = step;
    let (mut accp, mut acct, mut accn, mut skipped) = (0.0, 0.0, 0usize, 0usize);
    let stop = run.join("STOP");
    let zeros_p = vec![0.0f32; batch * ku * tmax];
    let zeros_t = vec![0.0f32; batch * tmax];
    while step < start + steps {
        let ts = Instant::now();
        let bx: Vec<&Ex> = (0..batch).map(|_| &data[rng.below(data.len())]).collect();
        let mut x = vec![0.0f32; batch * D * tmax];
        for (bi, e) in bx.iter().enumerate() {
            for t in 0..e.t {
                let row = &feats[(e.off + t) * D..(e.off + t + 1) * D];
                for d in 0..D {
                    x[(bi * D + d) * tmax + t] = f16(row[d]);
                }
            }
        }
        // forward pass (G = 0, lr = 0, no swap): z for this batch
        prog.write_input(&dev, kx, &x);
        prog.write_input(&dev, kgp, &zeros_p);
        prog.write_input(&dev, kgt, &zeros_t);
        prog.write_input(&dev, kh, &[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
        prog.run(&dev);
        let zp = prog.read(&dev, 1);
        let zt = prog.read(&dev, 2);
        let mut gp = vec![0.0f32; batch * ku * tmax];
        let mut gt = vec![0.0f32; batch * tmax];
        let (mut lp_sum, mut lt_sum, mut n_ok) = (0.0, 0.0, 0usize);
        // host CTC per example, in parallel (it is the step's bottleneck at up = 3)
        let at = |bi: usize, ti: usize, k: usize| (bi * ku + (ti % up) * K + k) * tmax + ti / up;
        let res: Vec<Option<(f64, Vec<f64>)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = bx
                .iter()
                .enumerate()
                .map(|(bi, e)| {
                    let zp = &zp;
                    sc.spawn(move || {
                        let t = e.t * up;
                        let mut lp = vec![0.0f64; t * K];
                        for ti in 0..t {
                            let col: Vec<f64> = (0..K).map(|k| zp[at(bi, ti, k)] as f64).collect();
                            let m = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let l = m + col.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
                            for k in 0..K {
                                lp[ti * K + k] = col[k] - l;
                            }
                        }
                        ctc(&lp, t, &e.phon)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (bi, (e, r)) in bx.iter().zip(res).enumerate() {
            if let Some((l, g)) = r {
                let w = 1.0 / (e.phon.len().max(1) as f64 * batch as f64);
                lp_sum += l / e.phon.len().max(1) as f64;
                n_ok += 1;
                for ti in 0..e.t * up {
                    for k in 0..K {
                        gp[at(bi, ti, k)] = (g[ti * K + k] * w) as f32;
                    }
                }
            }
            let t = e.t;
            let z: Vec<f64> = (0..t).map(|ti| zt[bi * tmax + ti] as f64).collect();
            let (l, g) = survival(&z, e.tstar);
            lt_sum += l;
            for ti in 0..t {
                gt[bi * tmax + ti] = (g[ti] * w_turn / batch as f64) as f32;
            }
        }
        let warm = 200usize;
        let lr = if step < warm { peak * (step + 1) as f64 / warm as f64 } else { peak * 0.5 * (1.0 + (std::f64::consts::PI * ((step - warm) as f64 / (total - warm).max(1) as f64).min(1.0)).cos()) };
        let tt = (step + 1) as i32;
        prog.write_input(&dev, kgp, &gp);
        prog.write_input(&dev, kgt, &gt);
        prog.write_input(&dev, kh, &[lr as f32, (1.0 - 0.9f64.powi(tt)) as f32, (1.0 - 0.999f64.powi(tt)) as f32, 0.0, 1.0, 0.0]);
        prog.run(&dev);
        let gn = prog.read(&dev, 0)[0];
        if !gn.is_finite() {
            eprintln!("step {}: non-finite grad norm {gn} (ctc {lp_sum} turn {lt_sum} ok {n_ok})", step + 1);
            skipped += 1;
            if skipped > 20 {
                return Err(anyhow!("too many non-finite steps"));
            }
            continue;
        }
        for i in 0..np {
            prog.swap_input_output(kp[i], op + i);
            prog.swap_input_output(km[i], om + i);
            prog.swap_input_output(kv[i], ov + i);
        }
        step += 1;
        let (lpm, ltm) = (lp_sum / n_ok.max(1) as f64, lt_sum / batch as f64);
        accp += lpm;
        acct += ltm;
        accn += 1;
        writeln!(log, "{step}\t{lpm}\t{ltm}\t{gn}\t{lr:.3e}\t{:.0}", ts.elapsed().as_secs_f64() * 1e3)?;
        if step % 50 == 0 {
            eprintln!("step {step:>6}  ctc ok {n_ok}/{batch}  ctc/phone {:.4}  turn {:.3}  |g| {gn:.3}  lr {lr:.2e}  {:.0} ms/step", accp / accn as f64, acct / accn as f64, ts.elapsed().as_secs_f64() * 1e3);
            accp = 0.0;
            acct = 0.0;
            accn = 0;
        }
        if step % 500 == 0 || stop.exists() {
            save(&prog, step)?;
            if stop.exists() {
                std::fs::remove_file(&stop).ok();
                eprintln!("STOP: saved step {step}");
                return Ok(());
            }
        }
    }
    save(&prog, step)?;
    eprintln!("saved {} at step {step}", ck_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ctc_grad_matches_fd() {
        let t = 6;
        let y = vec![3usize, 5, 5];
        let mut r = Rng::new(3);
        let z: Vec<f64> = (0..t * K).map(|_| (r.below(1000) as f64 / 300.0) - 1.5).collect();
        let lsm = |z: &[f64]| -> Vec<f64> {
            let mut o = vec![0.0; z.len()];
            for ti in 0..t {
                let c = &z[ti * K..(ti + 1) * K];
                let m = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let l = m + c.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
                for k in 0..K {
                    o[ti * K + k] = c[k] - l;
                }
            }
            o
        };
        let (l0, g) = ctc(&lsm(&z), t, &y).unwrap();
        for &i in &[0usize, 3 * K + 5, 4 * K, 5 * K + 3, 2 * K + 7] {
            let mut z2 = z.clone();
            z2[i] += 1e-5;
            let (l1, _) = ctc(&lsm(&z2), t, &y).unwrap();
            assert!(((l1 - l0) / 1e-5 - g[i]).abs() < 1e-4, "{i}: fd {} vs {}", (l1 - l0) / 1e-5, g[i]);
        }
    }
}
