//! V2 acoustic model (docs §14): derived backward vs central finite
//! differences on the f64 interpreter, tiny config, every parameter tensor.

use ribt::acoustic::{self, Dims, TConfig};
use ribt::model::Rng;

struct Batch {
    data: Vec<(&'static str, Vec<f32>)>,
}

/// Random but well-formed batch: 2 utterances of 5 and 3 tokens (N 6),
/// durations summing to 11 and 7 frames (T 12), reference crops of 6/4.
fn batch(tc: &TConfig, d: Dims, rng: &mut Rng) -> Batch {
    let toks: Vec<Vec<u16>> = vec![vec![0, 3, 5, 2, 0], vec![0, 6, 0]];
    let durs: Vec<Vec<u32>> = vec![vec![1, 3, 2, 4, 1], vec![2, 4, 1]];
    let rlen = [6usize, 4];
    let (b, n, t, r) = (d.batch, d.n, d.t, d.r);
    let mut onehot = vec![0.0f32; b * tc.vocab * n];
    let mut tmask = vec![0.0f32; b * n];
    let mut fmask = vec![0.0f32; b * t];
    let mut rw = vec![0.0f32; b * r];
    for (bi, row) in toks.iter().enumerate() {
        for (i, &id) in row.iter().enumerate() {
            onehot[(bi * tc.vocab + id as usize) * n + i] = 1.0;
            tmask[bi * n + i] = 1.0;
        }
        let fr: u32 = durs[bi].iter().sum();
        for f in 0..fr as usize {
            fmask[bi * t + f] = 1.0;
        }
        for k in 0..rlen[bi] {
            rw[bi * r + k] = 1.0 / rlen[bi] as f32;
        }
    }
    let dr: Vec<&[u32]> = durs.iter().map(|v| v.as_slice()).collect();
    let align = acoustic::align_matrix(&dr, n, t);
    let ntok: f32 = tmask.iter().sum();
    let nfr: f32 = fmask.iter().sum();
    let tw: Vec<f32> = tmask.iter().map(|m| m / ntok).collect();
    let fw: Vec<f32> = fmask.iter().map(|m| m / nfr).collect();
    let p_tok: Vec<f32> = tmask.iter().map(|m| m * rng.normal() as f32).collect();
    let e_tok: Vec<f32> = tmask.iter().map(|m| m * rng.normal() as f32).collect();
    let mut ld = vec![0.0f32; b * n];
    for (bi, d) in durs.iter().enumerate() {
        for (i, &k) in d.iter().enumerate() {
            ld[bi * n + i] = (k as f32 + 1.0).ln();
        }
    }
    let mel: Vec<f32> = (0..b * tc.n_mels * t).map(|_| (-5.0 + 2.0 * rng.normal()) as f32).collect();
    let refmel: Vec<f32> = (0..b * tc.n_mels * r).map(|_| (-5.0 + 2.0 * rng.normal()) as f32).collect();
    let lf0: Vec<f32> = (0..b * t).map(|_| rng.normal() as f32).collect();
    let vuv: Vec<f32> = (0..b * t).map(|i| if fmask[i] > 0.0 && rng.uniform() < 0.6 { 1.0 } else { 0.0 }).collect();
    let nv: f32 = vuv.iter().sum::<f32>().max(1.0);
    let vw: Vec<f32> = vuv.iter().map(|v| v / nv).collect();
    Batch {
        data: vec![
            ("onehot", onehot),
            ("tmask", tmask),
            ("refmel", refmel),
            ("rw", rw),
            ("align", align),
            ("fmask", fmask),
            ("p_tok", p_tok),
            ("e_tok", e_tok),
            ("ld_tok", ld),
            ("tw", tw),
            ("mel", mel),
            ("lf0", lf0),
            ("vuv", vuv),
            ("fw", fw),
            ("vw", vw),
        ],
    }
}

#[test]
fn acoustic_gradients_match_finite_differences() {
    let tc = TConfig::tiny();
    let d = Dims { batch: 2, n: 6, t: 12, r: 6 };
    let specs = tc.params();
    let mut rng = Rng::new(3);
    let params: Vec<Vec<f32>> = acoustic::init_params(&tc, 2).into_iter().map(|p| p.iter().map(|v| v + (0.1 * rng.normal()) as f32).collect()).collect();
    let bt = batch(&tc, d, &mut rng);
    let mut tg = acoustic::tts_graph(&tc, d);
    let data: Vec<(&str, &[f32])> = bt.data.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let base: Vec<Vec<f64>> = acoustic::assemble(&tg.b, &specs, &params, None, &data).iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
    let total = tg.losses[0];
    let grads = tg.b.g.backward(total, &tg.params).unwrap();
    let an = tg.b.g.eval(&base, &grads);
    let loss = |inp: &[Vec<f64>]| tg.b.g.eval(inp, &[total])[0][0];
    let parts = tg.b.g.eval(&base, &tg.losses);
    eprintln!("losses {:?}", parts.iter().map(|v| v[0]).collect::<Vec<_>>());
    let mut worst: f64 = 0.0;
    let mut n = 0;
    for (i, &pid) in tg.params.iter().enumerate() {
        let k = tg.b.input_index(pid);
        for _ in 0..3 {
            let j = rng.below(base[k].len());
            let h = 1e-6;
            let mut p = base.clone();
            p[k][j] += h;
            let mut m = base.clone();
            m[k][j] -= h;
            let fd = (loss(&p) - loss(&m)) / (2.0 * h);
            let a = an[i][j];
            let rel = (fd - a).abs() / fd.abs().max(a.abs()).max(1e-6);
            if rel > 1e-4 {
                eprintln!("  {}[{j}]: analytic {a:.6e} fd {fd:.6e} rel {rel:.1e}", specs[i].name);
            }
            worst = worst.max(rel);
            n += 1;
        }
    }
    eprintln!("acoustic tiny: {} tensors, {n} coords, worst rel {worst:.2e}", specs.len());
    assert!(worst < 1e-4, "{worst}");
    let _ = &mut tg;
}
