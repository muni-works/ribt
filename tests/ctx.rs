//! Context-aware prosody path (docs §25): an untrained path is exactly the
//! base encoder, and its gradients match central finite differences on the
//! f64 interpreter (tiny config).

use ribt::acoustic::{self, CtxConfig, Dims, TConfig, CTX_ROLES};
use ribt::model::Rng;

fn data(tc: &TConfig, d: Dims, m: usize, rng: &mut Rng) -> Vec<(&'static str, Vec<f32>)> {
    let toks: Vec<Vec<u16>> = vec![vec![0, 3, 5, 2, 0], vec![0, 6, 0]];
    let prev: Vec<Vec<u16>> = vec![vec![0, 5, 2, 4, 0], vec![0, 0]];
    let (b, n, r) = (d.batch, d.n, d.r);
    let nr = CTX_ROLES.len();
    let mut onehot = vec![0.0f32; b * tc.vocab * n];
    let mut tmask = vec![0.0f32; b * n];
    let mut po = vec![0.0f32; b * tc.vocab * m];
    let mut pm = vec![0.0f32; b * m];
    let mut role = vec![0.0f32; b * nr];
    let mut rw = vec![0.0f32; b * r];
    for bi in 0..b {
        for (i, &id) in toks[bi].iter().enumerate() {
            onehot[(bi * tc.vocab + id as usize) * n + i] = 1.0;
            tmask[bi * n + i] = 1.0;
        }
        for (i, &id) in prev[bi].iter().enumerate() {
            po[(bi * tc.vocab + id as usize) * m + i] = 1.0;
            pm[bi * m + i] = 1.0;
        }
        role[bi * nr + 1 + bi] = 1.0;
        for k in 0..4 {
            rw[bi * r + k] = 0.25;
        }
    }
    let ntok: f32 = tmask.iter().sum();
    let rnd = |rng: &mut Rng, k: usize| (0..k).map(|_| rng.normal() as f32).collect::<Vec<f32>>();
    vec![
        ("onehot", onehot.clone()),
        ("tmask", tmask.clone()),
        ("refmel", rnd(rng, b * tc.n_mels * r).iter().map(|v| v - 5.0).collect()),
        ("rw", rw),
        ("prev", po),
        ("pmask", pm),
        ("role", role),
        ("emph", tmask.iter().enumerate().map(|(i, m)| if i % 3 == 1 { *m } else { 0.0 }).collect()),
        ("gate", vec![1.0, 0.7]),
        ("p_tok", tmask.iter().map(|m| m * rng.normal() as f32).collect()),
        ("e_tok", tmask.iter().map(|m| m * rng.normal() as f32).collect()),
        ("ld_tok", tmask.iter().map(|m| m * (1.0 + rng.uniform() as f32)).collect()),
        ("tw", tmask.iter().map(|m| m / ntok).collect()),
    ]
}

fn f64s(v: Vec<Vec<f32>>) -> Vec<Vec<f64>> {
    v.into_iter().map(|x| x.into_iter().map(|y| y as f64).collect()).collect()
}

#[test]
fn untrained_ctx_path_is_the_base_encoder() {
    let (tc, cc) = (TConfig::tiny(), CtxConfig::tiny());
    let d = Dims { batch: 2, n: 6, t: 8, r: 6 };
    let m = 5;
    let mut rng = Rng::new(5);
    let base_p = acoustic::init_params(&tc, 2);
    let specs = acoustic::ctx_specs(&tc, &cc);
    let mut all = base_p.clone();
    let mut r2 = Rng::new(9);
    all.extend(cc.params(&tc).iter().map(|s| s.init_data(&mut r2)));
    let dt = data(&tc, d, m, &mut rng);
    let dr: Vec<(&str, &[f32])> = dt.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let eb = acoustic::enc_graph(&tc, d);
    let ec = acoustic::ctx_enc_graph(&tc, &cc, d, m);
    let ib = f64s(acoustic::assemble(&eb.b, &tc.params(), &base_p, None, &dr));
    let ic = f64s(acoustic::assemble(&ec.b, &specs, &all, None, &dr));
    let ob = eb.b.g.eval(&ib, &[eb.enc.ld, eb.enc.pitch, eb.enc.energy]);
    let oc = ec.b.g.eval(&ic, &[ec.enc.ld, ec.enc.pitch, ec.enc.energy]);
    let worst = ob.iter().zip(&oc).flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs())).fold(0.0, f64::max);
    eprintln!("zero-init ctx vs base: max |Δ| {worst:.2e}");
    assert!(worst < 1e-12, "{worst}");
}

#[test]
fn ctx_gradients_match_finite_differences() {
    let (tc, cc) = (TConfig::tiny(), CtxConfig::tiny());
    let d = Dims { batch: 2, n: 6, t: 8, r: 6 };
    let m = 5;
    let mut rng = Rng::new(3);
    let specs = acoustic::ctx_specs(&tc, &cc);
    let mut r2 = Rng::new(4);
    // non-zero AdaLN weights so every ctx tensor carries gradient
    let params: Vec<Vec<f32>> = specs.iter().map(|s| s.init_data(&mut r2).iter().map(|v| v + (0.2 * r2.normal()) as f32).collect()).collect();
    let dt = data(&tc, d, m, &mut rng);
    let dr: Vec<(&str, &[f32])> = dt.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let mut ct = acoustic::ctx_train_graph(&tc, &cc, d, m);
    let base = f64s(acoustic::assemble(&ct.b, &specs, &params, None, &dr));
    let total = ct.losses[0];
    let ids: Vec<_> = ct.params[ct.n_base..].to_vec();
    let grads = ct.b.g.backward(total, &ids).unwrap();
    let an = ct.b.g.eval(&base, &grads);
    let loss = |inp: &[Vec<f64>]| ct.b.g.eval(inp, &[total])[0][0];
    let mut worst: f64 = 0.0;
    for (i, &pid) in ids.iter().enumerate() {
        let k = ct.b.input_index(pid);
        for _ in 0..3 {
            let j = rng.below(base[k].len());
            let h = 1e-6;
            let mut p = base.clone();
            p[k][j] += h;
            let mut q = base.clone();
            q[k][j] -= h;
            let fd = (loss(&p) - loss(&q)) / (2.0 * h);
            let a = an[i][j];
            let rel = (fd - a).abs() / fd.abs().max(a.abs()).max(1e-6);
            if rel > 1e-4 {
                eprintln!("  {}[{j}]: analytic {a:.6e} fd {fd:.6e} rel {rel:.1e}", specs[ct.n_base + i].name);
            }
            worst = worst.max(rel);
        }
    }
    eprintln!("ctx tiny: {} ctx tensors, worst rel {worst:.2e}", ids.len());
    assert!(worst < 1e-4, "{worst}");
    let _ = &mut ct;
}
