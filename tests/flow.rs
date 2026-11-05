//! V2 rung 2 flow decoder (docs §15): derived backward of the CFM loss vs
//! central finite differences on the f64 interpreter, tiny config, every
//! parameter tensor, with a chunk-masked attention bias.

use ribt::flow::{self, FConfig, FDims};
use ribt::model::Rng;

#[test]
fn flow_gradients_match_finite_differences() {
    let fc = FConfig::tiny();
    let d = FDims { batch: 2, t: 10 };
    let specs = fc.params();
    let mut rng = Rng::new(5);
    // perturb: the AdaLN-zero gates and the zero output layer would
    // otherwise make most gradients exactly zero
    let params: Vec<Vec<f32>> = flow::init_params(&fc, 1).into_iter().map(|p| p.iter().map(|v| v + (0.3 * rng.normal()) as f32).collect()).collect();
    let lens = [10usize, 7];
    let (b, t, c) = (d.batch, d.t, fc.n_mels);
    let mut fmask = vec![0.0f32; b * t];
    for (bi, &n) in lens.iter().enumerate() {
        for f in 0..n {
            fmask[bi * t + f] = 1.0;
        }
    }
    let nfr: f32 = fmask.iter().sum();
    let fw: Vec<f32> = fmask.iter().map(|m| m / nfr).collect();
    let rnd = |rng: &mut Rng, n: usize| -> Vec<f32> { (0..n).map(|_| rng.normal() as f32).collect() };
    let xt = rnd(&mut rng, b * c * t);
    let mu = rnd(&mut rng, b * c * t);
    let hc = rnd(&mut rng, b * fc.dc * t);
    let ut = rnd(&mut rng, b * c * t);
    let mut temb = flow::time_feats(&fc, 0.3, 0.0);
    temb.extend(flow::time_feats(&fc, 0.8, 0.5));
    let abias = flow::attn_bias(&lens, t, Some((3, 1)));
    let fg = flow::flow_graph(&fc, d, true);
    let data: Vec<(&str, &[f32])> = vec![("xt", &xt), ("mu", &mu), ("hc", &hc), ("temb", &temb), ("fmask", &fmask), ("abias", &abias), ("ut", &ut), ("fw", &fw)];
    let base: Vec<Vec<f64>> = flow::assemble(&fg.b, &specs, &params, None, &data).iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
    let loss = fg.loss.unwrap();
    let mut g = fg.b.g.clone();
    let grads = g.backward(loss, &fg.params).unwrap();
    let an = g.eval(&base, &grads);
    let f = |inp: &[Vec<f64>]| g.eval(inp, &[loss])[0][0];
    eprintln!("loss {}", f(&base));
    let mut worst: f64 = 0.0;
    let mut n = 0;
    for (i, &pid) in fg.params.iter().enumerate() {
        let k = fg.b.input_index(pid);
        for _ in 0..3 {
            let j = rng.below(base[k].len());
            let h = 1e-6;
            let mut p = base.clone();
            p[k][j] += h;
            let mut m = base.clone();
            m[k][j] -= h;
            let fd = (f(&p) - f(&m)) / (2.0 * h);
            let a = an[i][j];
            let rel = (fd - a).abs() / fd.abs().max(a.abs()).max(1e-6);
            if rel > 1e-4 {
                eprintln!("  {}[{j}]: analytic {a:.6e} fd {fd:.6e} rel {rel:.1e}", specs[i].name);
            }
            worst = worst.max(rel);
            n += 1;
        }
    }
    eprintln!("flow tiny: {} tensors, {n} coords, worst rel {worst:.2e}", specs.len());
    assert!(worst < 1e-4, "{worst}");
}
