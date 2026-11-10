//! Minimal repros of the tang IR gaps hit building V1 (see the ribt paper).
//!   cargo run --release --example ir_gaps -- buffers   # panics today on Metal
//!   cargo run --release --example ir_gaps -- scatter   # prints the backward's cost
//!   cargo run --release --example ir_gaps -- cuda-typed-let     # codegen only, any host
//!   cargo run --release --example ir_gaps -- cuda-array-length  # codegen only, any host
//!   cargo run --release --no-default-features --features cuda --example ir_gaps -- cuda-mma  # 3090
//!   cargo run --release --example ir_gaps -- reshape-guard  # gap 8: substitute_maps panics
//!   cargo run --release --example ir_gaps -- reshape2       # gap 8: reshape of a reshape
//!   cargo run --release --example ir_gaps -- siblings       # gap 9: >31 buffers after merge_siblings
//!   cargo run --release --no-default-features --features cuda --example ir_gaps -- cuda-mha [K]  # gap 11: attention stage K alone (0-6), §20
//!   cargo run --release --example ir_gaps -- point-reduce   # gap 10: 1-point consumer inlines a big reduction

use tang_compile::tensor::{Arg, Form, Op, TensorGraph, View};
use tang_expr::ExprId;

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("buffers") => {
            // a map over 32 inputs is one kernel binding 33 buffers; Metal allows 31
            let mut g = TensorGraph::new();
            let xs: Vec<_> = (0..32).map(|_| g.input(&[4])).collect();
            let y = g.map(&xs, |v| v.iter().fold(ExprId::ZERO, |a, &x| a + x)).unwrap();
            // (on the build's backend: Metal caps at 31, wgpu's tang-gpu device at 16)
            let dev = ribt::runtime::Device::new();
            let data = vec![1.0f32; 4];
            let inputs: Vec<&[f32]> = (0..32).map(|_| data.as_slice()).collect();
            let mut p = ribt::runtime::Program::from_optimized(&dev, g, vec![y], &inputs, std::time::Instant::now());
            p.run(&dev);
            println!("{}: sum = {:?}", ribt::runtime::BACKEND, p.read(&dev, 0));
        }
        Some("scatter") => {
            // STFT framing x[b, hop·t + n] read unguarded vs guarded: the
            // backward of the unguarded read keeps every (t, n) as a gated
            // reduce dim; the guarded one eliminates n.
            for guard in [false, true] {
                let (l, hop, nfft) = (4096usize, 256usize, 1024usize);
                let frames = (l - nfft) / hop + 1;
                let mut g = TensorGraph::new();
                let x = g.input(&[l]);
                let mut f = Form::zero(2);
                f.coefs = vec![hop as i64, 1];
                let u = g
                    .comp(&[frames, nfft], &[], vec![Arg { src: x, view: View { forms: vec![f], guard, gather: None } }], None, |v| vec![v[0] * v[0]])
                    .unwrap();
                let s = g.reduce(u, 1, tang_compile::tensor::Monoid::sum()).unwrap();
                let s = g.reduce(s, 0, tang_compile::tensor::Monoid::sum()).unwrap();
                let dx = g.backward(s, &[x]).unwrap()[0];
                let red: Vec<usize> = match &g.nodes()[dx.index()].op {
                    Op::Comp(c) => c.red.iter().map(|d| d.n()).collect(),
                    _ => vec![],
                };
                println!("guard={guard}: dx is [{l}] reducing over {red:?} -> {} body evals", l * red.iter().product::<usize>());
            }
        }
        Some("cuda-typed-let") => {
            // a tiled conv with a small (≤16) inner reduce extent unrolls its
            // taps as `let k: u32 = 3u;` (tensor_tile.rs); wgsl_to_cuda turns a
            // typed `let` into `auto k: u32 = 3u;`, which nvrtc rejects
            let (g, outs) = conv_graph();
            let (g, outs) = ribt::runtime::optimize(&g, &outs);
            let p = g.to_cuda(&outs);
            let bad: Vec<String> = p
                .kernels
                .iter()
                .flat_map(|k| k.source.lines().filter(|l| l.trim_start().starts_with("auto ") && l.contains(": u32 =")).map(|l| l.trim().to_string()).collect::<Vec<_>>())
                .collect();
            println!("{} kernels, {} `auto name: u32 = …` lines, e.g. {:?}", p.kernels.len(), bad.len(), bad.first());
        }
        Some("cuda-array-length") => {
            // tiled kernels bound-check guarded gathers with WGSL's
            // `arrayLength(&bJ)`; wgsl_to_cuda passes it through, nvrtc fails.
            // The V1 inference graph at 187 frames (a 2 s clip) hits it.
            let cfg = ribt::model::Config::v1();
            let inf = ribt::model::infer_graph(&cfg, 1, 187);
            let (g, outs) = ribt::runtime::optimize(&inf.b.g, &[inf.audio]);
            let p = g.to_cuda(&outs);
            let n = p.kernels.iter().filter(|k| k.source.contains("arrayLength(")).count();
            println!("{} kernels, {n} still call arrayLength(..) after wgsl_to_cuda", p.kernels.len());
        }
        #[cfg(feature = "cuda")]
        Some("cuda-mma") => {
            // the same program with and without the wmma TF32 tiles vs the f64
            // interpreter: the mma build's gradients are off by ~10% relative
            let (g, outs) = linear_graph();
            let mut rng = ribt::model::Rng::new(3);
            let inputs: Vec<Vec<f32>> = [8 * 512 * 64, 1536 * 512, 8 * 1536 * 64].iter().map(|&n| (0..n).map(|_| (0.1 * rng.normal()) as f32).collect()).collect();
            let want = g.eval(&inputs.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect::<Vec<_>>(), &outs);
            let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
            for mma in ["0", "1"] {
                std::env::set_var("RIBT_CUDA_MMA", mma);
                let dev = ribt::runtime::Device::new();
                let mut p = ribt::runtime::Program::new(&dev, &g, &outs, &refs);
                p.run(&dev);
                let errs: Vec<String> = want
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        let got = p.read(&dev, i);
                        let scale = w.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
                        let e = got.iter().zip(w).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max) / scale;
                        format!("{e:.1e}")
                    })
                    .collect();
                println!("RIBT_CUDA_MMA={mma}: max rel err [y, dx, dw] of a 512→1536 linear = {errs:?}");
            }
        }
        Some(k @ ("reshape-guard" | "reshape2")) => {
            // map → reshape (→ reshape) → guarded offset read; the map is
            // inlinable (one read, no transcendental), so `substitute_maps`
            // inlines it and loses the reshape's source.
            let mut g = TensorGraph::new();
            let x = g.input(&[4, 6]);
            let m = g.map(&[x], |v| v[0] * ribt::ir::lit(2.0)).unwrap();
            let r = if k == "reshape2" {
                let r0 = g.reshape(m, &[24]).unwrap();
                g.reshape(r0, &[4, 2, 3]).unwrap()
            } else {
                g.reshape(m, &[4, 2, 3]).unwrap()
            };
            let mut f = Form::zero(3);
            f.coefs = vec![0, 1, 0];
            f.offset = -1;
            let v = View { forms: vec![Form { coefs: vec![1, 0, 0], offset: 0 }, f, Form { coefs: vec![0, 0, 1], offset: 0 }], guard: true, gather: None };
            let y = g.comp(&[4, 2, 3], &[], vec![Arg { src: r, view: v }], None, |v| vec![v[0] + ribt::ir::lit(1.0)]).unwrap();
            let res = std::panic::catch_unwind(|| ribt::runtime::optimize(&g, &[y]).0.count_comps(&[]));
            println!("{k}: optimize {}", if res.is_ok() { "ok (fixed)" } else { "PANICKED (gap 8 still open)" });
        }
        Some("siblings") => {
            // 16 sibling scalar reductions, each reading 2 feature maps, then
            // one map summing them: substitute_maps + merge_siblings put
            // them into one kernel binding 32+ buffers (Metal's cap is 31).
            let mut g = TensorGraph::new();
            let mut ss = Vec::new();
            let mut ins = Vec::new();
            for _ in 0..16 {
                let a = g.input(&[64]);
                let b = g.input(&[64]);
                ins.push(a);
                ins.push(b);
                let d = g.map(&[a, b], |v| v[0] - v[1]).unwrap();
                ss.push(g.reduce(d, 0, tang_compile::tensor::Monoid::sum()).unwrap());
            }
            let y = g.map(&ss, |v| v.iter().fold(ExprId::ZERO, |a, &x| a + x)).unwrap();
            let dev = ribt::runtime::Device::new();
            let data = vec![1.0f32; 64];
            let inputs: Vec<&[f32]> = ins.iter().map(|_| data.as_slice()).collect();
            let mut p = ribt::runtime::Program::new(&dev, &g, &[y], &inputs);
            p.run(&dev);
            println!("siblings: {} kernels, sum = {:?} (ok: gap 9 fixed)", p.kernels(), p.read(&dev, 0));
        }
        Some("cuda-mha") => {
            // gap 11 (§20): rung-1 encoder self-attention, stage by stage, GPU vs f64 interpreter
            use ribt::ir::{arg, lit, pick};
            use tang::Scalar;
            use tang_compile::tensor::Monoid;
            let (b, d, n, heads) = (3usize, 32usize, 24usize, 2usize);
            let dh = d / heads;
            let mut g = TensorGraph::new();
            let x = g.input(&[b, d, n]);
            let wq = g.input(&[d, d]);
            let wk = g.input(&[d, d]);
            let kb = g.input(&[b, 1, n]);
            let q = ribt::ir::linear(&mut g, x, wq);
            let q = g.reshape(q, &[b, heads, dh, n]).unwrap();
            let k = ribt::ir::linear(&mut g, x, wk);
            let k = g.reshape(k, &[b, heads, dh, n]).unwrap();
            let s0 = g.comp(&[b, heads, n, n], &[dh], vec![arg(q, pick(5, &[Some(0), Some(1), Some(4), Some(2)])), arg(k, pick(5, &[Some(0), Some(1), Some(4), Some(3)]))], Some(Monoid::sum()), |v| vec![v[0] * v[1]]).unwrap();
            let s = g.comp(&[b, heads, n, n], &[], vec![arg(s0, pick(4, &[Some(0), Some(1), Some(2), Some(3)])), arg(kb, pick(4, &[Some(0), None, Some(3)]))], None, |v| vec![v[0] * lit(0.25) + v[1]]).unwrap();
            let m = g.comp(&[b, heads, n], &[n], vec![arg(s, pick(4, &[Some(0), Some(1), Some(2), Some(3)]))], Some(Monoid::max()), |v| vec![v[0]]).unwrap();
            let p = g.comp(&[b, heads, n, n], &[], vec![arg(s, pick(4, &[Some(0), Some(1), Some(2), Some(3)])), arg(m, pick(4, &[Some(0), Some(1), Some(2)]))], None, |v| vec![(v[0] - v[1]).exp()]).unwrap();
            let l = g.comp(&[b, heads, n], &[n], vec![arg(p, pick(4, &[Some(0), Some(1), Some(2), Some(3)]))], Some(Monoid::sum()), |v| vec![v[0]]).unwrap();
            let wv = g.input(&[d, d]);
            let v = ribt::ir::linear(&mut g, x, wv);
            let v = g.reshape(v, &[b, heads, dh, n]).unwrap();
            let pv = g.comp(&[b, heads, dh, n], &[n], vec![arg(p, pick(5, &[Some(0), Some(1), Some(3), Some(4)])), arg(v, pick(5, &[Some(0), Some(1), Some(2), Some(4)]))], Some(Monoid::sum()), |v| vec![v[0] * v[1]]).unwrap();
            let o = g.comp(&[b, heads, dh, n], &[], vec![arg(pv, pick(4, &[Some(0), Some(1), Some(2), Some(3)])), arg(l, pick(4, &[Some(0), Some(1), Some(3)]))], None, |v| vec![v[0] / v[1]]).unwrap();
            // one output at a time: the intermediates fuse into their consumers
            let names = ["q", "scores", "scores+bias", "rowmax", "exp", "rowsum", "attn", "p@v", "v"];
            let all = [q, s0, s, m, p, l, o, pv, v];
            let which: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(99);
            let (outs, names): (Vec<_>, Vec<&str>) = if which < all.len() { (vec![all[which]], vec![names[which]]) } else { (all.to_vec(), names.to_vec()) };
            if std::env::var("DUMP").is_ok() {
                let (g2, o2) = ribt::runtime::optimize(&g, &outs);
                // the runtime's default CUDA path: WGSL program, transpiled
                for k in g2.to_program(&o2, tang_compile::tensor::wgsl::Target::Wgsl).kernels {
                    println!("// ---- kernel wg {:?}\n{}", k.workgroup_size, k.source);
                }
                return;
            }
            let mut r = ribt::model::Rng::new(3);
            let mut rnd = |k: usize, sc: f64| -> Vec<f32> { (0..k).map(|_| (sc * r.normal()) as f32).collect() };
            let mut kbv = vec![0.0f32; b * n];
            for bi in 0..b { for j in (n - 5 * bi)..n { kbv[bi * n + j] = -1e4; } }
            let ins = vec![rnd(b * d * n, 1.0), rnd(d * d, 0.3), rnd(d * d, 0.3), kbv, rnd(d * d, 0.3)];
            let i64s: Vec<Vec<f64>> = ins.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
            let want = g.eval(&i64s, &outs);
            let dev = ribt::runtime::Device::new();
            let refs: Vec<&[f32]> = ins.iter().map(|v| v.as_slice()).collect();
            let mut prog = ribt::runtime::Program::new(&dev, &g, &outs, &refs);
            prog.run(&dev);
            for (i, nm) in names.iter().enumerate() {
                let got = prog.read(&dev, i);
                let sc = want[i].iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-12);
                let e = got.iter().zip(&want[i]).map(|(a, w)| (*a as f64 - w).abs()).fold(0.0f64, f64::max) / sc;
                println!("{}: {nm:<12} rel {e:.2e}", ribt::runtime::BACKEND);
            }
        }
        Some("point-reduce") => {
            // a bias gradient: Σ over a [16, 64, 128·128] map per channel,
            // then its squared norm (one point). The 64-point reduction is
            // inlined into the 1-point kernel as a serial loop.
            let mut g = TensorGraph::new();
            let dy = g.input(&[16, 64, 16384]);
            let db = g.reduce(dy, 2, tang_compile::tensor::Monoid::sum()).unwrap();
            let db = g.reduce(db, 0, tang_compile::tensor::Monoid::sum()).unwrap();
            let sq = g.map(&[db], |v| v[0] * v[0]).unwrap();
            let n = g.reduce(sq, 0, tang_compile::tensor::Monoid::sum()).unwrap();
            let dev = ribt::runtime::Device::new();
            let data = vec![0.001f32; 16 * 64 * 16384];
            let mut p = ribt::runtime::Program::new(&dev, &g, &[db, n], &[&data]);
            let ms = p.time(&dev, 5);
            println!("point-reduce: {} kernels, {ms:.2} ms for 16.8M adds (~0.05 ms at bandwidth)", p.kernels());
        }
        _ => eprintln!("ir_gaps buffers|scatter|cuda-typed-let|cuda-array-length|cuda-mma|reshape-guard|reshape2|siblings|point-reduce"),
    }
}

/// One conv1d k7 (64→64 channels, batch 2 × 64 frames) and the
/// gradients of a weighted sum of its output: `[y, dL/dx, dL/dw]`.
fn conv_graph() -> (TensorGraph, Vec<tang_compile::tensor::TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[2, 64, 64]);
    let w = g.input(&[64, 64, 7]);
    let b = g.input(&[64]);
    let r = g.input(&[2, 64, 64]);
    let y = ribt::ir::conv1d(&mut g, x, w, b);
    let yr = g.map(&[y, r], |v| v[0] * v[1]).unwrap();
    let s = g.reduce(yr, 2, tang_compile::tensor::Monoid::sum()).unwrap();
    let s = g.reduce(s, 1, tang_compile::tensor::Monoid::sum()).unwrap();
    let s = g.reduce(s, 0, tang_compile::tensor::Monoid::sum()).unwrap();
    let d = g.backward(s, &[x, w]).unwrap();
    (g, vec![y, d[0], d[1]])
}

/// A per-frame linear 512→1536 (batch 8 × 64 frames, the ConvNeXt
/// expansion) and the gradients of a weighted sum: `[y, dL/dx, dL/dw]`.
fn linear_graph() -> (TensorGraph, Vec<tang_compile::tensor::TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[8, 512, 64]);
    let w = g.input(&[1536, 512]);
    let r = g.input(&[8, 1536, 64]);
    let y = ribt::ir::linear(&mut g, x, w);
    let yr = g.map(&[y, r], |v| v[0] * v[1]).unwrap();
    let s = g.reduce(yr, 2, tang_compile::tensor::Monoid::sum()).unwrap();
    let s = g.reduce(s, 1, tang_compile::tensor::Monoid::sum()).unwrap();
    let s = g.reduce(s, 0, tang_compile::tensor::Monoid::sum()).unwrap();
    let d = g.backward(s, &[x, w]).unwrap();
    (g, vec![y, d[0], d[1]])
}
