//! GPU checks (backend per cargo feature) (ignored by default; run under the GPU lock):
//!   ~/.cache/gpu-lock.sh ~/.cargo/bin/cargo test --release --test gpu -- --ignored

use ribt::dsp;
use ribt::model::{self, Config, Rng};
use ribt::runtime::{Device, Program};

#[test]
#[ignore]
fn istft_inverts_stft_v1_on_gpu() {
    let cfg = Config::v1();
    let frames = 94;
    let mut rng = Rng::new(2);
    let x: Vec<f32> = (0..frames * cfg.hop)
        .map(|i| (0.4 * (i as f64 * 0.03).sin() + 0.1 * rng.normal()) as f32)
        .collect();
    let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let (re, im, f) = dsp::stft_same_ref(&xd, cfg.n_fft, cfg.hop);
    assert_eq!(f, frames);
    let ig = model::istft_graph(&cfg, 1, frames);
    let re: Vec<f32> = re.iter().map(|&v| v as f32).collect();
    let im: Vec<f32> = im.iter().map(|&v| v as f32).collect();
    let inputs = ig.b.assemble(&[], None, &[("re", &re), ("im", &im)]);
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let dev = Device::new();
    let mut p = Program::new(&dev, &ig.b.g, &[ig.audio], &refs);
    p.run(&dev);
    let y = p.read(&dev, 0);
    let err = y.iter().zip(&x).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("v1 istft(stft(x)) on {}: max err {err:.2e} over {} samples", ribt::runtime::BACKEND, y.len());
    assert!(err < 1e-4, "{err}");
}

#[test]
#[ignore]
fn train_step_matches_interpreter() {
    let exe = env!("CARGO_BIN_EXE_ribt");
    for cfg in ["tiny", "check"] {
        let st = std::process::Command::new(exe).args(["gpucheck", "--cfg", cfg]).status().unwrap();
        assert!(st.success(), "{cfg}");
    }
}
