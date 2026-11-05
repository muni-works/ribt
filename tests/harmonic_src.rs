//! §29.5 harmonic source: the analytic STFT in `dsp::harmonic_feats` equals
//! the explicit DFT of the sine excitation it stands for (constant F0), and
//! the host-carried phase makes two chunks equal one window.
use ribt::dsp;
use std::f64::consts::PI;

#[test]
fn analytic_stft_matches_explicit() {
    let (sr, n, hop, nb, frames) = (24000usize, 1024usize, 256usize, 86usize, 12usize);
    let f = 157.3f64;
    let phi0 = 0.7;
    let f0 = vec![f as f32; frames];
    let (h, _) = dsp::harmonic_feats(&f0, frames, sr, n, hop, nb, phi0);
    let df = sr as f64 / n as f64;
    let w = dsp::hann(n);
    let nh = ((nb as f64 * df + 4.0 * df) / f).ceil() as usize;
    let mut worst = 0.0f64;
    let mut peak = 0.0f64;
    for t in 0..frames {
        let c = (t * hop + hop / 2) as f64; // frame centre, where φ = phi0 + 2π f (c − c0)/sr
        let s = c - (n / 2) as f64;
        for k in 0..nb {
            let (mut re, mut im) = (0.0, 0.0);
            for m in 0..n {
                let tt = s + m as f64;
                let phi = phi0 + 2.0 * PI * f * (tt - (hop / 2) as f64) / sr as f64;
                let x: f64 = (1..=nh).filter(|&q| (q as f64) * f < nb as f64 * df + 4.0 * df).map(|q| (q as f64 * phi).sin()).sum();
                let th = 2.0 * PI * (k * m) as f64 / n as f64;
                re += w[m] * x * th.cos();
                im -= w[m] * x * th.sin();
            }
            let (re, im) = (re * 4.0 / n as f64, im * 4.0 / n as f64);
            let (gr, gi) = (h[k * frames + t] as f64, h[(nb + k) * frames + t] as f64);
            worst = worst.max((re - gr).abs().max((im - gi).abs()));
            peak = peak.max((re * re + im * im).sqrt());
        }
    }
    // the analytic form drops the negative-frequency image and the far sidelobes
    assert!(peak > 0.9 && worst < 0.02 * peak, "worst {worst} peak {peak}");
}

#[test]
fn phase_carries_across_chunks() {
    let (sr, n, hop, nb) = (24000usize, 1024usize, 256usize, 86usize);
    let f0: Vec<f32> = (0..40).map(|t| if (10..14).contains(&t) { 0.0 } else { 120.0 + 3.0 * t as f32 }).collect();
    let (whole, ph) = dsp::harmonic_feats(&f0, 40, sr, n, hop, nb, 0.0);
    let (b, _) = dsp::harmonic_feats(&f0[25..], 15, sr, n, hop, nb, ph[25]);
    for r in 0..2 * nb {
        for t in 0..15 {
            assert!((whole[r * 40 + 25 + t] - b[r * 15 + t]).abs() < 1e-5);
        }
    }
}
