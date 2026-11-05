//! Host-side constants (windows, DFT bases, mel filterbank, iSTFT envelope)
//! and f64 CPU references used by the tests. Conventions follow torch /
//! torchaudio: periodic Hann, `MelSpectrogram(power=1, mel_scale="htk",
//! norm=None, f_min=0, f_max=sr/2)`, Vocos `padding="same"`.

use std::f64::consts::PI;

/// Periodic Hann (`torch.hann_window(n)`).
pub fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
        .collect()
}

/// Analysis bases `[n_fft, bins]` with the window folded in:
/// `re[k] = Σ_n x[n]·w[n]·cos(2πkn/N)`, `im[k] = −Σ_n x[n]·w[n]·sin(2πkn/N)`.
pub fn stft_bases(n_fft: usize) -> (Vec<f64>, Vec<f64>) {
    let bins = n_fft / 2 + 1;
    let w = hann(n_fft);
    let mut c = vec![0.0; n_fft * bins];
    let mut s = vec![0.0; n_fft * bins];
    for n in 0..n_fft {
        for k in 0..bins {
            let th = 2.0 * PI * ((k * n) % n_fft) as f64 / n_fft as f64;
            c[n * bins + k] = w[n] * th.cos();
            s[n * bins + k] = -w[n] * th.sin();
        }
    }
    (c, s)
}

/// Synthesis bases `[bins, n_fft]` for `irfft(re + i·im)·window`:
/// `frame[n] = Σ_k re[k]·C[k,n] − im[k]·S[k,n]` with
/// `C = c_k·cos(2πkn/N)·w[n]/N`, `S = c_k·sin(2πkn/N)·w[n]/N`, `c_k` = 1 at
/// DC/Nyquist, 2 elsewhere (the imaginary parts there drop out, like irfft).
pub fn istft_bases(n_fft: usize) -> (Vec<f64>, Vec<f64>) {
    let bins = n_fft / 2 + 1;
    let w = hann(n_fft);
    let mut c = vec![0.0; bins * n_fft];
    let mut s = vec![0.0; bins * n_fft];
    for k in 0..bins {
        let ck = if k == 0 || k == bins - 1 { 1.0 } else { 2.0 };
        for n in 0..n_fft {
            let th = 2.0 * PI * ((k * n) % n_fft) as f64 / n_fft as f64;
            c[k * n_fft + n] = ck * th.cos() * w[n] / n_fft as f64;
            s[k * n_fft + n] = ck * th.sin() * w[n] / n_fft as f64;
        }
    }
    (c, s)
}

/// Geometry of Vocos' "same"-padded iSTFT for `frames` frames: output
/// sample `m = hop·q + r` (q < frames, r < hop) reads frame `t = q − j + c0`
/// at in-frame index `n = r + base + hop·j`, `j ∈ 0..taps`.
#[derive(Clone, Copy, Debug)]
pub struct Ola {
    pub c0: i64,
    pub base: i64,
    pub taps: usize,
}

pub fn ola(n_fft: usize, hop: usize) -> Ola {
    assert_eq!(n_fft % hop, 0, "n_fft must be a multiple of hop");
    let pad = ((n_fft - hop) / 2) as i64;
    let c0 = (pad + hop as i64 - 1) / hop as i64;
    Ola {
        c0,
        base: pad - hop as i64 * c0,
        taps: n_fft / hop + 1,
    }
}

/// `1 / Σ_t w²[u − hop·t]` at each kept output sample, `[frames, hop]`.
pub fn istft_inv_env(n_fft: usize, hop: usize, frames: usize) -> Vec<f64> {
    let w = hann(n_fft);
    let pad = (n_fft - hop) / 2;
    let full = hop * (frames - 1) + n_fft;
    let mut env = vec![0.0; full];
    for t in 0..frames {
        for n in 0..n_fft {
            env[t * hop + n] += w[n] * w[n];
        }
    }
    (0..frames * hop)
        .map(|m| {
            let e = env[m + pad];
            assert!(e > 1e-11);
            1.0 / e
        })
        .collect()
}

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// torchaudio `melscale_fbanks(n_freqs, 0, sr/2, n_mels, sr, norm=None,
/// mel_scale="htk")`, `[bins, n_mels]`.
pub fn mel_fbank(sr: usize, n_fft: usize, n_mels: usize) -> Vec<f64> {
    let bins = n_fft / 2 + 1;
    let fmax = (sr / 2) as f64;
    let freqs: Vec<f64> = (0..bins)
        .map(|i| fmax * i as f64 / (bins - 1) as f64)
        .collect();
    let (m0, m1) = (hz_to_mel(0.0), hz_to_mel(fmax));
    let f_pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut fb = vec![0.0; bins * n_mels];
    for (i, &f) in freqs.iter().enumerate() {
        for m in 0..n_mels {
            let down = (f - f_pts[m]) / (f_pts[m + 1] - f_pts[m]);
            let up = (f_pts[m + 2] - f) / (f_pts[m + 2] - f_pts[m + 1]);
            fb[i * n_mels + m] = down.min(up).max(0.0);
        }
    }
    fb
}

/// Reflect-pad (`torch.nn.functional.pad(mode="reflect")`) by `p` each side.
pub fn reflect_pad(x: &[f32], p: usize) -> Vec<f32> {
    let n = x.len();
    assert!(n > p);
    (0..n + 2 * p)
        .map(|i| {
            let j = i as i64 - p as i64;
            let j = if j < 0 {
                -j
            } else if j >= n as i64 {
                2 * (n as i64 - 1) - j
            } else {
                j
            };
            x[j as usize]
        })
        .collect()
}

/// f64 reference: log-mel of an (already padded) signal, no centering:
/// `ln(max(fbᵀ·|STFT|, 1e-5))`, `[n_mels, frames]`.
pub fn log_mel_ref(x: &[f64], sr: usize, n_fft: usize, hop: usize, n_mels: usize) -> Vec<f64> {
    let mag = stft_mag_ref(x, n_fft, hop);
    let bins = n_fft / 2 + 1;
    let frames = mag.len() / bins;
    let fb = mel_fbank(sr, n_fft, n_mels);
    let mut out = vec![0.0; n_mels * frames];
    for t in 0..frames {
        for m in 0..n_mels {
            let s: f64 = (0..bins).map(|k| fb[k * n_mels + m] * mag[t * bins + k]).sum();
            out[m * frames + t] = s.max(1e-5).ln();
        }
    }
    out
}

/// f64 reference: `|STFT|`, Hann, no centering, `[frames, bins]`.
pub fn stft_mag_ref(x: &[f64], n_fft: usize, hop: usize) -> Vec<f64> {
    let bins = n_fft / 2 + 1;
    let frames = (x.len() - n_fft) / hop + 1;
    let w = hann(n_fft);
    let mut out = vec![0.0; frames * bins];
    for t in 0..frames {
        for k in 0..bins {
            let (mut re, mut im) = (0.0, 0.0);
            for n in 0..n_fft {
                let th = 2.0 * PI * ((k * n) % n_fft) as f64 / n_fft as f64;
                let v = x[t * hop + n] * w[n];
                re += v * th.cos();
                im -= v * th.sin();
            }
            out[t * bins + k] = (re * re + im * im).sqrt();
        }
    }
    out
}

/// f64 reference: complex STFT (for the round-trip test), `[bins, frames]`
/// re and im, Vocos "same" framing (reflect pad (n_fft−hop)/2).
pub fn stft_same_ref(x: &[f64], n_fft: usize, hop: usize) -> (Vec<f64>, Vec<f64>, usize) {
    let p = (n_fft - hop) / 2;
    let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let xp: Vec<f64> = reflect_pad(&xf, p).iter().map(|&v| v as f64).collect();
    let bins = n_fft / 2 + 1;
    let frames = (xp.len() - n_fft) / hop + 1;
    let w = hann(n_fft);
    let mut re = vec![0.0; bins * frames];
    let mut im = vec![0.0; bins * frames];
    for t in 0..frames {
        for k in 0..bins {
            for n in 0..n_fft {
                let th = 2.0 * PI * ((k * n) % n_fft) as f64 / n_fft as f64;
                let v = xp[t * hop + n] * w[n];
                re[k * frames + t] += v * th.cos();
                im[k * frames + t] -= v * th.sin();
            }
        }
    }
    (re, im, frames)
}

/// YIN F0 per mel frame (de Cheveigné & Kawahara 2002), on a 3x-decimated
/// copy of the reflect-padded clip `xp` (`frames·hop + 2·pad` samples, the
/// mel frontend's input). Frame `t` is centred where mel frame `t` is,
/// `t·hop + n_fft/2`. Search 50–500 Hz with a 30 ms window at sr/3; voiced
/// when the cumulative-mean-normalized difference dips below 0.25 and the
/// frame is not silent. Returns Hz, 0 for unvoiced.
pub fn yin_f0(xp: &[f32], sr: usize, n_fft: usize, hop: usize, frames: usize) -> Vec<f32> {
    let dec = 3;
    let fs = sr as f64 / dec as f64;
    let d: Vec<f64> = xp.chunks(dec).map(|c| c.iter().map(|&v| v as f64).sum::<f64>() / c.len() as f64).collect();
    let n = d.len() as i64;
    let at = |i: i64| d[i.clamp(0, n - 1) as usize];
    let w = (0.030 * fs) as i64;
    let tmin = (fs / 500.0).floor() as usize;
    let tmax = (fs / 50.0).ceil() as usize;
    let mut out = vec![0.0f32; frames];
    let mut diff = vec![0.0f64; tmax + 2];
    for (t, o) in out.iter_mut().enumerate() {
        let c = ((t * hop + n_fft / 2) / dec) as i64;
        let s0 = c - w / 2 - tmax as i64 / 2;
        let energy: f64 = (0..w).map(|j| at(s0 + j) * at(s0 + j)).sum::<f64>() / w as f64;
        if energy < 1e-6 {
            continue;
        }
        for tau in 1..=tmax + 1 {
            diff[tau] = (0..w).map(|j| {
                let e = at(s0 + j) - at(s0 + j + tau as i64);
                e * e
            }).sum();
        }
        // cumulative mean normalized difference
        let mut cmnd = vec![1.0f64; tmax + 2];
        let mut run = 0.0;
        for tau in 1..=tmax + 1 {
            run += diff[tau];
            cmnd[tau] = if run > 0.0 { diff[tau] * tau as f64 / run } else { 1.0 };
        }
        let mut best = None;
        let mut tau = tmin;
        while tau <= tmax {
            if cmnd[tau] < 0.15 {
                while tau < tmax && cmnd[tau + 1] < cmnd[tau] {
                    tau += 1;
                }
                best = Some(tau);
                break;
            }
            tau += 1;
        }
        let tau = best.unwrap_or_else(|| (tmin..=tmax).min_by(|&a, &b| cmnd[a].partial_cmp(&cmnd[b]).unwrap()).unwrap());
        if cmnd[tau] > 0.25 {
            continue;
        }
        // parabolic interpolation
        let (a, b, cc) = (cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]);
        let den = a - 2.0 * b + cc;
        let shift = if den.abs() > 1e-12 { (0.5 * (a - cc) / den).clamp(-1.0, 1.0) } else { 0.0 };
        *o = (fs / (tau as f64 + shift)) as f32;
    }
    out
}

/// `yin_f0` for a batch of padded crops laid out row-major, in parallel.
pub fn yin_f0_batch(xp: &[f32], batch: usize, sr: usize, n_fft: usize, hop: usize, frames: usize) -> Vec<f32> {
    let row = xp.len() / batch;
    let rows: Vec<&[f32]> = xp.chunks(row).collect();
    let parts: Vec<Vec<f32>> = std::thread::scope(|s| {
        let hs: Vec<_> = rows.iter().map(|r| s.spawn(move || yin_f0(r, sr, n_fft, hop, frames))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    parts.concat()
}

/// F0 below this is unvoiced.
pub const F0_MIN: f64 = 40.0;

/// Frame-rate harmonic excitation, `[B, n_mels + 2, T]` row-major, from
/// `f0` `[B, T]` (Hz, 0 = unvoiced):
/// - channels `0..n_mels`: `ln(0.01 + Σ_k fb[k, m] · exp(κ(cos(2π f_k / f0) − 1)))`
///   for voiced frames (`ln 0.01` otherwise), a von Mises comb peaked at
///   every harmonic of f0, width ≈ 1.5 bins (`κ = (f0 / (2π·1.5·Δf))²`),
///   projected onto the mel bands: the spectral envelope of a harmonic
///   source at frame rate, no sample-rate oscillator;
/// - channel `n_mels`: voicing 0/1; `n_mels + 1`: `ln f0 − 5` (0 unvoiced).
pub fn source_feats(f0: &[f32], batch: usize, frames: usize, sr: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let fb = mel_fbank(sr, n_fft, n_mels);
    let df = sr as f64 / n_fft as f64;
    let c = n_mels + 2;
    let mut out = vec![0.0f32; batch * c * frames];
    let mut comb = vec![0.0f64; bins];
    for b in 0..batch {
        for t in 0..frames {
            let f = f0[b * frames + t] as f64;
            let at = |ch: usize| (b * c + ch) * frames + t;
            if f < F0_MIN {
                for m in 0..n_mels {
                    out[at(m)] = 0.01f64.ln() as f32;
                }
                continue;
            }
            let kap = (f / (2.0 * PI * 1.5 * df)).powi(2);
            for (k, v) in comb.iter_mut().enumerate() {
                *v = (kap * ((2.0 * PI * k as f64 * df / f).cos() - 1.0)).exp();
            }
            for m in 0..n_mels {
                let e: f64 = (0..bins).map(|k| fb[k * n_mels + m] * comb[k]).sum();
                out[at(m)] = (e + 0.01).ln() as f32;
            }
            out[at(n_mels)] = 1.0;
            out[at(n_mels + 1)] = (f.ln() - 5.0) as f32;
        }
    }
    out
}

/// §29.5 harmonic source (vocoder-plan rung 1; NSF / Wavehax): the STFT of a
/// sample-rate sine excitation `Σ_h sin(h·φ(n))` at the harmonics of `f0`
/// (Hz per frame, 0 = unvoiced), on G's framing (Hann `n_fft`, hop `hop`,
/// "same": frame t is centred at sample `t·hop + hop/2`), for the lowest
/// `nb` bins. Out `[2·nb, frames]`: re rows, then im rows, scaled so a lone
/// harmonic on a bin has magnitude 1.
///
/// The STFT is analytic under local stationarity (f0 fixed across one
/// window): a harmonic with phase θ at the window centre puts
/// `(−1)^k·K(ω − ω_k)/2 · (sin θ, −cos θ)` on bin k, with `K` the Hann
/// window's (real, centred) transform, tabulated. That costs H×9 bins per
/// frame instead of an n_fft-point DFT, so the source adds nothing to TTFA.
/// The phase φ is accumulated on the host (the IR has no Scan, gap 16) by
/// the trapezoid rule between frame centres; `phi0` is φ at frame 0's
/// centre, and the returned vector holds φ at every frame centre, so a
/// streaming caller starts the next window from the phase it already
/// emitted (continuous across chunks).
pub fn harmonic_feats(f0: &[f32], frames: usize, sr: usize, n_fft: usize, hop: usize, nb: usize, phi0: f64) -> (Vec<f32>, Vec<f64>) {
    use std::sync::OnceLock;
    // K(Δ) for Δ ∈ [−R, R] bins at 1/S resolution, / (n_fft/2) so K(0) = 1
    const R: i64 = 4;
    const S: i64 = 32;
    static TAB: OnceLock<(usize, Vec<f64>)> = OnceLock::new();
    let (tn, tab) = TAB.get_or_init(|| {
        let w = hann(n_fft);
        let half = n_fft as f64 / 2.0;
        let t = (-R * S..=R * S)
            .map(|j| {
                let d = 2.0 * PI * (j as f64 / S as f64) / n_fft as f64;
                (0..n_fft).map(|n| w[n] * (d * (n as f64 - half)).cos()).sum::<f64>() / half
            })
            .collect();
        (n_fft, t)
    });
    assert_eq!(*tn, n_fft, "harmonic_feats: one n_fft per process");
    let kof = |d: f64| -> f64 {
        let x = (d * S as f64).clamp((-R * S) as f64, (R * S) as f64) + (R * S) as f64;
        let i = (x.floor() as usize).min(tab.len() - 2);
        let a = x - i as f64;
        tab[i] * (1.0 - a) + tab[i + 1] * a
    };
    let df = sr as f64 / n_fft as f64;
    let fmax = nb as f64 * df;
    let mut out = vec![0.0f32; 2 * nb * frames];
    let mut phis = Vec::with_capacity(frames);
    let fe = |t: usize| -> f64 {
        let f = f0[t] as f64;
        if f >= F0_MIN { f } else { 0.0 }
    };
    let mut phi = phi0.rem_euclid(2.0 * PI);
    for t in 0..frames {
        if t > 0 {
            phi = (phi + 2.0 * PI * hop as f64 * 0.5 * (fe(t - 1) + fe(t)) / sr as f64).rem_euclid(2.0 * PI);
        }
        phis.push(phi);
        let f = fe(t);
        if f == 0.0 {
            continue;
        }
        let mut h = 1usize;
        while (h as f64) * f < fmax + R as f64 * df {
            let hf = h as f64 * f / df;
            let th = (h as f64 * phi).rem_euclid(2.0 * PI);
            let (s, c) = th.sin_cos();
            let k0 = (hf.round() as i64 - R).max(0);
            let k1 = (hf.round() as i64 + R).min(nb as i64 - 1);
            for k in k0..=k1 {
                let a = kof(hf - k as f64) * if k % 2 == 0 { 1.0 } else { -1.0 };
                let k = k as usize;
                out[k * frames + t] += (a * s) as f32;
                out[(nb + k) * frames + t] -= (a * c) as f32;
            }
            h += 1;
        }
    }
    (out, phis)
}

/// The vocoder's whole "src" input `[B, n_mels + 2 + 2·hsrc, T]`:
/// `source_feats` rows, then (hsrc > 0) `harmonic_feats` rows, each item's
/// excitation starting at phase `phi0[b]` (0 when `phi0` is empty). Returns
/// the rows and φ at every frame centre `[B, T]` (empty when hsrc = 0).
#[allow(clippy::too_many_arguments)]
pub fn source_rows(f0: &[f32], batch: usize, frames: usize, sr: usize, n_fft: usize, hop: usize, n_mels: usize, hsrc: usize, phi0: &[f64]) -> (Vec<f32>, Vec<f64>) {
    let base = source_feats(f0, batch, frames, sr, n_fft, n_mels);
    if hsrc == 0 {
        return (base, vec![]);
    }
    let (c0, c) = (n_mels + 2, n_mels + 2 + 2 * hsrc);
    let mut out = vec![0.0f32; batch * c * frames];
    let mut phis = Vec::with_capacity(batch * frames);
    for b in 0..batch {
        out[b * c * frames..(b * c + c0) * frames].copy_from_slice(&base[b * c0 * frames..(b + 1) * c0 * frames]);
        let (h, ph) = harmonic_feats(&f0[b * frames..(b + 1) * frames], frames, sr, n_fft, hop, hsrc, phi0.get(b).copied().unwrap_or(0.0));
        out[(b * c + c0) * frames..(b + 1) * c * frames].copy_from_slice(&h);
        phis.extend(ph);
    }
    (out, phis)
}
