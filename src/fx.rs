//! Frame-level voice edits (the ribt paper): applied to the flow's
//! log-mel and the decoder's frame F0, i.e. to the vocoder's *inputs*, after
//! the acoustic model has planned the utterance.
//!
//! §28.3 found that token-pitch spread is ignored by the decoder (it re-predicts
//! the contour). Range, tremor and the slip catch therefore act on the frame
//! F0 contour here, and the mel's low (harmonic-resolving) bands are warped by
//! the same ratio so the harmonics in the mel agree with the vocoder's F0
//! source. Voice quality acts on the mel envelope:
//! - press (effort): more harmonic contrast below 2.5 kHz,
//! - breathy (fatigue, heat, relief): less harmonic contrast (valleys filled)
//!   and an aspiration floor above 1.5 kHz,
//! - tilt: dB per octave around 1 kHz (+ = brighter / pressed, − = darker).
//!
//! Also here: a procedural inhale (filtered turbulence noise), because the
//! LibriTTS-R restoration leaves almost no clean breaths to mine (§28.6).

/// Edits for one frame. `Default` is the identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameFx {
    /// F0 range around the utterance centre (1 = unchanged)
    pub range: f32,
    /// extra F0 offset, semitones (slip catch)
    pub f0_st: f32,
    /// tremor depth (semitones, dB) and rate (Hz)
    pub tremor_st: f32,
    pub tremor_db: f32,
    pub tremor_hz: f32,
    /// broadband gain, dB
    pub gain_db: f32,
    /// spectral tilt, dB per octave around 1 kHz
    pub tilt: f32,
    /// 0..1
    pub breathy: f32,
    /// 0..1
    pub press: f32,
}

impl Default for FrameFx {
    fn default() -> Self {
        Self { range: 1.0, f0_st: 0.0, tremor_st: 0.0, tremor_db: 0.0, tremor_hz: 6.0, gain_db: 0.0, tilt: 0.0, breathy: 0.0, press: 0.0 }
    }
}

impl FrameFx {
    pub fn is_identity(&self) -> bool {
        let d = Self::default();
        self.range == d.range && self.f0_st == 0.0 && self.tremor_st == 0.0 && self.tremor_db == 0.0 && self.gain_db == 0.0 && self.tilt == 0.0 && self.breathy == 0.0 && self.press == 0.0
    }
}

/// Centre frequency of each HTK mel band (as `dsp::mel_fbank`), Hz.
pub fn band_hz(sr: usize, n_mels: usize) -> Vec<f32> {
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let m1 = hz_to_mel(sr as f64 / 2.0);
    (0..n_mels).map(|m| mel_to_hz(m1 * (m + 1) as f64 / (n_mels + 1) as f64) as f32).collect()
}

const DB_LN: f32 = std::f32::consts::LN_10 / 20.0; // dB → ln magnitude

fn smooth_bands(col: &[f32], r: usize) -> Vec<f32> {
    let n = col.len();
    (0..n)
        .map(|i| {
            let (a, b) = (i.saturating_sub(r), (i + r + 1).min(n));
            col[a..b].iter().sum::<f32>() / (b - a) as f32
        })
        .collect()
}

/// Interpolate a mel column at frequency `f` (Hz) over band centres `hz`.
fn at_hz(col: &[f32], hz: &[f32], f: f32) -> f32 {
    if f <= hz[0] {
        return col[0];
    }
    let i = hz.partition_point(|&h| h < f).min(hz.len() - 1);
    let (h0, h1) = (hz[i - 1], hz[i]);
    let w = ((f - h0) / (h1 - h0)).clamp(0.0, 1.0);
    col[i - 1] * (1.0 - w) + col[i] * w
}

/// Apply per-frame edits in place. `mel` is `[C, t]` (row c at `c*t`), `f0`
/// `[t]` Hz (0 = unvoiced), `fx[f]` the edit of frame f (only `valid` frames
/// are touched), `abs0` the absolute frame index of column 0 (tremor phase is
/// absolute, so overlapping windows agree), `centre_st` the utterance F0
/// centre in semitones re 100 Hz, `hop_s` seconds per frame.
#[allow(clippy::too_many_arguments)]
pub fn apply(mel: &mut [f32], f0: &mut [f32], t: usize, valid: usize, fx: &[FrameFx], abs0: usize, centre_st: f32, hz: &[f32], hop_s: f64) {
    let c_m = hz.len();
    let mut col = vec![0.0f32; c_m];
    for f in 0..valid.min(fx.len()) {
        let e = &fx[f];
        if e.is_identity() {
            continue;
        }
        for c in 0..c_m {
            col[c] = mel[c * t + f];
        }
        let ph = (2.0 * std::f64::consts::PI * e.tremor_hz as f64 * (abs0 + f) as f64 * hop_s).sin() as f32;
        // F0 contour: range around the centre, offset, tremor; warp the low bands with it
        if f0[f] > 0.0 {
            let st = 12.0 * (f0[f] / 100.0).log2();
            let tgt = centre_st + (st - centre_st) * e.range + e.f0_st + e.tremor_st * ph;
            let r = 2f32.powf((tgt - st) / 12.0);
            if std::env::var("RIBT_FX_NOF0").is_err() {
                f0[f] *= r;
            }
            if (r - 1.0).abs() > 1e-4 && std::env::var("RIBT_FX_NOWARP").is_err() {
                let src = col.clone();
                for c in 0..c_m {
                    let w = ((2000.0 - hz[c]) / 1000.0).clamp(0.0, 1.0);
                    if w > 0.0 {
                        col[c] = (1.0 - w) * src[c] + w * at_hz(&src, hz, hz[c] / r);
                    }
                }
            }
        }
        // harmonic contrast: press sharpens, breathy fills the valleys
        if e.press > 0.0 || e.breathy > 0.0 {
            let sm = smooth_bands(&col, 2);
            for c in 0..c_m {
                if hz[c] < 3000.0 {
                    let d = col[c] - sm[c];
                    col[c] += (0.9 * e.press - 0.5 * e.breathy) * d;
                }
            }
        }
        // aspiration floor above 1.5 kHz: ln(exp(col) + a·exp(smoothed))
        if e.breathy > 0.0 {
            let sm = smooth_bands(&col, 4);
            let a = 0.35 * e.breathy;
            for c in 0..c_m {
                if hz[c] > 1500.0 {
                    let w = ((hz[c] - 1500.0) / 1500.0).min(1.0) * a;
                    col[c] = (col[c].exp() + w * (sm[c] - 0.7).exp()).ln();
                }
            }
        }
        let g = (e.gain_db + e.tremor_db * ph) * DB_LN;
        for c in 0..c_m {
            let oct = (hz[c].max(500.0) / 1000.0).log2();
            mel[c * t + f] = col[c] + g + e.tilt * oct * DB_LN;
        }
    }
}

/// Procedural inhale: turbulence noise through three resonances (open-mouth
/// /h/-like, rising slightly as the lungs fill), raised-cosine envelope with a
/// long rise. `depth` 0..1 sets level and length; `sharp` 0..1 makes it a
/// gasp (shorter, brighter, louder onset). Deterministic in `seed`.
pub fn inhale(sr: usize, dur_s: f32, depth: f32, sharp: f32, seed: u64) -> Vec<f32> {
    let n = (dur_s * sr as f32).max(1.0) as usize;
    let mut rng = crate::model::Rng::new(seed ^ 0x9e37_79b9);
    let res = [(650.0f32, 2.5f32, 0.7f32), (1500.0, 3.0, 1.0), (2700.0, 3.5, 0.6)];
    let mut st = [[0.0f32; 2]; 3];
    let mut hp = (0.0f32, 0.0f32);
    let mut out = vec![0.0f32; n];
    let peak_at = 0.65 - 0.4 * sharp;
    for (i, o) in out.iter_mut().enumerate() {
        let x = rng.normal() as f32;
        let p = i as f32 / n as f32;
        let mut y = 0.0;
        for (k, &(f, q, g)) in res.iter().enumerate() {
            let f = f * (1.0 + 0.12 * p + 0.25 * sharp);
            // 2-pole resonator (bandpass), coefficients per sample (slow sweep)
            let w = 2.0 * std::f32::consts::PI * f / sr as f32;
            let r = (-w / (2.0 * q)).exp();
            let (a1, a2) = (2.0 * r * w.cos(), -r * r);
            let v = (1.0 - r) * x + a1 * st[k][0] + a2 * st[k][1];
            st[k][1] = st[k][0];
            st[k][0] = v;
            y += g * v;
        }
        // a little direct broadband (high-passed) turbulence
        let h = x - hp.0 + 0.95 * hp.1;
        hp = (x, h);
        y += 0.08 * (0.5 + sharp) * h;
        let env = if p < peak_at { 0.5 - 0.5 * (std::f32::consts::PI * p / peak_at).cos() } else { 0.5 + 0.5 * (std::f32::consts::PI * (p - peak_at) / (1.0 - peak_at)).cos() };
        *o = y * env;
    }
    let rms = (out.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt().max(1e-9);
    // speech here is ~ -20 dBFS RMS: an inhale sits 14-26 dB below it
    let lvl = 10f32.powf((-46.0 + 12.0 * depth + 8.0 * sharp) / 20.0);
    out.iter_mut().for_each(|v| *v *= lvl / rms);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_exact() {
        let hz = band_hz(24000, 100);
        let mut mel: Vec<f32> = (0..100 * 8).map(|i| (i as f32 * 0.37).sin() - 3.0).collect();
        let m0 = mel.clone();
        let mut f0 = vec![200.0f32; 8];
        apply(&mut mel, &mut f0, 8, 8, &[FrameFx::default(); 8], 0, 12.0, &hz, 0.0107);
        assert_eq!(mel, m0);
        assert_eq!(f0, vec![200.0; 8]);
    }

    #[test]
    fn range_scales_contour() {
        let hz = band_hz(24000, 100);
        let mut mel = vec![-3.0f32; 100 * 2];
        let centre = 12.0 * (200.0f32 / 100.0).log2();
        let mut f0 = vec![400.0f32, 100.0];
        let fx = FrameFx { range: 0.5, ..Default::default() };
        apply(&mut mel, &mut f0, 2, 2, &[fx; 2], 0, centre, &hz, 0.0107);
        assert!((f0[0] - 200.0 * 2f32.sqrt()).abs() < 0.5 && (f0[1] - 200.0 / 2f32.sqrt()).abs() < 0.5, "{f0:?}");
    }

    #[test]
    fn tilt_and_gain() {
        let hz = band_hz(24000, 100);
        let mut mel = vec![-3.0f32; 100];
        let mut f0 = vec![0.0f32];
        let fx = FrameFx { tilt: 6.0, gain_db: 0.0, ..Default::default() };
        apply(&mut mel, &mut f0, 1, 1, &[fx], 0, 0.0, &hz, 0.0107);
        let i4k = hz.iter().position(|&h| h > 4000.0).unwrap();
        let db = (mel[i4k] + 3.0) / DB_LN;
        assert!((db - 6.0 * (hz[i4k] / 1000.0).log2()).abs() < 0.01, "{db}");
        assert!((mel[0] + 3.0 + 6.0 * DB_LN).abs() < 1e-5, "below 500 Hz the tilt holds its 500 Hz value");
    }

    #[test]
    fn inhale_level_and_shape() {
        let x = inhale(24000, 0.4, 0.5, 0.0, 1);
        assert_eq!(x.len(), 9600);
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        assert!((20.0 * rms.log10() + 40.0).abs() < 0.1);
        assert!(x[0].abs() < 1e-3 && x[x.len() - 1].abs() < 1e-2);
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
