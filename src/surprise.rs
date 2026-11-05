//! Multi-timescale surprise (the ribt paper): one predictor per
//! timescale on the body bus, and the band that is surprised picks the voice's
//! response — fast → reflex, mid → prosody/surprised mood, slow → worry drift.
//!
//! Each band uses the rule of `ipse_world::Clock::Surprise` (mtrssm.md): it
//! fires when this step's error exceeds `ratio` × a running average of the
//! *previous* steps' errors, with a `min_s` refractory. (No `max_s`: the world model refreshes its slow
//! state on a quiet stretch; a voice has nothing to say then). The
//! error here is model-free — |x − EMA(x)| per channel, the `|Δobs|` clock that
//! reached 1.7–1.8× chance on the real K1 traces — and the slot is the same
//! one the MTRSSM fast KL fills later.
//!
//! Efference: effort is self-caused (the robot commanded it), so the fast band
//! does not watch it; a gait start or a stitched seam is not a reflex. Tilt and
//! capture margin are what the world does to the body. A real efference copy
//! (predicted effort from the command) would let the fast band watch effort too.
//!
//! Habituation: the information in a recurring event falls with how often its
//! band has fired lately, `h = 1 / (1 + n)`, where `n` is the band's decayed
//! fire count (memory `ev_tau`). The fifth identical shove is less news than
//! the first; a quiet minute restores it.

/// Channels: tilt rate (rad/s), effort (W), capture margin (m).
pub const NCH: usize = 3;
pub const BANDS: [&str; 3] = ["fast", "mid", "slow"];

#[derive(Clone, Copy, Debug)]
pub struct BandCfg {
    /// the band sees the signal smoothed at its own scale, s (0 = raw)
    pub in_tau: f64,
    /// prediction EMA time constant, s
    pub p_tau: f64,
    /// running average of error, s
    pub e_tau: f64,
    /// an event opens when error > ratio × average
    pub ratio: f64,
    /// the event fires this long after it opens, at its peak, s
    pub integ_s: f64,
    /// no two fires closer than this, s (a burst is one event)
    pub min_s: f64,
    /// habituation memory, s
    pub ev_tau: f64,
    /// channels this band watches (tilt, effort, margin)
    pub ch: [bool; NCH],
}

pub const CFG: [BandCfg; 3] = [
    BandCfg { in_tau: 0.0, p_tau: 0.5, e_tau: 5.0, ratio: 3.0, integ_s: 0.06, min_s: 0.3, ev_tau: 30.0, ch: [true, false, true] },
    BandCfg { in_tau: 0.25, p_tau: 1.5, e_tau: 20.0, ratio: 3.0, integ_s: 0.5, min_s: 1.0, ev_tau: 120.0, ch: [true, true, true] },
    BandCfg { in_tau: 10.0, p_tau: 60.0, e_tau: 600.0, ratio: 2.5, integ_s: 5.0, min_s: 30.0, ev_tau: 900.0, ch: [true, true, true] },
];

#[derive(Clone, Copy, Debug, Default)]
struct Band {
    pred: [f64; NCH],
    inp: [f64; NCH],
    avg: [f64; NCH],
    since: f64,
    age: f64,
    /// open event: (time since open, peak s, h at open)
    open: Option<(f64, f64, f32)>,
    count: f64,
    seeded: bool,
}

/// One step of the tower.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Out {
    /// error / running average (max over channels)
    pub s: [f32; 3],
    /// the band's clock fired
    pub fired: [bool; 3],
    /// habituation factor at this step, 1 = novel
    pub h: [f32; 3],
    /// event amplitude 0..1 on a fire, else 0: min(1, peak s / 2·ratio) × h
    pub amp: [f32; 3],
}

#[derive(Clone, Debug, Default)]
pub struct Tower {
    b: [Band; 3],
}

/// just-noticeable error per channel (tilt rad/s, effort W, margin m): an error
/// is surprising relative to its usual size plus this floor
const EPS: [f64; NCH] = [0.1, 5.0, 0.02];

impl Tower {
    pub fn new() -> Self {
        Self::default()
    }

    /// Step by `dt` seconds with observation `x` (None = channel missing).
    pub fn step(&mut self, x: [Option<f64>; NCH], dt: f64) -> Out {
        let mut o = Out::default();
        for (k, (b, c)) in self.b.iter_mut().zip(CFG.iter()).enumerate() {
            b.count *= (-dt / c.ev_tau).exp();
            o.h[k] = (1.0 / (1.0 + b.count)) as f32;
            b.since += dt;
            b.age += dt;
            let mut s = 0.0f64;
            for i in 0..NCH {
                let Some(xi) = x[i].filter(|_| c.ch[i]) else { continue };
                if !b.seeded {
                    b.pred[i] = xi;
                    b.inp[i] = xi;
                    continue;
                }
                b.inp[i] += (xi - b.inp[i]) * if c.in_tau > 0.0 { (dt / c.in_tau).min(1.0) } else { 1.0 };
                let xi = b.inp[i];
                let err = (xi - b.pred[i]).abs();
                // the average is of previous steps: a step cannot surprise itself
                if b.avg[i] > 0.0 {
                    s = s.max(err / (b.avg[i] + EPS[i]));
                }
                let ae = (dt / c.e_tau).min(1.0);
                b.avg[i] = if b.avg[i] == 0.0 { err.max(1e-9) } else { b.avg[i] + (err - b.avg[i]) * ae };
                b.pred[i] += (xi - b.pred[i]) * (dt / c.p_tau).min(1.0);
            }
            b.seeded = true;
            o.s[k] = s as f32;
            // warm-up: the predictor has converged and the average has seen min_s
            if let Some((age, peak, h)) = b.open.as_mut() {
                *age += dt;
                *peak = peak.max(s);
                if *age >= c.integ_s {
                    o.fired[k] = true;
                    o.amp[k] = (*peak / (2.0 * c.ratio)).min(1.0) as f32 * *h;
                    b.open = None;
                }
            } else if b.age >= 3.0 * c.p_tau + c.min_s && b.since >= c.min_s && s > c.ratio {
                // habituation counts the onset: a long event is one event
                b.open = Some((0.0, s, o.h[k]));
                b.count += 1.0;
                b.since = 0.0;
            }
        }
        o
    }
}

pub fn obs(s: &crate::body::BodyState) -> [Option<f64>; NCH] {
    [s.tilt_rate.map(f64::from), s.effort_w.map(f64::from), s.margin_m.map(f64::from)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// steady walking-like tilt noise with shove spikes at `shoves` (s)
    fn run(shoves: &[f64], secs: f64) -> Vec<(f64, Out)> {
        let mut t = Tower::new();
        let dt = 0.02;
        (0..(secs / dt) as usize)
            .map(|k| {
                let tt = k as f64 * dt;
                let gait = 0.42 + 0.05 * (tt * 2.0 * std::f64::consts::PI * 1.6).sin();
                let spike: f64 = shoves.iter().map(|&s| if tt >= s && tt < s + 0.2 { 0.9 } else { 0.0 }).sum();
                (tt, t.step([Some(gait + spike), Some(20.0), None], dt))
            })
            .collect()
    }

    fn fast_amp_near(r: &[(f64, Out)], at: f64) -> f32 {
        r.iter().filter(|(t, _)| (*t - at).abs() < 0.3).map(|(_, o)| o.amp[0]).fold(0.0, f32::max)
    }

    #[test]
    fn steady_gait_is_quiet() {
        let r = run(&[], 60.0);
        assert!(r.iter().skip(250).all(|(_, o)| !o.fired[0]), "fast band fired on plain gait");
    }

    #[test]
    fn shove_fires_fast_and_habituates() {
        let shoves: Vec<f64> = (0..5).map(|i| 20.0 + 4.0 * i as f64).collect();
        let r = run(&shoves, 60.0);
        let a: Vec<f32> = shoves.iter().map(|&s| fast_amp_near(&r, s)).collect();
        assert!(a[0] > 0.5, "first shove {a:?}");
        assert!(a[4] < 0.5 * a[0], "no habituation {a:?}");
        // a quiet stretch restores novelty
        let r2 = run(&[20.0, 24.0, 28.0, 32.0, 36.0, 136.0], 140.0);
        assert!(fast_amp_near(&r2, 136.0) > 2.0 * fast_amp_near(&r2, 36.0));
    }
}
