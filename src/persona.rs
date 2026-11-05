//! Body voice v1 (the ribt paper): body dynamics, mood,
//! reflexes and the style that turns them into voice edits.
//!
//! A [`Timeline`] is built from a [`BodyTrack`] by stepping the same dynamics
//! a live robot would step at 50 Hz:
//! - body states with time constants: effort builds up (1.5 s) and recovers
//!   slowly (4 s); breathlessness integrates sustained effort and stays a few
//!   seconds after stopping (hysteresis); heat and fatigue are slow;
//! - slips (tilt or margin crossing, or a bus `slip` event) leave a decaying
//!   tremor and a pitch catch;
//! - reflexes: slip → "whoa", push → "hup", impact → "oof", fall → "eep",
//!   a small slip → a gasp; played by the caller through the barge machinery;
//! - mood (appraisal): surprised, relieved, proud, weary, curious, happy,
//!   worried, each an intensity in [0, 1] with its own decay; slips, recovery,
//!   steady footfalls, low battery, heat and interaction cues push them.
//!
//! [`Timeline::style`] maps one instant to token-level edits (pitch offset,
//! energy, pace, pauses: the acoustic model renders them) and frame-level
//! edits (range, catch, tremor, tilt, press, breathy: [`crate::fx`]), all
//! scaled by one `drama` knob (0 = plain voice, 1 = designed, 2 = theatrical).

use crate::body::{BodyState, BodyTrack, Event, VoiceMod};
use crate::fx::FrameFx;

pub const DT: f64 = 0.02;
pub const MOODS: [&str; 7] = ["surprised", "relieved", "proud", "weary", "curious", "happy", "worried"];
const TAU: [f64; 7] = [1.5, 4.0, 10.0, 12.0, 6.0, 8.0, 5.0];

/// Per-mood style row: pitch st, range ×, pace ×, energy dB, tilt dB/oct,
/// breathy, press, extra clause pause s, tremor st.
const STYLE: [[f32; 9]; 7] = [
    [3.5, 1.7, 0.90, 3.0, 1.5, 0.15, 0.3, 0.0, 0.0],   // surprised
    [-1.0, 1.15, 1.10, -1.5, -1.5, 0.55, 0.0, 0.25, 0.0], // relieved (a sigh)
    [1.2, 1.3, 0.97, 2.5, 1.5, 0.0, 0.45, 0.0, 0.0],   // proud
    [-2.5, 0.5, 1.30, -4.0, -3.0, 0.4, 0.0, 0.3, 0.0], // weary
    [2.2, 1.45, 0.97, 0.5, 0.5, 0.1, 0.0, 0.0, 0.0],   // curious
    [2.2, 1.5, 0.94, 2.5, 1.0, 0.0, 0.25, 0.0, 0.0],   // happy
    [0.8, 0.7, 1.07, -1.5, 0.0, 0.35, 0.1, 0.15, 0.35], // worried
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reflex {
    Whoa,
    Hup,
    Oof,
    Eep,
    Gasp,
}

impl Reflex {
    /// The spoken word. With a Midwest accent on (`--accent midwest…`, §26.8)
    /// the Minnesotan set: big slip "Oh geez!", push / small slip "Ope!",
    /// fall "Oh jeez, oh no!"; impact stays "Oof.".
    pub fn text(self) -> &'static str {
        if crate::lilt::accent::midwest() {
            match self {
                Self::Whoa => return "Oh geez!",
                Self::Hup => return "Ope!",
                Self::Eep => return "Oh jeez, oh no!",
                _ => {}
            }
        }
        match self {
            Self::Whoa => "Whoa!",
            Self::Hup => "Hup!",
            Self::Oof => "Oof.",
            Self::Eep => "Eep!",
            Self::Gasp => "",
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Whoa => "whoa",
            Self::Hup => "hup",
            Self::Oof => "oof",
            Self::Eep => "eep",
            Self::Gasp => "gasp",
        }
    }
    pub const ALL: [Reflex; 5] = [Self::Whoa, Self::Hup, Self::Oof, Self::Eep, Self::Gasp];
}

/// A near-threshold surprise-tower fire: a gasp, or "Ope!" (a [`Reflex::Hup`],
/// which cuts and resumes) for a Midwest accent when it is at least 0.2. The
/// raw-slip gasp stays a gasp: it is the rising edge of a big slip ~20 ms
/// before its "Oh geez!", so a word there would be cut at once.
fn small_slip(a: f32) -> Reflex {
    if crate::lilt::accent::midwest() && a >= 0.2 {
        Reflex::Hup
    } else {
        Reflex::Gasp
    }
}

/// The state at one instant.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    pub t: f64,
    pub raw: BodyState,
    pub effort: f32,
    pub heat: f32,
    pub fatigue: f32,
    pub breathless: f32,
    pub slip: f32,
    /// frame-level pitch catch, st (decaying after each slip onset)
    pub catch_st: f32,
    pub mood: [f32; 7],
    pub fallen: bool,
    /// multi-timescale surprise at this instant (zeros unless built with it)
    pub surprise: crate::surprise::Out,
}

#[derive(Clone, Debug, Default)]
pub struct Timeline {
    pub t0: f64,
    pub s: Vec<Sample>,
    pub reflexes: Vec<(f64, Reflex, f32)>,
    pub footfalls: Vec<f64>,
    pub slips: Vec<(f64, f32)>,
    /// mood fixed for every sample (the emotion set)
    pub forced: Option<[f32; 7]>,
}

/// Combined edit for one instant.
#[derive(Clone, Copy, Debug)]
pub struct Style {
    pub tok: VoiceMod,
    pub fx: FrameFx,
    /// 0..1: how much breath the body needs (sets breath depth and frequency)
    pub need: f32,
    /// breathe at every clause boundary
    pub clause_breath: bool,
    /// extra pause at clause boundaries, s
    pub clause_pause: f32,
}

impl Timeline {
    /// Step the dynamics over the track at 50 Hz. `prime_s`: run the first
    /// sample for this long before t0 (a body that has been in that state a while).
    pub fn build(tr: &BodyTrack, prime_s: f64) -> Self {
        Self::build_with(tr, prime_s, false)
    }

    /// `surprise`: the reflexes and the surprised/worried moods come from the
    /// multi-timescale surprise tower (§28.12) instead of fixed slip
    /// thresholds: the fast band's fire is the reflex (amplitude = how
    /// surprising × how novel), the mid band feeds `surprised`, the slow band
    /// `worried`. Bus events (push, impact, fall, cues) are unchanged.
    pub fn build_with(tr: &BodyTrack, prime_s: f64, surprise: bool) -> Self {
        let t0 = tr.t.first().copied().unwrap_or(0.0);
        let t1 = tr.t.last().copied().unwrap_or(0.0) + 30.0;
        let mut tl = Timeline { t0, ..Default::default() };
        let (mut eff, mut heat, mut fat, mut br, mut slip) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut mood = [0.0f32; 7];
        let (mut prev_slip_raw, mut prev_phase) = (0.0f32, None::<f32>);
        let mut last_slip = f64::NEG_INFINITY;
        let mut last_gasp = f64::NEG_INFINITY;
        let mut relief_due: Option<(f64, f32)> = None;
        let mut calm_since = f64::NEG_INFINITY;
        let mut ev_idx = 0usize;
        let mut tower = crate::surprise::Tower::new();
        let n_prime = (prime_s / DT) as usize;
        let n = n_prime + ((t1 - t0) / DT) as usize;
        for k in 0..n {
            let priming = k < n_prime;
            let t = if priming { t0 } else { t0 + (k - n_prime) as f64 * DT };
            let raw = tr.at(t);
            let v = raw.vec();
            let dt = DT as f32;
            // body dynamics
            let te = if v.effort > eff { 1.5 } else { 4.0 };
            eff += (v.effort - eff) * dt / te;
            heat += (v.heat - heat) * dt / 5.0;
            fat += (v.fatigue - fat) * dt / 10.0;
            let bt = ((eff - 0.2) * 1.5).clamp(0.0, 1.0);
            br += (bt - br) * dt / if bt > br { 8.0 } else { 5.0 };
            slip = slip.max(v.slip) - slip * dt / 1.0;
            slip = slip.max(0.0);
            let so = tower.step(crate::surprise::obs(&raw), DT);
            // events (not while priming)
            let mut events: Vec<Event> = Vec::new();
            if !priming && surprise {
                let a = so.amp[0];
                if a >= 0.35 {
                    last_slip = t;
                    tl.slips.push((t, a));
                    tl.reflexes.push((t, Reflex::Whoa, a));
                    slip = slip.max(a);
                    mood[6] = (mood[6] + 0.3 * a).min(1.0);
                    relief_due = Some((t, mood[0].max(a)));
                } else if a > 0.1 && t - last_gasp >= 3.0 {
                    tl.reflexes.push((t, small_slip(a), a));
                    last_gasp = t;
                }
                mood[0] = mood[0].max(0.9 * a).max(0.8 * so.amp[1]);
                mood[6] = mood[6].max(0.7 * so.amp[2]);
            }
            if !priming {
                while ev_idx < tr.t.len() && tr.t[ev_idx] <= t {
                    if let Some(e) = tr.s[ev_idx].event {
                        events.push(e);
                    }
                    ev_idx += 1;
                }
                if surprise {
                } else if v.slip >= crate::body::SLIP_ON && prev_slip_raw < crate::body::SLIP_ON && t - last_slip >= crate::body::SLIP_REFRACTORY_S {
                    events.push(Event::Slip);
                } else if v.slip >= 0.35 && prev_slip_raw < 0.35 && t - last_slip >= crate::body::SLIP_REFRACTORY_S && t - last_gasp >= 3.0 {
                    tl.reflexes.push((t, Reflex::Gasp, v.slip));
                    last_gasp = t;
                }
                // impacts come only from a bus `impact` event: the logs have no
                // accelerometer, and an effort step is also what a stitched seam looks like
                if let (Some(a), Some(b)) = (prev_phase, raw.phase) {
                    if (b * 2.0).floor() != (a * 2.0).floor() {
                        tl.footfalls.push(t);
                        if eff > 0.1 && slip < 0.1 {
                            mood[2] = (mood[2] + 0.03).min(0.8);
                        }
                    }
                }
                if raw.fallen && !tl.s.last().is_some_and(|s| s.fallen) {
                    events.push(Event::Fall);
                }
            }
            for e in events {
                match e {
                    Event::Slip => {
                        let st = v.slip.max(0.6);
                        last_slip = t;
                        tl.slips.push((t, st));
                        tl.reflexes.push((t, Reflex::Whoa, st));
                        slip = slip.max(st);
                        mood[0] = mood[0].max(0.9);
                        mood[6] = (mood[6] + 0.3).min(1.0);
                        relief_due = Some((t, mood[0]));
                    }
                    Event::Push => {
                        tl.reflexes.push((t, Reflex::Hup, 0.8));
                        mood[0] = mood[0].max(0.7);
                        relief_due = Some((t, 0.7));
                    }
                    Event::Impact => {
                        tl.reflexes.push((t, Reflex::Oof, 0.8));
                        mood[0] = mood[0].max(0.6);
                        mood[6] = (mood[6] + 0.3).min(1.0);
                    }
                    Event::Fall => {
                        tl.reflexes.push((t, Reflex::Eep, 1.0));
                        mood[0] = 1.0;
                        mood[6] = 1.0;
                        relief_due = None;
                    }
                    Event::Greet => mood[5] = mood[5].max(0.8),
                    Event::Question | Event::Novel => mood[4] = mood[4].max(0.8),
                    Event::Praise => {
                        mood[5] = mood[5].max(0.6);
                        mood[2] = mood[2].max(0.7);
                    }
                }
            }
            // relief: the slip/push is over and the body is calm again for 1 s
            if v.slip < 0.2 && !raw.fallen {
                if calm_since == f64::NEG_INFINITY {
                    calm_since = t;
                }
            } else {
                calm_since = f64::NEG_INFINITY;
            }
            if let Some((te, amp)) = relief_due {
                if t - te > 0.5 && calm_since > f64::NEG_INFINITY && t - calm_since >= 1.0 {
                    mood[1] = mood[1].max(0.85 * amp.max(0.6));
                    relief_due = None;
                }
            }
            // slow appraisals and decay
            mood[3] += (fat.max(0.7 * br * heat.max(0.5)) - mood[3]) * dt / TAU[3] as f32;
            let worry = 0.6 * heat + 0.4 * slip;
            if worry > mood[6] {
                mood[6] += (worry - mood[6]) * dt / 3.0;
            }
            for i in [0, 1, 2, 4, 5, 6] {
                mood[i] -= mood[i] * dt / TAU[i] as f32;
            }
            prev_slip_raw = v.slip;
            prev_phase = raw.phase;
            if priming {
                continue;
            }
            let catch_st = tl.slips.iter().filter(|s| s.0 <= t).map(|s| 6.0 * s.1 * (-((t - s.0) / 0.25) as f32).exp()).sum();
            tl.s.push(Sample { t, raw, effort: eff, heat, fatigue: fat, breathless: br, slip, catch_st, mood, fallen: raw.fallen, surprise: if surprise { so } else { Default::default() } });
        }
        tl
    }

    pub fn force_mood(&mut self, m: [f32; 7]) {
        self.forced = Some(m);
        self.s.iter_mut().for_each(|s| s.mood = m);
    }

    pub fn at(&self, t: f64) -> Sample {
        if self.s.is_empty() {
            return Sample::default();
        }
        let i = (((t - self.t0) / DT).round().max(0.0) as usize).min(self.s.len() - 1);
        self.s[i]
    }

    /// The edit at time t, scaled by `drama`.
    pub fn style(&self, t: f64, drama: f32) -> Style {
        style_of(&self.at(t), drama)
    }
}

/// Map a sample to voice edits (hand-designed, §28.6). Each body term is
/// affine in its state; mood rows add with their intensities; `drama` scales
/// every delta (0 = identity).
pub fn style_of(s: &Sample, drama: f32) -> Style {
    let (e, h, f, b, sl) = (s.effort, s.heat, s.fatigue, s.breathless, s.slip);
    let mut pitch = 2.5 * e - 1.5 * h - 2.0 * f;
    let mut range = 0.25 * e - 0.4 * h - 0.5 * f; // delta
    let mut lpace = ((1.0 - 0.07 * e) * (1.0 + 0.25 * h) * (1.0 + 0.25 * f) * (1.0 + 0.1 * b)).ln();
    let mut energy = 5.0 * e - 3.0 * h - 5.0 * f - 1.0 * b;
    let mut tilt = 4.5 * e - 2.5 * h - 3.0 * f - 0.5 * b;
    let mut breathy = 0.35 * h + 0.4 * f + 0.3 * b;
    let mut press = 1.0 * e * (1.0 - 0.3 * b);
    let mut tremor = 0.8 * sl;
    let mut clause_pause = 0.0f32;
    for (i, w) in s.mood.iter().enumerate() {
        let r = &STYLE[i];
        pitch += w * r[0];
        range += w * (r[1] - 1.0);
        lpace += w * r[2].ln();
        energy += w * r[3];
        tilt += w * r[4];
        breathy += w * r[5];
        press += w * r[6];
        clause_pause += w * r[7];
        tremor += w * r[8];
    }
    let d = drama;
    // soft limits keep drama 2 theatrical but still the same speaker (§28.9: likeness)
    let sat = |x: f32, m: f32| m * (x / m).tanh();
    let need = ((0.8 * b).max(0.5 * e) * d.min(1.5)).min(1.0);
    let phrase_words = if need > 0.15 { (14.0 - 10.0 * need).round().max(3.0) as usize } else { 0 };
    Style {
        tok: VoiceMod {
            pitch_st: sat(pitch * d, 6.0),
            range: 1.0,
            energy_db: sat(energy * d, 10.0),
            pace: (lpace * d).exp().clamp(0.75, 1.6),
            phrase_words,
            pause_s: if phrase_words > 0 { 0.2 + 0.35 * need } else { 0.0 },
        },
        fx: FrameFx {
            range: (1.0 + range * d).clamp(0.35, 3.0),
            f0_st: s.catch_st * d,
            tremor_st: tremor * d,
            tremor_db: 2.5 * sl * d,
            tremor_hz: 6.5,
            gain_db: 0.0,
            tilt: sat(tilt * d, 8.0),
            breathy: (breathy * d).clamp(0.0, 0.8),
            press: (press * d).clamp(0.0, 1.0),
        },
        need,
        clause_breath: need > 0.25,
        clause_pause: clause_pause * d,
    }
}

/// One-hot mood by name.
pub fn mood_named(name: &str, w: f32) -> Option<[f32; 7]> {
    let i = MOODS.iter().position(|m| *m == name)?;
    let mut m = [0.0; 7];
    m[i] = w;
    Some(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::preset;

    #[test]
    fn rest_is_identity_at_any_drama() {
        let tl = Timeline::build(&BodyTrack::constant(BodyState::default()), 30.0);
        for d in [0.0, 1.0, 2.0] {
            let s = tl.style(3.0, d);
            assert_eq!(s.tok, VoiceMod::IDENTITY);
            assert!(s.fx.is_identity());
        }
    }

    #[test]
    fn drama_zero_is_identity() {
        let tl = Timeline::build(&BodyTrack::constant(preset("walking").unwrap()), 30.0);
        let s = tl.style(1.0, 0.0);
        assert!(s.fx.is_identity() && s.tok.pitch_st == 0.0 && s.tok.pace == 1.0);
    }

    #[test]
    fn breathless_builds_and_lingers() {
        let walk = preset("walking").unwrap();
        let rest = preset("rest").unwrap();
        let tr = BodyTrack { t: vec![0.0, 20.0], s: vec![walk, rest] };
        let tl = Timeline::build(&tr, 0.0);
        let b = |t| tl.at(t).breathless;
        assert!(b(2.0) < b(10.0) && b(10.0) < b(19.9), "builds");
        assert!(b(23.0) > 0.4 * b(19.9), "lingers after stopping: {} vs {}", b(23.0), b(19.9));
        assert!(b(45.0) < 0.1 * b(19.9), "recovers");
    }

    #[test]
    fn slip_surprise_then_relief_and_whoa() {
        let calm = BodyState { tilt_rate: Some(0.1), ..Default::default() };
        let slip = BodyState { tilt_rate: Some(1.6), ..Default::default() };
        let tr = BodyTrack { t: vec![0.0, 2.0, 2.3], s: vec![calm, slip, calm] };
        let tl = Timeline::build(&tr, 0.0);
        assert!(tl.reflexes.iter().any(|r| r.1 == Reflex::Whoa && (r.0 - 2.0).abs() < 0.05));
        assert!(tl.at(2.1).mood[0] > 0.8);
        assert!(tl.at(2.1).catch_st > 3.0);
        assert!(tl.at(4.0).mood[1] > 0.5, "relieved after recovery");
        assert!(tl.at(4.0).mood[0] < 0.4, "surprise decays");
    }

    #[test]
    fn low_battery_weary() {
        let tl = Timeline::build(&BodyTrack::constant(preset("lowbatt").unwrap()), 60.0);
        assert!(tl.at(1.0).mood[3] > 0.6);
        let s = tl.style(1.0, 1.0);
        assert!(s.tok.pitch_st < -3.0 && s.fx.breathy > 0.4 && s.fx.range < 0.7);
    }
}
