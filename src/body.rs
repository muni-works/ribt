//! Body voice (the ribt paper): robot state shapes the voice.
//!
//! The bus fields of docs/k1-stack.md §3 become a small normalized body
//! vector, and a hand-designed, documented map turns it into edits of the
//! acoustic model's *token-level* conditioning (pitch, pitch range, energy,
//! duration, breath pauses). The edits act before the decoder, flow and
//! vocoder, so the voice is re-planned, not post-processed: §11.6's lesson
//! (knobs belong in the acoustic model, not the vocoder).
//!
//! Every field is `Option`: missing is explicit, never zero (bus rule), and a
//! missing field contributes its rest value. Rest maps to the identity edit,
//! so a robot with no bus speaks exactly as without it.

/// Raw bus fields (k1-stack §3 units).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyState {
    /// mechanical effort Σ|τ_est·dq| over the serial joints, W (1 s EMA)
    pub effort_w: Option<f32>,
    /// hottest parallel-array motor temperature, °C (serial reads 0)
    pub heat_c: Option<f32>,
    /// capture margin from `fall::stoppability`, m (≤ 0 = cannot stop)
    pub margin_m: Option<f32>,
    /// trunk tilt rate |d(roll, pitch)/dt|, rad/s: the slip proxy when no margin exists
    pub tilt_rate: Option<f32>,
    /// battery state of charge (vendor coulomb counter), 0..1
    pub soc: Option<f32>,
    /// fall flag / posture abort: speech must stop
    pub fallen: bool,
    /// gait phase, cycles (footfalls at integer and half-integer phase)
    pub phase: Option<f32>,
    /// discrete bus event or interaction cue at this sample (§28.7)
    pub event: Option<Event>,
}

/// Discrete events: bus reflex triggers and interaction cues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Push,
    Impact,
    Fall,
    Slip,
    Greet,
    Question,
    Novel,
    Praise,
}

impl Event {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "push" => Self::Push,
            "impact" => Self::Impact,
            "fall" => Self::Fall,
            "slip" => Self::Slip,
            "greet" => Self::Greet,
            "question" => Self::Question,
            "novel" => Self::Novel,
            "praise" => Self::Praise,
            _ => return None,
        })
    }
}

/// Normalized body vector, each in [0, 1]; 0 = rest.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyVec {
    pub effort: f32,
    pub heat: f32,
    pub slip: f32,
    pub fatigue: f32,
    pub fallen: bool,
}

/// Normalization constants (§28.2).
pub const EFFORT_FULL_W: f32 = 100.0; // ours walking on ground: median 25 W, p95 76 W
pub const HEAT_REST_C: f32 = 40.0; // gantry/ankle walks: 29–46 °C
pub const HEAT_FULL_C: f32 = 65.0; // on-ground hip roll reached 61 °C in 3 min
pub const SLIP_TILT0: f32 = 0.7; // rad/s; our on-ground walk sways up to 0.66 (p99 0.59)
pub const SLIP_TILT1: f32 = 1.5; // kd-1 ankle-shake walk peaks at 1.40
pub const SLIP_MARGIN0: f32 = 0.05; // m; capture margin below this starts to count
pub const SOC_OK: f32 = 0.5;
pub const SOC_EMPTY: f32 = 0.15;

fn ramp(x: f32, a: f32, b: f32) -> f32 {
    ((x - a) / (b - a)).clamp(0.0, 1.0)
}

impl BodyState {
    pub fn vec(&self) -> BodyVec {
        let slip_m = self.margin_m.map_or(0.0, |m| ramp(-m, -SLIP_MARGIN0, 0.05));
        let slip_t = self
            .tilt_rate
            .map_or(0.0, |w| ramp(w, SLIP_TILT0, SLIP_TILT1));
        BodyVec {
            effort: self.effort_w.map_or(0.0, |p| ramp(p, 0.0, EFFORT_FULL_W)),
            heat: self
                .heat_c
                .map_or(0.0, |t| ramp(t, HEAT_REST_C, HEAT_FULL_C)),
            slip: slip_m.max(slip_t),
            fatigue: self
                .soc
                .map_or(0.0, |s| ramp(SOC_OK - s, 0.0, SOC_OK - SOC_EMPTY)),
            fallen: self.fallen,
        }
    }
}

/// Edit of the token conditioning for one stretch of speech.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceMod {
    /// global pitch offset, semitones
    pub pitch_st: f32,
    /// pitch range around the utterance centre (1 = unchanged)
    pub range: f32,
    /// token energy offset, dB of power
    pub energy_db: f32,
    /// duration scale (> 1 = slower)
    pub pace: f32,
    /// insert a breath pause after this many words without one (0 = never)
    pub phrase_words: usize,
    /// breath pause length, s
    pub pause_s: f32,
}

impl VoiceMod {
    pub const IDENTITY: VoiceMod = VoiceMod {
        pitch_st: 0.0,
        range: 1.0,
        energy_db: 0.0,
        pace: 1.0,
        phrase_words: 0,
        pause_s: 0.0,
    };
}

/// The v0 map (hand-designed, §28.2). Each knob is affine in the body vector,
/// so it is monotone and smooth by construction:
/// - effort ↑ → higher pitch and energy (more subglottal pressure), shorter
///   phrases split by breath pauses, slightly faster;
/// - heat ↑ → slower, quieter, lower and flatter (strain);
/// - fatigue (low battery) ↑ → quieter, lower, slower, flatter;
/// - slip is an *event*, not a level: see [`SlipEvent`].
pub fn map_v0(b: &BodyVec) -> VoiceMod {
    let (e, h, f) = (b.effort, b.heat, b.fatigue);
    let phrase_words = if e > 0.2 {
        (12.0 - 8.0 * e).round().max(3.0) as usize
    } else {
        0
    };
    VoiceMod {
        pitch_st: 1.5 * e - 1.0 * h - 1.5 * f,
        range: (1.0 + 0.15 * e - 0.3 * h - 0.4 * f).max(0.3),
        energy_db: 3.0 * e - 2.0 * h - 4.0 * f,
        pace: (1.0 - 0.06 * e) * (1.0 + 0.15 * h) * (1.0 + 0.15 * f),
        phrase_words,
        pause_s: if phrase_words > 0 {
            0.12 + 0.15 * e
        } else {
            0.0
        },
    }
}

/// A balance slip: a pitch catch (a jump decaying over `tokens`) and a pause
/// at the next word boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlipEvent {
    pub pitch_st: f32,
    pub tokens: usize,
    pub pause_s: f32,
}

pub const SLIP_ON: f32 = 0.5;
/// energy offset of a breath-pause token, dB (the stretched space token otherwise murmurs)
pub const PAUSE_DB: f32 = -25.0;
pub const SLIP_REFRACTORY_S: f64 = 1.5;

pub fn slip_v0(slip: f32) -> SlipEvent {
    SlipEvent {
        pitch_st: 2.0 + 3.0 * slip,
        tokens: 4,
        pause_s: 0.2 + 0.2 * slip,
    }
}

/// A body signal over time: (t seconds, state) samples, sample-and-hold.
#[derive(Clone, Debug, Default)]
pub struct BodyTrack {
    pub t: Vec<f64>,
    pub s: Vec<BodyState>,
}

impl BodyTrack {
    pub fn constant(s: BodyState) -> Self {
        Self {
            t: vec![0.0],
            s: vec![s],
        }
    }
    pub fn at(&self, t: f64) -> BodyState {
        if self.t.is_empty() {
            return BodyState::default();
        }
        let i = self.t.partition_point(|&x| x <= t).saturating_sub(1);
        self.s[i]
    }
    /// Slip onsets: vec().slip crossing SLIP_ON upward, with a refractory time.
    pub fn slips(&self) -> Vec<(f64, f32)> {
        let mut out: Vec<(f64, f32)> = Vec::new();
        let mut prev = 0.0f32;
        for (t, s) in self.t.iter().zip(&self.s) {
            let v = s.vec().slip;
            if v >= SLIP_ON
                && prev < SLIP_ON
                && out.last().is_none_or(|l| t - l.0 >= SLIP_REFRACTORY_S)
            {
                out.push((*t, v));
            }
            prev = v;
        }
        out
    }
    /// Load `scripts/body_from_trace.py` output: a tsv whose header names
    /// t, effort_w, heat_c, margin_m, tilt_rate, soc, fallen ("nan" = missing).
    pub fn load_tsv(text: &str) -> Self {
        let mut lines = text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
        let head: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
        let col = |n: &str| head.iter().position(|h| *h == n);
        let (ct, ce, ch, cm, cr, cs, cf) = (
            col("t"),
            col("effort_w"),
            col("heat_c"),
            col("margin_m"),
            col("tilt_rate"),
            col("soc"),
            col("fallen"),
        );
        let (cp, cv) = (col("phase"), col("event"));
        let mut tr = Self::default();
        for l in lines {
            let v: Vec<&str> = l.split('\t').collect();
            let g = |c: Option<usize>| {
                c.and_then(|i| v.get(i))
                    .and_then(|x| x.parse::<f32>().ok())
                    .filter(|x| x.is_finite())
            };
            let Some(t) = g(ct) else { continue };
            let ev = cv.and_then(|i| v.get(i)).and_then(|x| Event::parse(x.trim()));
            tr.t.push(t as f64);
            tr.s.push(BodyState {
                effort_w: g(ce),
                heat_c: g(ch),
                margin_m: g(cm),
                tilt_rate: g(cr),
                soc: g(cs),
                fallen: g(cf).is_some_and(|x| x > 0.5),
                phase: g(cp),
                event: ev,
            });
        }
        tr
    }
}

/// Named demo states (§28.4).
pub fn preset(name: &str) -> Option<BodyState> {
    let rest = BodyState {
        effort_w: Some(0.2),
        heat_c: Some(33.0),
        margin_m: None,
        tilt_rate: Some(0.05),
        soc: Some(0.9),
        fallen: false,
        phase: None,
        event: None,
    };
    Some(match name {
        "rest" => rest,
        "walking" => BodyState {
            effort_w: Some(60.0),
            heat_c: Some(36.0),
            tilt_rate: Some(0.2),
            ..rest
        },
        "hot" => BodyState {
            effort_w: Some(20.0),
            heat_c: Some(61.0),
            ..rest
        },
        "slipping" => BodyState {
            effort_w: Some(80.0),
            tilt_rate: Some(1.6),
            ..rest
        },
        "lowbatt" => BodyState {
            soc: Some(0.17),
            ..rest
        },
        _ => return None,
    })
}

/// Apply a [`VoiceMod`] to one committed token in place. `pp` is normalized
/// log-F0 ((ln F0 − μ)/σ), `ep` normalized log-energy, `dur` frames; `f0_sd`
/// and `en_sd` are the normalizer σ's; `centre` is the pitch the range scales
/// around (the utterance's mean so far).
pub fn apply_token(
    m: &VoiceMod,
    pp: &mut f32,
    ep: &mut f32,
    dur: &mut u32,
    centre: f32,
    f0_sd: f64,
    en_sd: f64,
) {
    if *m == VoiceMod::IDENTITY {
        return;
    }
    let dst = (m.pitch_st as f64 * std::f64::consts::LN_2 / 12.0 / f0_sd) as f32;
    *pp = centre + (*pp - centre) * m.range + dst;
    *ep += (m.energy_db as f64 * std::f64::consts::LN_10 / 10.0 / en_sd) as f32;
    *dur = (*dur as f32 * m.pace).round() as u32;
}

/// Pitch offset (normalized units) of a slip catch on the j-th token after the
/// event (linear decay to 0 over `ev.tokens`).
pub fn slip_pitch(ev: &SlipEvent, j: usize, f0_sd: f64) -> f32 {
    if j >= ev.tokens {
        return 0.0;
    }
    let w = 1.0 - j as f32 / ev.tokens as f32;
    (ev.pitch_st as f64 * std::f64::consts::LN_2 / 12.0 / f0_sd) as f32 * w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rest_and_missing_are_identity() {
        assert_eq!(map_v0(&BodyState::default().vec()), VoiceMod::IDENTITY);
        let r = map_v0(&preset("rest").unwrap().vec());
        assert!(
            r.pitch_st.abs() < 0.01
                && r.energy_db.abs() < 0.01
                && (r.pace - 1.0).abs() < 1e-3
                && r.phrase_words == 0
        );
    }

    #[test]
    fn knobs_monotone() {
        let sweep = |f: &dyn Fn(f32) -> BodyVec| {
            (0..=10)
                .map(|i| map_v0(&f(i as f32 / 10.0)))
                .collect::<Vec<_>>()
        };
        let e = sweep(&|x| BodyVec {
            effort: x,
            ..Default::default()
        });
        let h = sweep(&|x| BodyVec {
            heat: x,
            ..Default::default()
        });
        let f = sweep(&|x| BodyVec {
            fatigue: x,
            ..Default::default()
        });
        for w in e.windows(2) {
            assert!(
                w[1].pitch_st > w[0].pitch_st
                    && w[1].energy_db > w[0].energy_db
                    && w[1].pace < w[0].pace
            );
        }
        for s in [&h, &f] {
            for w in s.windows(2) {
                assert!(
                    w[1].pitch_st < w[0].pitch_st
                        && w[1].energy_db < w[0].energy_db
                        && w[1].pace > w[0].pace
                        && w[1].range < w[0].range
                );
            }
        }
    }

    #[test]
    fn identity_edit_is_exact() {
        let (mut p, mut e, mut d) = (0.3f32, -0.2f32, 7u32);
        apply_token(&VoiceMod::IDENTITY, &mut p, &mut e, &mut d, 0.1, 0.35, 2.48);
        assert_eq!((p, e, d), (0.3, -0.2, 7));
    }

    #[test]
    fn slip_onsets_refractory() {
        let mut tr = BodyTrack::default();
        for i in 0..100 {
            let w = if (10..15).contains(&i) || (20..25).contains(&i) || (90..95).contains(&i) {
                1.8
            } else {
                0.1
            };
            tr.t.push(i as f64 * 0.1);
            tr.s.push(BodyState {
                tilt_rate: Some(w),
                ..Default::default()
            });
        }
        let s = tr.slips();
        assert_eq!(s.len(), 2, "{s:?}");
        assert!((s[0].0 - 1.0).abs() < 1e-9 && (s[1].0 - 9.0).abs() < 1e-9);
    }

    #[test]
    fn tsv_missing_is_none() {
        let tr =
            BodyTrack::load_tsv("t\teffort_w\theat_c\tsoc\n0\t10\tnan\tnan\n0.5\t20\t50\t0.4\n");
        assert_eq!(tr.s.len(), 2);
        assert_eq!(tr.at(0.2).heat_c, None);
        assert_eq!(tr.at(0.7).heat_c, Some(50.0));
    }
}
