//! `ribt-tts stream`: streaming chunked synthesis of the acoustic, flow and vocoder chain
//! (the ribt paper).
//!
//!   ribt-tts stream --run DIR [--ckpt C] --voc V --flow F [--nfe 1 --temp 0.3 --seed 0]
//!                   (--text "…" | --lines FILE | --stdin) --out-dir DIR [--ref REF.wav]
//!                   [--wps 0]          simulated word arrival rate (words/s; 0 = all text at t=0)
//!                   [--look-words 2]   text lookahead: words held back from the encoder commit
//!                   [--no-clause]      don't commit early at clause punctuation
//!                   [--chunk 48 --lctx 40 --rctx 40 --win 128 --xfade 4]   frame chunking
//!                   [--first C0 --first-rctx R0]  smaller chunk 0 / right context (default = chunk/rctx)
//!                   [--full]           also render the whole utterance in one window with the
//!                                      same noise and the same committed conditioning
//!                                      (`{name}_{spk}.full.wav`): the seam reference
//!                   [--pcm FILE]       append s16le 24 kHz mono as each chunk is produced ("-": stdout)
//!                   [--phones]         (--stdin) each phoneme on stderr as it's committed, before its audio:
//!                                      `phone\t<sample>\t<ipa>`, sample counted from the run's first --pcm sample
//!
//! Design (host-carried state; the IR has no Scan):
//! - text: words go into lilt's streaming `Phonemizer::push`. After each
//!   push that emits words, the encoder (+ duration/pitch/energy predictors)
//!   re-runs on the whole token prefix (a few ms). Tokens of all but the
//!   last `look-words` emitted words (or up to the last clause punctuation)
//!   are *committed*: their h, pitch, energy and duration are frozen and never
//!   recomputed, so audio already emitted is never contradicted.
//! - frames: chunk k is frames [kC, (k+1)C). It is rendered once the
//!   committed frames reach (k+1)C + rctx (or the text is final), by running
//!   rung-1 decoder → 1-step flow → vocoder on one fixed window of `win`
//!   frames starting at kC − lctx: left/right context is recomputed, not
//!   carried. One compiled program per stage (fixed shapes).
//! - flow noise: one noise field per utterance indexed by absolute frame, so
//!   overlapping windows see the same x₀.
//! - seams: each chunk also renders `xfade` frames past its end; the next
//!   chunk's head is linearly crossfaded with them.
//!
//! Clock (file/--text modes) is simulated: word i arrives at i/wps; each
//! compute step adds its measured wall time. Playback starts when chunk 0 is
//! ready; a chunk ready after its scheduled play time is an underrun
//! (playback stalls until it is ready).

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use ribt::acoustic::{self, Dims, TConfig};
use ribt::flow::{self, FConfig, FDims};
use ribt::model::Rng;
use ribt::runtime::Device;
use ribt::{body, ckpt, fx, lilt, persona};
use std::sync::Arc;

use super::flow_cmd::{FNorm, Vel};
use super::{dur_of, load_run, onehot_rows, write_npy, Args, Frontend, Norm, Tts, Vocoder, DEMO_REF, HOP, REF_T, SR};

/// longest utterance the noise field covers (frames, ~87 s)
const NMAX: usize = 8192;
/// log-mel floor used to pad windows past the end of the utterance
const MEL_FLOOR: f32 = -11.512925;
/// tokens of context on each side of a decoder window (pitch/energy embeddings are k3 convs)
const TOK_PAD: usize = 2;

#[derive(Clone, Debug)]
pub struct SCfg {
    pub look_words: usize,
    pub clause: bool,
    pub chunk: usize,
    pub lctx: usize,
    pub rctx: usize,
    pub win: usize,
    pub xfade: usize,
    /// chunk 0 size and right context (smaller = earlier first audio)
    pub first: usize,
    pub first_rctx: usize,
    pub nfe: usize,
    pub shortcut: bool,
    pub temp: f64,
    pub pace: f64,
    /// global pitch offset, semitones (--pitch)
    pub pitch: f64,
    /// body voice (§28.2): the body signal over speech time (None = the plain voice exactly)
    pub body: Option<body::BodyTrack>,
    /// speech time 0 = this time in the body track, s
    pub body_t0: f64,
    /// body voice v1 (§28.6): dynamics + mood timeline; takes precedence over `body`
    pub persona: Option<Arc<persona::Timeline>>,
    /// v1 master knob: 0 = plain voice, 1 = designed, 2 = theatrical
    pub drama: f32,
}

/// The models, compiled at fixed shapes (caches keyed by bucket).
pub struct Engine {
    dev: Device,
    tc: TConfig,
    params: Vec<Vec<f32>>,
    nm: Norm,
    fc: FConfig,
    /// None with a hidden-input vocoder (hidden-decoder rung 1: no flow stage)
    fparams: Option<Vec<Vec<f32>>>,
    fnm: Option<FNorm>,
    voc_path: PathBuf,
    enc: HashMap<usize, Tts>,
    dec: HashMap<(usize, usize), Tts>,
    vel: HashMap<usize, Vel>,
    voc: HashMap<usize, Vocoder>,
    refmel: Vec<f32>,
    rw: Vec<f32>,
    pub spk_tag: String,
    fe: Frontend,
    pub cfg: SCfg,
}

/// One chunk's record.
#[derive(Clone, Debug)]
pub struct ChunkLog {
    pub k: usize,
    pub f0: usize,
    pub f1: usize,
    pub words_in: usize,
    pub committed_frames: usize,
    pub compute_ms: f64,
    pub ready_ms: f64,
}

/// Streaming state of one utterance (all the carried state lives here, on the host).
pub struct Utt {
    ph: lilt::Phonemizer,
    /// [0] + ids of the emitted words (+ [0] once final)
    toks: Vec<u16>,
    /// token end index of each emitted word, and whether it closes a clause
    word_end: Vec<(usize, bool)>,
    /// lilt word texts (same order as word_end)
    pub word_text: Vec<String>,
    /// committed token-level conditioning
    h: Vec<Vec<f32>>,
    pitch: Vec<f32>,
    energy: Vec<f32>,
    durs: Vec<u32>,
    spk: Vec<f32>,
    final_: bool,
    next_chunk: usize,
    noise: Vec<f32>,
    pub audio: Vec<f32>,
    tail: Vec<f32>,
    pub phonemes: String,
    /// (token index committed up to, words emitted by lilt so far) per commit
    pub commits: Vec<(usize, usize)>,
    /// barge-in: no more rendering (§28.1)
    pub cancelled: bool,
    /// body voice state: words since the last pause, slip onsets consumed,
    /// active slip catch (event, tokens into it), pause owed at the next word end
    words_since_pause: usize,
    slips_used: usize,
    slip: Option<(body::SlipEvent, usize)>,
    pause_owed: f32,
    /// per committed token: (speech time s, body vector, edit, slip pitch st)
    pub body_log: Vec<(f64, body::BodyVec, body::VoiceMod, f32)>,
    /// v1: per committed token (speech time s, style)
    pub persona_log: Vec<(f64, persona::Style)>,
    /// v1: utterance F0 centre, semitones re 100 Hz (frozen at the first commit)
    pub centre_st: Option<f32>,
    /// v1: inhales to mix in, (absolute sample, audio)
    pub breaths: Vec<(usize, Vec<f32>)>,
    /// §29.5 harmonic source: excitation phase at each emitted frame centre
    /// (absolute frame index), so every window starts from the phase already
    /// played and the sine stays continuous across chunks (host-carried: no Scan)
    phase: Vec<f64>,
}

impl Utt {
    pub fn new(n_mels: usize, seed: u64, temp: f64) -> Self {
        let mut rng = Rng::new(seed);
        let noise = (0..n_mels * NMAX).map(|_| (temp * rng.normal()) as f32).collect();
        Self {
            ph: lilt::Phonemizer::new(),
            toks: vec![0],
            word_end: Vec::new(),
            word_text: Vec::new(),
            h: Vec::new(),
            pitch: Vec::new(),
            energy: Vec::new(),
            durs: Vec::new(),
            spk: Vec::new(),
            final_: false,
            next_chunk: 0,
            noise,
            audio: Vec::new(),
            tail: Vec::new(),
            phonemes: String::new(),
            commits: Vec::new(),
            cancelled: false,
            words_since_pause: 0,
            slips_used: 0,
            slip: None,
            pause_owed: 0.0,
            body_log: Vec::new(),
            persona_log: Vec::new(),
            centre_st: None,
            breaths: Vec::new(),
            phase: Vec::new(),
        }
    }
    /// all text is in and every chunk has been rendered
    pub fn final_done(&self) -> bool {
        self.final_ && self.next_chunk > 0 && self.audio.len() >= self.frames() * HOP
    }
    /// token end index (exclusive) of each lilt word
    pub fn word_ends(&self) -> Vec<usize> {
        self.word_end.iter().map(|w| w.0).collect()
    }
    /// committed frames
    pub fn frames(&self) -> usize {
        self.durs.iter().sum::<u32>() as usize
    }
    pub fn durs(&self) -> &[u32] {
        &self.durs
    }
}

fn expand_home(r: &str) -> PathBuf {
    match r.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(r),
    }
}

impl Engine {
    pub fn new(a: &Args, cfg: SCfg) -> Result<Self> {
        let tc = a.cfg()?;
        let run = PathBuf::from(a.need("--run")?);
        let (params, nm, _) = load_run(&run, &tc, a.opt("--ckpt"))?;
        let fc = FConfig::by_name(&a.opt("--fcfg").unwrap_or_else(|| "base".into())).ok_or_else(|| anyhow!("fcfg"))?;
        let (fparams, fnm) = match a.opt("--flow") {
            Some(f) => {
                let fp = PathBuf::from(f);
                (Some(ckpt::load_specs(&fp, &fc.params(), false)?.params), Some(FNorm::load(&fp)?))
            }
            None => (None, None),
        };
        let dev = Device::new();
        let r = a.opt("--ref").unwrap_or_else(|| DEMO_REF.into());
        let fe = Frontend::new(&dev, 1, REF_T);
        Ok(Self {
            dev,
            tc,
            params,
            nm,
            fc,
            fparams,
            fnm,
            voc_path: PathBuf::from(a.need("--voc")?),
            enc: HashMap::new(),
            dec: HashMap::new(),
            vel: HashMap::new(),
            voc: HashMap::new(),
            refmel: Vec::new(),
            rw: Vec::new(),
            spk_tag: String::new(),
            fe,
            cfg,
        })
        .and_then(|mut e| {
            e.set_ref(&r)?;
            Ok(e)
        })
    }

    /// Speaker reference: REF_T frames after 0.3 s of `r` (as synth).
    pub fn set_ref(&mut self, r: &str) -> Result<()> {
        let (x, sr) = ribt::data::read_wav(&expand_home(r))?;
        if sr as usize != SR {
            return Err(anyhow!("{r}: {sr} Hz"));
        }
        let fr = x.len() / HOP;
        let off = if fr > REF_T + 30 { 30 } else { 0 };
        let len = (fr - off).min(REF_T);
        self.refmel = self.fe.run(&self.dev, &[x[off * HOP..(off + len) * HOP].to_vec()]);
        self.rw = vec![0.0f32; REF_T];
        self.rw.iter_mut().take(len).for_each(|v| *v = 1.0 / len as f32);
        self.spk_tag = Path::new(r).file_stem().unwrap().to_string_lossy().split('_').next().unwrap_or("spk").to_string();
        Ok(())
    }

    pub fn n_mels(&self) -> usize {
        self.tc.n_mels
    }

    /// Compile the fixed-shape programs ahead of time (encoder buckets up to
    /// `max_tok`, decoder token buckets 32/64, the window programs).
    pub fn warm(&mut self, max_tok: usize) -> Result<()> {
        for n in (32..=max_tok).step_by(32) {
            self.enc_prog(n);
        }
        for nb in [32, 64] {
            self.dec_prog(nb, self.cfg.win);
        }
        if self.fparams.is_some() {
            self.vel_prog(self.cfg.win);
        }
        self.voc_prog(self.cfg.win)?;
        Ok(())
    }

    fn enc_prog(&mut self, n: usize) -> &mut Tts {
        let (dev, tc, params) = (&self.dev, &self.tc, &self.params);
        self.enc.entry(n).or_insert_with(|| Tts::new(dev, tc, params, Dims { batch: 1, n, t: 64, r: REF_T }))
    }
    fn dec_prog(&mut self, n: usize, t: usize) -> &mut Tts {
        let (dev, tc, params) = (&self.dev, &self.tc, &self.params);
        self.dec.entry((n, t)).or_insert_with(|| Tts::new(dev, tc, params, Dims { batch: 1, n, t, r: REF_T }))
    }
    fn vel_prog(&mut self, t: usize) -> &mut Vel {
        let (dev, fc, fp) = (&self.dev, &self.fc, self.fparams.as_ref().expect("--flow"));
        self.vel.entry(t).or_insert_with(|| Vel::new(dev, fc, fp, FDims { batch: 1, t }))
    }
    fn voc_prog(&mut self, t: usize) -> Result<&mut Vocoder> {
        if !self.voc.contains_key(&t) {
            let v = Vocoder::new(&self.dev, &self.voc_path, t)?;
            self.voc.insert(t, v);
        }
        Ok(self.voc.get_mut(&t).unwrap())
    }

    /// Feed text (a word with its trailing space; `fin` ends the utterance).
    /// Returns compute ms (lilt + encoder).
    pub fn push_text(&mut self, u: &mut Utt, text: &str, fin: bool) -> f64 {
        let t0 = Instant::now();
        let mut words: Vec<lilt::WordOut> = u.ph.push(text).collect();
        if fin {
            words.extend(u.ph.finish());
        }
        for w in &words {
            if w.prosody.oov {
                eprintln!("OOV (dropped by lilt): {}", w.text);
            }
            let mut s = w.phonemes.clone();
            if w.ws {
                s.push(' ');
            }
            u.phonemes.push_str(&s);
            u.toks.extend(lilt::vocab::ids(&s).iter().map(|&v| v as u16));
            u.word_end.push((u.toks.len(), w.prosody.boundary.is_some()));
            u.word_text.push(w.text.clone());
        }
        if fin && !u.final_ {
            u.toks.push(0);
            u.final_ = true;
        }
        let nw = u.word_end.len();
        let commit = if u.final_ {
            u.toks.len()
        } else {
            let by_look = if nw > self.cfg.look_words { u.word_end[nw - 1 - self.cfg.look_words].0 } else { 0 };
            let by_clause = if self.cfg.clause { u.word_end.iter().rev().find(|w| w.1).map_or(0, |w| w.0) } else { 0 };
            by_look.max(by_clause)
        };
        if commit > u.durs.len() {
            self.encode_commit(u, commit);
            u.commits.push((commit, nw));
        }
        t0.elapsed().as_secs_f64() * 1e3
    }

    /// Encoder over the token prefix; freeze tokens [committed, upto).
    fn encode_commit(&mut self, u: &mut Utt, upto: usize) {
        let n = u.toks.len().div_ceil(32) * 32;
        let (oh, tm) = onehot_rows(&self.tc, &[&u.toks], n);
        self.enc_prog(n);
        let tts = self.enc.get_mut(&n).unwrap();
        let [h, ld, mut pp, ep, spk] = tts.encode(&self.dev, &oh, &tm, &self.refmel, &self.rw);
        super::shift_pitch(&mut pp, self.cfg.pitch, &self.nm);
        let hd = h.len() / n;
        let nt = u.toks.len();
        let centre = pp[1..nt.saturating_sub(1).max(2)].iter().sum::<f32>() / (nt.saturating_sub(2).max(1)) as f32;
        for i in u.durs.len()..upto {
            u.h.push((0..hd).map(|c| h[c * n + i]).collect());
            let (mut p, mut e, mut d) = (pp[i], ep[i], dur_of(ld[i], self.cfg.pace));
            if self.cfg.persona.is_some() {
                self.persona_token(u, i, centre, &mut p, &mut e, &mut d);
            } else if self.cfg.body.is_some() {
                self.body_token(u, i, centre, &mut p, &mut e, &mut d);
            }
            u.pitch.push(p);
            u.energy.push(e);
            u.durs.push(d);
        }
        u.spk = spk;
        if u.centre_st.is_none() && upto > 2 {
            let m = u.pitch[1..upto - 1].iter().sum::<f32>() / (upto - 2) as f32;
            let hz = ((m as f64) * self.nm.f0_sd + self.nm.f0_mu).exp();
            u.centre_st = Some((12.0 * (hz / 100.0).log2()) as f32);
        }
    }

    /// Body voice v1 edit of token i (§28.6): style at its speech time,
    /// pauses and footfall-synced inhales at word ends.
    fn persona_token(&self, u: &mut Utt, i: usize, centre: f32, p: &mut f32, e: &mut f32, d: &mut u32) {
        let tl = self.cfg.persona.as_ref().unwrap().clone();
        let t = u.frames() as f64 * HOP as f64 / SR as f64;
        let tb = t + self.cfg.body_t0;
        let st = tl.style(tb, self.cfg.drama);
        body::apply_token(&st.tok, p, e, d, centre, self.nm.f0_sd, self.nm.en_sd);
        if let Some(w) = u.word_end.iter().find(|w| w.0 == i + 1).copied() {
            u.words_since_pause += 1;
            let (mut pause, mut breathe) = (0.0f32, false);
            if w.1 {
                u.words_since_pause = 0;
                pause = st.clause_pause.max(0.0);
                if st.clause_breath {
                    pause += st.tok.pause_s;
                    breathe = true;
                }
            } else if st.tok.phrase_words > 0 && u.words_since_pause >= st.tok.phrase_words {
                pause = st.tok.pause_s;
                breathe = true;
                u.words_since_pause = 0;
            }
            // the last word of the utterance: no pause after it
            if u.final_ && i + 2 >= u.toks.len() {
                pause = 0.0;
            }
            if pause > 0.0 {
                let d0 = *d;
                let pf = (pause as f64 * SR as f64 / HOP as f64).round() as u32;
                *d += pf;
                *e += (body::PAUSE_DB as f64 * std::f64::consts::LN_10 / 10.0 / self.nm.en_sd) as f32;
                if breathe {
                    let ps = (u.frames() + d0 as usize) as f64 * HOP as f64 / SR as f64; // pause start, speech time
                    let pl = pause as f64;
                    let dur = (0.18 + 0.32 * st.need as f64).min(pl - 0.06).max(0.1);
                    // sync the inhale onset to a footfall inside the pause when there is one
                    let lo = ps + 0.03 + self.cfg.body_t0;
                    let hi = ps + pl - dur - 0.02 + self.cfg.body_t0;
                    let on = tl.footfalls.iter().copied().find(|&f| f >= lo && f <= hi).unwrap_or(lo) - self.cfg.body_t0;
                    let seed = (u.breaths.len() as u64 + 1) * 7919 + (on * 1e3) as u64;
                    let x = fx::inhale(SR, dur as f32, st.need.max(0.3), 0.0, seed);
                    u.breaths.push(((on * SR as f64) as usize, x));
                }
            }
        }
        u.persona_log.push((t, st));
    }

    /// Body voice edit of token i (§28.2), at its speech time.
    fn body_token(&self, u: &mut Utt, i: usize, centre: f32, p: &mut f32, e: &mut f32, d: &mut u32) {
        let tr = self.cfg.body.as_ref().unwrap();
        let t = u.frames() as f64 * HOP as f64 / SR as f64;
        let tb = t + self.cfg.body_t0;
        let bv = tr.at(tb).vec();
        let mut m = body::map_v0(&bv);
        // diagnostic: RIBT_BODY_ONLY=pitch|range|energy|pace|pause keeps one knob of the map
        if let Ok(k) = std::env::var("RIBT_BODY_ONLY") {
            let i = body::VoiceMod::IDENTITY;
            m = match k.as_str() {
                "pitch" => body::VoiceMod { pitch_st: m.pitch_st, ..i },
                "range" => body::VoiceMod { range: m.range, ..i },
                "energy" => body::VoiceMod { energy_db: m.energy_db, ..i },
                "pace" => body::VoiceMod { pace: m.pace, ..i },
                "pause" => body::VoiceMod { phrase_words: m.phrase_words, pause_s: m.pause_s, ..i },
                _ => m,
            };
        }
        body::apply_token(&m, p, e, d, centre, self.nm.f0_sd, self.nm.en_sd);
        // slip onsets that have happened by this token's time
        let slips = tr.slips();
        let mut slip_st = 0.0f32;
        while u.slips_used < slips.len() && slips[u.slips_used].0 <= tb {
            let ev = body::slip_v0(slips[u.slips_used].1);
            u.slip = Some((ev, 0));
            u.pause_owed = u.pause_owed.max(ev.pause_s);
            u.slips_used += 1;
        }
        if let Some((ev, j)) = u.slip {
            let dp = body::slip_pitch(&ev, j, self.nm.f0_sd);
            *p += dp;
            slip_st = (dp as f64 * self.nm.f0_sd * 12.0 / std::f64::consts::LN_2) as f32;
            u.slip = (j + 1 < ev.tokens).then_some((ev, j + 1));
        }
        // word boundary: the last token of a lilt word
        if let Some(w) = u.word_end.iter().find(|w| w.0 == i + 1) {
            u.words_since_pause += 1;
            let mut pause = u.pause_owed;
            if w.1 {
                u.words_since_pause = 0;
            } else if m.phrase_words > 0 && u.words_since_pause >= m.phrase_words {
                pause = pause.max(m.pause_s);
            }
            if pause > 0.0 {
                *d += (pause as f64 * SR as f64 / HOP as f64).round() as u32;
                // a stretched boundary token alone renders as a murmur; drop its energy
                *e += (body::PAUSE_DB as f64 * std::f64::consts::LN_10 / 10.0 / self.nm.en_sd) as f32;
                u.words_since_pause = 0;
                u.pause_owed = 0.0;
            }
        }
        u.body_log.push((t, bv, m, slip_st));
    }

    /// Barge-in (§28.1): stop rendering this utterance. Playback-side fade and
    /// the spoken-word report live in `ribt::barge`.
    pub fn cancel(&mut self, u: &mut Utt) {
        u.cancelled = true;
    }

    /// Render frames [w0, w0 + t) of the committed conditioning in one window:
    /// decoder → flow → vocoder. Returns (audio t·HOP, valid frames, mel [C, valid]).
    pub fn render(&mut self, u: &Utt, w0: usize, t: usize) -> Result<(Vec<f32>, usize, Vec<f32>)> {
        let fa = u.frames();
        let valid = fa.min(w0 + t).saturating_sub(w0);
        // token span intersecting the window, plus TOK_PAD each side
        let mut pos = 0usize;
        let (mut ta, mut tb) = (usize::MAX, 0usize);
        let mut starts = Vec::with_capacity(u.durs.len());
        for (i, &d) in u.durs.iter().enumerate() {
            starts.push(pos);
            let e = pos + d as usize;
            if d > 0 && e > w0 && pos < w0 + valid {
                ta = ta.min(i);
                tb = i + 1;
            }
            pos = e;
        }
        if ta == usize::MAX {
            return Err(anyhow!("empty window at {w0}"));
        }
        let ta = ta.saturating_sub(TOK_PAD);
        let tb = (tb + TOK_PAD).min(u.durs.len());
        let nw = tb - ta;
        let nb = nw.div_ceil(32) * 32;
        let hd = u.h[0].len();
        let mut h = vec![0.0f32; hd * nb];
        let mut pp = vec![0.0f32; nb];
        let mut ep = vec![0.0f32; nb];
        let mut tm = vec![0.0f32; nb];
        for j in 0..nw {
            for c in 0..hd {
                h[c * nb + j] = u.h[ta + j][c];
            }
            pp[j] = u.pitch[ta + j];
            ep[j] = u.energy[ta + j];
            tm[j] = 1.0;
        }
        let al = acoustic::align_matrix_from(&[&u.durs[ta..tb]], &[w0 - starts[ta]], nb, t);
        let mut fm = vec![0.0f32; t];
        fm[..valid].iter_mut().for_each(|v| *v = 1.0);
        self.dec_prog(nb, t);
        let (mel, f0, hid) = self.dec.get_mut(&(nb, t)).unwrap().decode(&self.dev, &h, &pp, &ep, &tm, &al, &fm, &u.spk);
        let c_m = self.tc.n_mels;
        self.voc_prog(t)?;
        if self.voc[&t].is_hid() {
            // hidden-decoder rung 1: G reads the decoder's last layer; no flow, no mel fx
            let (mu, sd) = (self.nm.f0_mu, self.nm.f0_sd);
            let f0hz: Vec<f32> = (0..t).map(|f| if f < valid && f0[t + f] > 0.0 { ((f0[f] as f64) * sd + mu).exp() as f32 } else { 0.0 }).collect();
            let phi0 = u.phase.get(w0).copied().unwrap_or(0.0);
            let v = self.voc.get_mut(&t).unwrap();
            let x = v.hid_input(&hid, t, valid, t);
            let y = v.run_phase(&self.dev, &x, &f0hz, t, phi0);
            let mel_v: Vec<f32> = (0..c_m).flat_map(|c| mel[c * t..c * t + valid].to_vec()).collect();
            return Ok((y, valid, mel_v));
        }
        let fnm = self.fnm.as_ref().ok_or_else(|| anyhow!("a mel vocoder needs --flow"))?;
        let mun = fnm.norm(&mel, &fm, 1, t);
        let ab = flow::attn_bias(&[valid], t, None);
        let mut x0 = vec![0.0f32; c_m * t];
        for c in 0..c_m {
            x0[c * t..c * t + valid].copy_from_slice(&u.noise[c * NMAX + w0..c * NMAX + w0 + valid]);
        }
        let (nfe, sc) = (self.cfg.nfe, self.cfg.shortcut);
        self.vel_prog(t);
        let vel = self.vel.get_mut(&t).unwrap();
        vel.set_cond(&self.dev, &mun, &hid, &fm, &ab);
        let (x1, _) = vel.sample(&self.dev, &x0, nfe, sc);
        let mel = self.fnm.as_ref().unwrap().denorm(&x1);
        let mut mel_c = vec![MEL_FLOOR; c_m * t];
        for c in 0..c_m {
            mel_c[c * t..c * t + valid].copy_from_slice(&mel[c * t..c * t + valid]);
        }
        let (mu, sd) = (self.nm.f0_mu, self.nm.f0_sd);
        let mut f0hz: Vec<f32> = (0..t).map(|f| if f < valid && f0[t + f] > 0.0 { ((f0[f] as f64) * sd + mu).exp() as f32 } else { 0.0 }).collect();
        if let Some(tl) = self.cfg.persona.as_ref() {
            let only = std::env::var("RIBT_FX_ONLY").ok();
            let fxs: Vec<fx::FrameFx> = (0..valid)
                .map(|f| {
                    let e = tl.style((w0 + f) as f64 * HOP as f64 / SR as f64 + self.cfg.body_t0, self.cfg.drama).fx;
                    // diagnostic: RIBT_FX_ONLY=range|tilt|breathy|press keeps one frame knob
                    let i = fx::FrameFx::default();
                    match only.as_deref() {
                        Some("range") => fx::FrameFx { range: e.range, ..i },
                        Some("tilt") => fx::FrameFx { tilt: e.tilt, ..i },
                        Some("breathy") => fx::FrameFx { breathy: e.breathy, ..i },
                        Some("press") => fx::FrameFx { press: e.press, ..i },
                        Some("none") => i,
                        Some(x) if x.starts_with("rg=") => fx::FrameFx { range: x[3..].parse().unwrap_or(1.0), ..i },
                        Some(x) if x.starts_with("st=") => fx::FrameFx { f0_st: x[3..].parse().unwrap_or(0.0), ..i },
                        _ => e,
                    }
                })
                .collect();
            let hz = fx::band_hz(SR, c_m);
            fx::apply(&mut mel_c, &mut f0hz, t, valid, &fxs, w0, u.centre_st.unwrap_or(12.0), &hz, HOP as f64 / SR as f64);
        }
        self.voc_prog(t)?;
        let phi0 = u.phase.get(w0).copied().unwrap_or(0.0);
        let y = self.voc.get_mut(&t).unwrap().run_phase(&self.dev, &mel_c, &f0hz, t, phi0);
        let mel_v: Vec<f32> = (0..c_m).flat_map(|c| mel_c[c * t..c * t + valid].to_vec()).collect();
        Ok((y, valid, mel_v))
    }

    /// Render every chunk that is ready; returns (k, f0, f1, compute ms) for
    /// each. The utterance's audio grows by (f1 − f0)·HOP samples per chunk.
    pub fn pump(&mut self, u: &mut Utt) -> Result<Vec<(usize, usize, usize, f64)>> {
        self.pump_n(u, usize::MAX)
    }

    /// As `pump`, but render at most `max` chunks (playout-driven rendering).
    pub fn pump_n(&mut self, u: &mut Utt, max: usize) -> Result<Vec<(usize, usize, usize, f64)>> {
        let SCfg { chunk: c, lctx, rctx, win, xfade, first, first_rctx, .. } = self.cfg.clone();
        let mut out = Vec::new();
        while !u.cancelled && out.len() < max {
            let fa = u.frames();
            let k = u.next_chunk;
            let (f0, ck, rk) = if k == 0 { (0, first, first_rctx) } else { (first + (k - 1) * c, c, rctx) };
            let ready = if u.final_ { f0 < fa } else { fa >= f0 + ck + rk };
            if !ready {
                break;
            }
            let t0 = Instant::now();
            let f1 = (f0 + ck).min(fa);
            let w0 = f0.saturating_sub(lctx);
            let (y, valid, _) = self.render(u, w0, win)?;
            // harmonic source: keep the phase of frames emitted so far, extend with this window's
            let ph = &self.voc[&win].phis;
            for f in u.phase.len().max(w0)..(w0 + valid).min(w0 + ph.len()) {
                u.phase.push(ph[f - w0]);
            }
            // this chunk's audio, plus xfade frames of tail for the next one
            let s0 = (f0 - w0) * HOP;
            let s1 = (f1 - w0) * HOP;
            let st = ((f1 + xfade).min(w0 + valid) - w0) * HOP;
            let mut seg = y[s0..s1].to_vec();
            let n = u.tail.len().min(seg.len());
            for i in 0..n {
                let w = (i as f32 + 0.5) / n as f32;
                seg[i] = u.tail[i] * (1.0 - w) + seg[i] * w;
            }
            u.tail = y[s1..st].to_vec();
            let a0 = u.audio.len();
            for (bs, bx) in &u.breaths {
                let (lo, hi) = ((*bs).max(a0), (bs + bx.len()).min(a0 + seg.len()));
                for k in lo..hi {
                    seg[k - a0] += bx[k - bs];
                }
            }
            u.audio.extend_from_slice(&seg);
            u.next_chunk += 1;
            out.push((k, f0, f1, t0.elapsed().as_secs_f64() * 1e3));
        }
        Ok(out)
    }
}

fn pcm_write(p: &mut Option<Box<dyn Write>>, x: &[f32]) -> Result<()> {
    if let Some(p) = p.as_mut() {
        let b: Vec<u8> = x.iter().flat_map(|&v| ((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()).collect();
        p.write_all(&b)?;
        p.flush()?;
    }
    Ok(())
}

/// Stream config from the CLI (shared by `stream`, `barge`, `body`).
pub fn scfg_of(a: &Args) -> Result<SCfg> {
    let flag = |k: &str| a.0.iter().any(|x| x == k);
    let cfg = SCfg {
        look_words: a.get("--look-words", 2),
        clause: !flag("--no-clause"),
        chunk: a.get("--chunk", 48),
        lctx: a.get("--lctx", 40),
        rctx: a.get("--rctx", 40),
        win: a.get("--win", 128),
        xfade: a.get("--xfade", 4),
        first: a.get("--first", a.get("--chunk", 48)),
        first_rctx: a.get("--first-rctx", a.get("--rctx", 40)),
        nfe: a.get("--nfe", 1),
        shortcut: !flag("--no-shortcut"),
        temp: a.get("--temp", 0.3),
        pace: a.get("--pace", 1.0) / a.get("--rate", 1.0),
        pitch: a.get("--pitch", 0.0),
        body: None,
        body_t0: 0.0,
        persona: None,
        drama: a.get("--drama", 1.0),
    };
    if cfg.lctx + cfg.chunk + cfg.rctx > cfg.win || cfg.xfade > cfg.rctx {
        return Err(anyhow!("need lctx + chunk + rctx <= win and xfade <= rctx"));
    }
    Ok(cfg)
}

pub fn stream(a: &Args) -> Result<()> {
    let flag = |k: &str| a.0.iter().any(|x| x == k);
    let cfg = scfg_of(a)?;
    let seed: u64 = a.get("--seed", 0);
    let wps: f64 = a.get("--wps", 0.0);
    let full = flag("--full");
    let out_dir = PathBuf::from(a.opt("--out-dir").unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&out_dir)?;
    let mut eng = Engine::new(a, cfg.clone())?;
    let tw = Instant::now();
    eng.warm(256)?;
    eprintln!("compiled in {:.1} s; {cfg:?}", tw.elapsed().as_secs_f64());
    let mut pcm: Option<Box<dyn Write>> = match a.opt("--pcm").as_deref() {
        Some("-") => Some(Box::new(std::io::stdout())),
        Some(p) => Some(Box::new(std::fs::File::create(p)?)),
        None => None,
    };
    // warm-up utterance: the first run of each program is slower
    for _ in 0..2 {
        let mut u = Utt::new(eng.n_mels(), seed, cfg.temp);
        for w in "This is a warm up sentence, so that every program has run at least once.".split_whitespace() {
            eng.push_text(&mut u, &format!("{w} "), false);
            eng.pump(&mut u)?;
        }
        eng.push_text(&mut u, "", true);
        eng.pump(&mut u)?;
        if full {
            let t = u.frames().div_ceil(64) * 64;
            eng.render(&u, 0, t)?;
        }
    }
    if flag("--stdin") {
        let phones = a.0.iter().any(|x| x == "--phones");
        return stream_stdin(&mut eng, seed, &out_dir, &mut pcm, phones);
    }
    let dflt_ref = a.opt("--ref").unwrap_or_else(|| DEMO_REF.into());
    let jobs: Vec<(String, String, String)> = match (a.opt("--lines"), a.opt("--text")) {
        (Some(f), _) => std::fs::read_to_string(f)?
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .filter_map(|l| {
                let c: Vec<&str> = l.split('\t').collect();
                (c.len() >= 2).then(|| (c[0].to_string(), c[1].to_string(), c.get(2).map_or_else(|| dflt_ref.clone(), |r| r.to_string())))
            })
            .collect(),
        (None, Some(t)) => vec![("out".into(), t, dflt_ref.clone())],
        _ => return Err(anyhow!("--text, --lines or --stdin")),
    };
    let mut meta = std::fs::File::create(out_dir.join("synth.tsv"))?;
    writeln!(meta, "name\tspeaker\ttext\tphonemes\ttokens\tframes\tseconds\tttfa_ms\tfirst_compute_ms\tcompute_ms\trtf\tchunks\tunderruns\tstall_ms\tmax_chunk_ms\tfull_ms")?;
    let mut summ: Vec<(f64, f64, f64, usize, f64)> = Vec::new();
    for (name, text, rf) in &jobs {
        eng.set_ref(rf)?;
        let tag = eng.spk_tag.clone();
        let words: Vec<&str> = text.split_whitespace().collect();
        let arrive: Vec<f64> = (0..words.len()).map(|i| if wps > 0.0 { i as f64 * 1e3 / wps } else { 0.0 }).collect();
        let mut u = Utt::new(eng.n_mels(), seed, cfg.temp);
        let (mut now, mut compute) = (0.0f64, 0.0f64);
        let mut logs: Vec<ChunkLog> = Vec::new();
        let mut commit_at: Vec<(usize, f64, usize)> = Vec::new();
        for i in 0..=words.len() {
            let fin = i == words.len();
            if !fin {
                now = now.max(arrive[i]);
            }
            let nc = u.commits.len();
            let ms = if fin { eng.push_text(&mut u, "", true) } else { eng.push_text(&mut u, &format!("{} ", words[i]), false) };
            now += ms;
            compute += ms;
            if u.commits.len() > nc {
                commit_at.push((u.commits.last().unwrap().0, now, u.frames()));
            }
            let before = u.audio.len();
            for (k, f0, f1, ms) in eng.pump(&mut u)? {
                now += ms;
                compute += ms;
                logs.push(ChunkLog { k, f0, f1, words_in: (i + 1).min(words.len()), committed_frames: u.frames(), compute_ms: ms, ready_ms: now });
            }
            pcm_write(&mut pcm, &u.audio[before..])?;
        }
        // playback simulation
        let (mut sched, mut under, mut stall, mut first_compute) = (0.0f64, 0usize, 0.0f64, 0.0f64);
        let mut chunks = std::fs::File::create(out_dir.join(format!("{name}_{tag}.chunks.tsv")))?;
        writeln!(chunks, "k\tf0\tf1\twords_in\tcommitted_frames\tcompute_ms\tready_ms\tplay_ms\tunderrun")?;
        for (j, l) in logs.iter().enumerate() {
            let dur = (l.f1 - l.f0) as f64 * HOP as f64 * 1e3 / SR as f64;
            let mut ur = false;
            if j == 0 {
                sched = l.ready_ms;
                first_compute = l.ready_ms - arrive[l.words_in - 1];
            } else if l.ready_ms > sched + 1e-9 {
                ur = true;
                under += 1;
                stall += l.ready_ms - sched;
                sched = l.ready_ms;
            }
            writeln!(chunks, "{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{}", l.k, l.f0, l.f1, l.words_in, l.committed_frames, l.compute_ms, l.ready_ms, sched, ur as u8)?;
            sched += dur;
        }
        // word arrivals + commits, for the video
        let mut ev = std::fs::File::create(out_dir.join(format!("{name}_{tag}.events.tsv")))?;
        writeln!(ev, "kind\ti\tms\tinfo")?;
        for (i, w) in words.iter().enumerate() {
            writeln!(ev, "word\t{i}\t{:.2}\t{w}", arrive[i])?;
        }
        let mut pos = 0usize;
        for (i, &d) in u.durs().iter().enumerate() {
            writeln!(ev, "tok\t{i}\t{pos}\t{d}")?;
            pos += d as usize;
        }
        for (i, (w, (e, _))) in u.word_text.iter().zip(&u.word_end).enumerate() {
            writeln!(ev, "lword\t{i}\t{e}\t{w}")?;
        }
        for (tok, ms, fr) in &commit_at {
            writeln!(ev, "commit\t{tok}\t{ms:.2}\t{fr}")?;
        }
        let ttfa = logs.first().map_or(f64::NAN, |l| l.ready_ms);
        let maxc = logs.iter().map(|l| l.compute_ms).fold(0.0, f64::max);
        let secs = u.audio.len() as f64 / SR as f64;
        let stem = format!("{name}_{tag}");
        ribt::data::write_wav(&out_dir.join(format!("{stem}.wav")), &u.audio, SR as u32)?;
        let mut full_ms = 0.0;
        if full {
            let fr = u.frames();
            let t = fr.div_ceil(64) * 64;
            let t0 = Instant::now();
            let (y, _, mel) = eng.render(&u, 0, t)?;
            full_ms = t0.elapsed().as_secs_f64() * 1e3;
            ribt::data::write_wav(&out_dir.join(format!("{stem}.full.wav")), &y[..fr * HOP], SR as u32)?;
            write_npy(&out_dir.join(format!("{stem}.full.mel.npy")), &[eng.n_mels(), fr], &mel)?;
        }
        let rtf = compute / 1e3 / secs.max(1e-9);
        writeln!(
            meta,
            "{name}\t{tag}\t{text}\t{}\t{}\t{}\t{secs:.2}\t{ttfa:.1}\t{first_compute:.1}\t{compute:.1}\t{rtf:.4}\t{}\t{under}\t{stall:.1}\t{maxc:.1}\t{full_ms:.1}",
            u.phonemes,
            u.toks.len(),
            u.frames(),
            logs.len()
        )?;
        eprintln!("{stem}: {secs:.2} s, {} chunks, TTFA {ttfa:.1} ms (first-chunk compute {first_compute:.1}), compute {compute:.1} ms (RTF {rtf:.3}), underruns {under} ({stall:.0} ms), full {full_ms:.1} ms", logs.len());
        summ.push((ttfa, first_compute, rtf, under, maxc));
    }
    if !summ.is_empty() {
        let n = summ.len() as f64;
        let mut tt: Vec<f64> = summ.iter().map(|s| s.0).collect();
        tt.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut fc: Vec<f64> = summ.iter().map(|s| s.1).collect();
        fc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let line = format!(
            "summary n={}: TTFA median {:.1} ms p90 {:.1} max {:.1}; first-chunk compute median {:.1} max {:.1}; RTF mean {:.3}; utts with underruns {}; max chunk {:.1} ms",
            summ.len(),
            tt[tt.len() / 2],
            tt[(tt.len() * 9 / 10).min(tt.len() - 1)],
            tt[tt.len() - 1],
            fc[fc.len() / 2],
            fc[fc.len() - 1],
            summ.iter().map(|s| s.2).sum::<f64>() / n,
            summ.iter().filter(|s| s.3 > 0).count(),
            summ.iter().map(|s| s.4).fold(0.0, f64::max)
        );
        eprintln!("{line}");
        std::fs::write(out_dir.join("summary.txt"), line + "\n")?;
    }
    Ok(())
}

/// Real clock: words from stdin as they arrive (one or more per line); an
/// empty line ends an utterance, EOF ends the run. Audio goes to `--pcm` as
/// each chunk is produced; a log of word/chunk wall times per utterance.
fn stream_stdin(eng: &mut Engine, seed: u64, out_dir: &Path, pcm: &mut Option<Box<dyn Write>>, phones: bool) -> Result<()> {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut idx = 0usize;
    let mut eof = false;
    // samples written to --pcm by earlier utterances: where this one's sample 0 lands
    let mut base = 0usize;
    while !eof {
        let mut u = Utt::new(eng.n_mels(), seed, eng.cfg.temp);
        // tokens already announced, and the frame the next one starts on
        let (mut said, mut frame) = (0usize, 0usize);
        let t0 = Instant::now();
        let mut log = String::from("event\tms\tinfo\n");
        let mut any = false;
        loop {
            let line = match lines.next() {
                Some(Ok(l)) => Some(l),
                _ => {
                    eof = true;
                    None
                }
            };
            let fin = line.as_ref().is_none_or(|l| l.trim().is_empty());
            if let Some(l) = line.filter(|l| !l.trim().is_empty()) {
                any = true;
                log += &format!("word\t{:.2}\t{}\n", t0.elapsed().as_secs_f64() * 1e3, l.trim());
                for w in l.split_whitespace() {
                    eng.push_text(&mut u, &format!("{w} "), false);
                }
            }
            if fin && any {
                eng.push_text(&mut u, "", true);
            }
            if phones {
                let mut e = std::io::stderr().lock();
                for (&t, &d) in u.toks[said..u.durs().len()].iter().zip(&u.durs()[said..]) {
                    let ipa = ribt::lilt::vocab::VOCAB.iter().find(|v| v.1 as u16 == t).map_or(' ', |v| v.0);
                    writeln!(e, "phone\t{}\t{ipa}", base + frame * HOP)?;
                    frame += d as usize;
                }
                said = u.durs().len();
            }
            let before = u.audio.len();
            for (k, f0, f1, ms) in eng.pump(&mut u)? {
                log += &format!("chunk\t{:.2}\tk={k} frames={f0}..{f1} compute={ms:.1}\n", t0.elapsed().as_secs_f64() * 1e3);
            }
            pcm_write(pcm, &u.audio[before..])?;
            if fin {
                base += u.audio.len();
                break;
            }
        }
        if any {
            let stem = format!("stdin{idx:03}_{}", eng.spk_tag);
            ribt::data::write_wav(&out_dir.join(format!("{stem}.wav")), &u.audio, SR as u32)?;
            std::fs::write(out_dir.join(format!("{stem}.log.tsv")), log)?;
            eprintln!("{stem}: {:.2} s", u.audio.len() as f64 / SR as f64);
            idx += 1;
        }
    }
    Ok(())
}
