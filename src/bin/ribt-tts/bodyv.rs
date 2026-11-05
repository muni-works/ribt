//! `ribt-tts barge` and `ribt-tts body`: the robot-native voice (the ribt paper).
//!
//!   ribt-tts barge <stream args> --text "…" --at-ms 900,1700 [--block-ms 10 --fade-ms 10]
//!                  [--keep 0.8] [--gap-ms 350] [--replan "new text"] --out-dir DIR
//!       Plays the utterance through a `barge::Playout` (device blocks), cancels at each
//!       --at-ms (playback time), reports what was heard and the cancel latency, then
//!       resumes from the interrupted word (or speaks --replan) after --gap-ms.
//!   ribt-tts body <stream args> (--text "…" | --lines FILE) --states rest,walking,hot,slipping,lowbatt
//!                 [--body-tsv F --body-t0 S --tag NAME] --out-dir DIR
//!       Renders every line at every body state (or driven by a body track) and logs the
//!       per-token body vector and edit (`{stem}.body.tsv`).

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use ribt::barge::{self, Playout};
use ribt::body::{self, BodyState, BodyTrack};
use ribt::persona::{self, Reflex, Timeline};
use std::sync::Arc;

use super::stream::{scfg_of, Engine, Utt};
use super::{Args, DEMO_REF, HOP, SR};

fn warm(eng: &mut Engine, seed: u64) -> Result<()> {
    eng.warm(256)?;
    for _ in 0..2 {
        let mut u = Utt::new(eng.n_mels(), seed, eng.cfg.temp);
        eng.push_text(
            &mut u,
            "This is a warm up sentence, so that every program has run at least once. ",
            true,
        );
        eng.pump(&mut u)?;
    }
    Ok(())
}

/// Whole utterance through the engine (all text at t = 0); returns first-chunk compute ms.
fn say(eng: &mut Engine, u: &mut Utt, text: &str) -> Result<f64> {
    let t0 = Instant::now();
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut first = f64::NAN;
    for (i, w) in words.iter().enumerate() {
        let pm = eng.push_text(u, &format!("{w} "), i + 1 == words.len());
        if std::env::var("RIBT_DBG").is_ok() {
            eprintln!(
                "push {w}: {pm:.1} ms at {:.1}",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        if first.is_nan() && !eng.pump_n(u, 1)?.is_empty() {
            first = t0.elapsed().as_secs_f64() * 1e3;
        }
        eng.pump(u)?;
    }
    if words.is_empty() {
        eng.push_text(u, "", true);
    }
    eng.pump(u)?;
    Ok(first)
}

pub fn barge(a: &Args) -> Result<()> {
    let cfg = scfg_of(a)?;
    let seed: u64 = a.get("--seed", 0);
    let text = a.need("--text")?;
    let ats: Vec<f64> = a
        .need("--at-ms")?
        .split(',')
        .map(|x| x.parse().unwrap_or(0.0))
        .collect();
    let block = (a.get("--block-ms", 10.0f64) * SR as f64 / 1e3) as usize;
    let fade = (a.get("--fade-ms", 10.0f64) * SR as f64 / 1e3) as usize;
    let keep: f32 = a.get("--keep", 0.8);
    let gap = (a.get("--gap-ms", 350.0f64) * SR as f64 / 1e3) as usize;
    let replan = a.opt("--replan");
    let out_dir = PathBuf::from(a.opt("--out-dir").unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&out_dir)?;
    let mut eng = Engine::new(a, cfg)?;
    eng.set_ref(&a.opt("--ref").unwrap_or_else(|| DEMO_REF.into()))?;
    warm(&mut eng, seed)?;
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut log = std::fs::File::create(out_dir.join("barge.tsv"))?;
    writeln!(
        log,
        "at_ms\tsignal_ms\tend_ms\taudio_after_signal_ms\tcancel_us\tchunks_rendered\tchunks_total\tword\tword_frac\theard\tresume_from\tresume_text\tresume_first_chunk_ms\thard_step\tfaded_step\tspeech_p999_step"
    )?;
    // reference: the full utterance (chunk count, and the uncancelled audio)
    let mut full = Utt::new(eng.n_mels(), seed, eng.cfg.temp);
    say(&mut eng, &mut full, &text)?;
    let total_chunks = full.audio.len().div_ceil(HOP * eng.cfg.chunk);
    ribt::data::write_wav(&out_dir.join("full.wav"), &full.audio, SR as u32)?;
    let mut steps: Vec<f32> = full.audio.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p999 = steps[(steps.len() as f64 * 0.999) as usize];
    for (n, &at) in ats.iter().enumerate() {
        let at_s = (at * SR as f64 / 1e3) as usize;
        let mut u = Utt::new(eng.n_mels(), seed, eng.cfg.temp);
        for (i, w) in words.iter().enumerate() {
            eng.push_text(&mut u, &format!("{w} "), i + 1 == words.len());
        }
        let mut po = Playout::new(block, fade);
        let mut played: Vec<f32> = Vec::new();
        let (mut rendered, mut cut, mut cancel_us) = (0usize, None, 0.0f64);
        loop {
            // render just in time: keep one chunk queued ahead of the device
            while !u.cancelled && po.buf.len() < po.played + block + HOP * eng.cfg.chunk {
                let before = u.audio.len();
                let r = eng.pump_n(&mut u, 1)?;
                if r.is_empty() {
                    break;
                }
                rendered += 1;
                po.push(&u.audio[before..]);
            }
            if cut.is_none() && po.played >= at_s {
                let t = Instant::now();
                let c = po.cancel();
                eng.cancel(&mut u);
                cancel_us = t.elapsed().as_secs_f64() * 1e6;
                cut = Some(c);
            }
            match po.next_block() {
                Some(b) => played.extend_from_slice(&b),
                None => break,
            }
        }
        let c = cut.ok_or_else(|| anyhow!("--at-ms {at} is past the end of the utterance"))?;
        let sp = barge::spoken(u.durs(), &u.word_ends(), HOP, c.end.saturating_sub(1));
        let heard = u.word_text[..sp.words_done.min(u.word_text.len())].join(" ");
        let rw = barge::resume_word(&sp, keep);
        let rtext = replan
            .clone()
            .unwrap_or_else(|| u.word_text[rw.min(u.word_text.len())..].join(" "));
        let hard_step = played
            .get(c.signal.saturating_sub(1))
            .map_or(0.0, |v| v.abs());
        let faded_step = played.last().map_or(0.0, |v| v.abs());
        played.extend(std::iter::repeat_n(0.0f32, gap));
        let mut r = Utt::new(eng.n_mels(), seed, eng.cfg.temp);
        let first = if rtext.trim().is_empty() {
            f64::NAN
        } else {
            say(&mut eng, &mut r, &rtext)?
        };
        played.extend_from_slice(&r.audio);
        ribt::data::write_wav(&out_dir.join(format!("barge{n}.wav")), &played, SR as u32)?;
        let ms = |s: usize| s as f64 * 1e3 / SR as f64;
        writeln!(
            log,
            "{at}\t{:.1}\t{:.1}\t{:.1}\t{cancel_us:.1}\t{rendered}\t{total_chunks}\t{}\t{:.2}\t{heard}\t{rw}\t{rtext}\t{first:.1}\t{hard_step:.4}\t{faded_step:.5}\t{p999:.4}",
            ms(c.signal),
            ms(c.end),
            ms(c.end) - at,
            sp.word,
            sp.word_frac
        )?;
        eprintln!(
            "cancel @{at} ms: audio after signal {:.1} ms (cancel call {cancel_us:.1} µs), rendered {rendered}/{total_chunks} chunks, in word {} '{}' ({:.0}% heard); heard \"{heard}\"; resume from {rw}: \"{rtext}\" (first chunk {first:.1} ms)",
            ms(c.end) - at,
            sp.word,
            u.word_text.get(sp.word).map_or("", |s| s.as_str()),
            sp.word_frac * 100.0
        );
    }
    Ok(())
}

/// A named state as a track; "slipping" also carries a slip onset at 0.9 s.
/// Custom states: `effort=60+heat=50+soc=0.3+tilt=0.2` (unset fields = rest).
fn state_track(name: &str) -> Option<BodyTrack> {
    if name.contains('=') {
        let mut s = body::preset("rest")?;
        for kv in name.split('+') {
            let (k, v) = kv.split_once('=')?;
            let v: f32 = v.parse().ok()?;
            match k {
                "effort" => s.effort_w = Some(v),
                "heat" => s.heat_c = Some(v),
                "soc" => s.soc = Some(v),
                "tilt" => s.tilt_rate = Some(v),
                _ => return None,
            }
        }
        return Some(BodyTrack::constant(s));
    }
    if name == "missing" {
        return Some(BodyTrack::constant(BodyState::default()));
    }
    let s = body::preset(name)?;
    if name != "slipping" {
        return Some(BodyTrack::constant(s));
    }
    let calm = BodyState {
        tilt_rate: Some(0.2),
        ..s
    };
    Some(BodyTrack {
        t: vec![0.0, 0.9, 1.2],
        s: vec![calm, s, calm],
    })
}

pub fn body(a: &Args) -> Result<()> {
    let cfg = scfg_of(a)?;
    let seed: u64 = a.get("--seed", 0);
    let out_dir = PathBuf::from(a.opt("--out-dir").unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&out_dir)?;
    let mut eng = Engine::new(a, cfg)?;
    let dflt_ref = a.opt("--ref").unwrap_or_else(|| DEMO_REF.into());
    eng.set_ref(&dflt_ref)?;
    warm(&mut eng, seed)?;
    let jobs: Vec<(String, String)> = match (a.opt("--lines"), a.opt("--text")) {
        (Some(f), _) => std::fs::read_to_string(f)?
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .filter_map(|l| {
                let c: Vec<&str> = l.split('\t').collect();
                (c.len() >= 2).then(|| (c[0].to_string(), c[1].to_string()))
            })
            .collect(),
        (None, Some(t)) => vec![("out".into(), t)],
        _ => return Err(anyhow!("--text or --lines")),
    };
    let mut tracks: Vec<(String, Option<BodyTrack>)> = Vec::new();
    if let Some(f) = a.opt("--body-tsv") {
        tracks.push((
            a.opt("--tag").unwrap_or_else(|| "replay".into()),
            Some(BodyTrack::load_tsv(&std::fs::read_to_string(f)?)),
        ));
        eng.cfg.body_t0 = a.get("--body-t0", 0.0);
    }
    for s in a
        .opt("--states")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
    {
        if s.starts_with("mood=") {
            continue;
        }
        if s == "none" {
            tracks.push((s.into(), None));
        } else {
            tracks.push((
                s.into(),
                Some(state_track(s).ok_or_else(|| anyhow!("unknown state {s}"))?),
            ));
        }
    }
    let v1 = a.0.iter().any(|x| x == "--v1");
    let prime: f64 = a.get("--prime", 30.0);
    // v1 timelines: tracks through the dynamics; "mood=NAME" = rest with that mood forced
    let mut tls: Vec<Option<Arc<Timeline>>> = Vec::new();
    for s in a.opt("--states").unwrap_or_default().split(',').filter(|s| s.starts_with("mood=")) {
        let m = persona::mood_named(&s[5..], a.get("--mood-w", 1.0)).ok_or_else(|| anyhow!("unknown mood {s}"))?;
        tracks.push((s.into(), Some(BodyTrack::constant(body::preset("rest").unwrap()))));
        let _ = m;
    }
    for (tag, tr) in &tracks {
        tls.push(tr.as_ref().filter(|_| v1).map(|tr| {
            let mut tl = Timeline::build(tr, if tag == "replay" { 0.0 } else { prime });
            if let Some(m) = tag.strip_prefix("mood=").and_then(|m| persona::mood_named(m, a.get("--mood-w", 1.0))) {
                tl.force_mood(m);
            }
            Arc::new(tl)
        }));
    }
    let mut meta = std::fs::File::create(out_dir.join("synth.tsv"))?;
    writeln!(meta, "name\tspeaker\ttext\tframes\tseconds\tbreaths")?;
    for (name, text) in &jobs {
        for ((tag, tr), tl) in tracks.iter().zip(&tls) {
            eng.cfg.body = if v1 { None } else { tr.clone() };
            eng.cfg.persona = tl.clone();
            let mut u = Utt::new(eng.n_mels(), seed, eng.cfg.temp);
            say(&mut eng, &mut u, text)?;
            let stem = format!("{name}_{tag}");
            ribt::data::write_wav(&out_dir.join(format!("{stem}.wav")), &u.audio, SR as u32)?;
            let mut f = std::fs::File::create(out_dir.join(format!("{stem}.body.tsv")))?;
            writeln!(f, "token\tt\tframes\teffort\theat\tslip\tfatigue\tpitch_st\trange\tenergy_db\tpace\tslip_st")?;
            for (i, ((t, b, m, ss), d)) in u.body_log.iter().zip(u.durs()).enumerate() {
                writeln!(
                    f,
                    "{i}\t{t:.3}\t{d}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.2}\t{:.3}\t{:.2}\t{:.3}\t{ss:.2}",
                    b.effort, b.heat, b.slip, b.fatigue, m.pitch_st, m.range, m.energy_db, m.pace
                )?;
            }
            if v1 {
                write_persona_log(&out_dir.join(format!("{stem}.style.tsv")), &u, eng.cfg.body_t0, tl.as_deref())?;
            }
            let secs = u.audio.len() as f64 / SR as f64;
            writeln!(meta, "{name}\t{tag}\t{text}\t{}\t{secs:.2}\t{}", u.frames(), u.breaths.len())?;
            eprintln!("{stem}: {secs:.2} s");
        }
    }
    Ok(())
}

fn write_persona_log(p: &std::path::Path, u: &Utt, t0: f64, tl: Option<&Timeline>) -> Result<()> {
    let mut f = std::fs::File::create(p)?;
    writeln!(f, "token\tt\tframes\teffort\theat\tfatigue\tbreathless\tslip\t{}\tpitch_st\tenergy_db\tpace\trange\tcatch_st\ttremor_st\ttilt\tbreathy\tpress\tneed", persona::MOODS.join("\t"))?;
    for (i, ((t, st), d)) in u.persona_log.iter().zip(u.durs()).enumerate() {
        let s = tl.map(|tl| tl.at(t + t0)).unwrap_or_default();
        let m: Vec<String> = s.mood.iter().map(|v| format!("{v:.3}")).collect();
        writeln!(
            f,
            "{i}\t{t:.3}\t{d}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{:.2}\t{:.2}\t{:.3}\t{:.3}\t{:.2}\t{:.2}\t{:.2}\t{:.3}\t{:.3}\t{:.3}",
            s.effort, s.heat, s.fatigue, s.breathless, s.slip, m.join("\t"), st.tok.pitch_st, st.tok.energy_db, st.tok.pace, st.fx.range, st.fx.f0_st, st.fx.tremor_st, st.fx.tilt, st.fx.breathy, st.fx.press, st.need
        )?;
    }
    let mut g = std::fs::File::create(p.with_extension("breaths.tsv"))?;
    writeln!(g, "start_s\tdur_s")?;
    for (s0, x) in &u.breaths {
        writeln!(g, "{:.3}\t{:.3}", *s0 as f64 / SR as f64, x.len() as f64 / SR as f64)?;
    }
    Ok(())
}

/// Reflex clips, rendered once at startup (surprised style) so a reflex costs
/// one device block: a sharp inhale, then the word (§28.7).
fn reflex_clips(eng: &mut Engine, seed: u64, drama: f32) -> Result<Vec<(Reflex, Vec<f32>)>> {
    let saved = (eng.cfg.persona.clone(), eng.cfg.body.clone(), eng.cfg.drama, eng.cfg.body_t0);
    let mut tl = Timeline::build(&BodyTrack::constant(body::preset("rest").unwrap()), 0.0);
    tl.force_mood(persona::mood_named("surprised", 1.0).unwrap());
    eng.cfg.persona = Some(Arc::new(tl));
    eng.cfg.body = None;
    eng.cfg.drama = drama.max(1.0);
    eng.cfg.body_t0 = 0.0;
    let mut out = Vec::new();
    for (k, r) in Reflex::ALL.iter().enumerate() {
        let gasp = ribt::fx::inhale(SR, 0.16, 0.9, 1.0, 1000 + k as u64);
        let mut x = gasp.clone();
        if !r.text().is_empty() {
            let mut u = Utt::new(eng.n_mels(), seed, eng.cfg.temp);
            say(eng, &mut u, r.text())?;
            // trim leading/trailing near-silence of the word, overlap 30 ms with the gasp tail
            let thr = 0.01 * u.audio.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let a = u.audio.iter().position(|v| v.abs() > thr).unwrap_or(0).saturating_sub(240);
            let b = u.audio.iter().rposition(|v| v.abs() > thr).map_or(u.audio.len(), |i| (i + 2400).min(u.audio.len()));
            let ov = (0.03 * SR as f64) as usize;
            let s0 = x.len() - ov;
            x.resize(s0 + (b - a), 0.0);
            for (i, v) in u.audio[a..b].iter().enumerate() {
                x[s0 + i] += v;
            }
        }
        out.push((*r, x));
    }
    (eng.cfg.persona, eng.cfg.body, eng.cfg.drama, eng.cfg.body_t0) = saved;
    Ok(out)
}

/// `ribt-tts story`: a timeline drives speech, reflexes and mood (§28.8).
///
///   ribt-tts story <stream args> --body-tsv TRACK --script FILE [--drama 1] [--gap 0.6]
///                  [--block-ms 10 --fade-ms 10 --keep 0.5] --out-dir DIR
///
/// FILE lines: `start_s<TAB>text` (start "-" = previous end + --gap). The body
/// track runs on the playback clock; each utterance is rendered just in time
/// with the style at its playback time; a reflex event cuts the current
/// utterance (barge machinery), plays its clip at the next device block, and
/// the rest of the utterance resumes after it (or nothing, on a fall).
pub fn story(a: &Args) -> Result<()> {
    let mut cfg = scfg_of(a)?;
    let seed: u64 = a.get("--seed", 0);
    let out_dir = PathBuf::from(a.opt("--out-dir").unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&out_dir)?;
    let tr = BodyTrack::load_tsv(&std::fs::read_to_string(a.need("--body-tsv")?)?);
    let tl = Arc::new(Timeline::build_with(&tr, a.get("--prime", 0.0), a.0.iter().any(|x| x == "--surprise")));
    cfg.persona = Some(tl.clone());
    let drama = cfg.drama;
    let block = (a.get("--block-ms", 10.0f64) * SR as f64 / 1e3) as usize;
    let fade = (a.get("--fade-ms", 10.0f64) * SR as f64 / 1e3) as usize;
    let keep: f32 = a.get("--keep", 0.5);
    let gap: f64 = a.get("--gap", 0.6);
    let mut eng = Engine::new(a, cfg)?;
    eng.set_ref(&a.opt("--ref").unwrap_or_else(|| DEMO_REF.into()))?;
    warm(&mut eng, seed)?;
    let t_clip = Instant::now();
    let clips = reflex_clips(&mut eng, seed, drama)?;
    let clip_ms = t_clip.elapsed().as_secs_f64() * 1e3;
    for (r, x) in &clips {
        ribt::data::write_wav(&out_dir.join(format!("reflex_{}.wav", r.name())), x, SR as u32)?;
    }
    let mut queue: std::collections::VecDeque<(Option<f64>, String)> = std::fs::read_to_string(a.need("--script")?)?
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (s, t) = l.split_once('\t').unwrap_or(("-", l));
            (s.trim().parse::<f64>().ok(), t.trim().to_string())
        })
        .collect();
    let mut events = String::from("kind\tt\tinfo\n");
    let mut out: Vec<f32> = Vec::new();
    let mut cur: Option<(Utt, Playout, f64)> = None;
    let mut refl: Option<(Vec<f32>, usize)> = None;
    let mut fired = 0usize;
    let mut next_free = 0.0f64; // earliest start of the next utterance
    let mut fallen = false;
    let mut lat: Vec<(f64, f64)> = Vec::new();
    let t_end = tl.s.last().map_or(0.0, |s| s.t);
    loop {
        let t = out.len() as f64 / SR as f64;
        // reflexes due at this block
        while fired < tl.reflexes.len() && tl.reflexes[fired].0 <= t {
            let (te, r, st) = tl.reflexes[fired];
            fired += 1;
            if r == Reflex::Gasp && refl.is_some() {
                continue;
            }
            // the bus samples at 50 Hz: the physical onset is up to one bus period
            // before the sample that reports it (deterministic pseudo-random offset)
            let onset = te - persona::DT * (((fired as u64).wrapping_mul(2654435761) % 1000) as f64 / 1000.0);
            let wall = Instant::now();
            // a gasp is mixed over the speech (no cut); words cut the utterance
            if let (Some((u, po, t0)), false) = (cur.as_mut(), r == Reflex::Gasp) {
                let c = po.cancel();
                eng.cancel(u);
                let sp = barge::spoken(u.durs(), &u.word_ends(), HOP, c.end.saturating_sub(1));
                let rw = barge::resume_word(&sp, keep);
                let heard = u.word_text[..sp.words_done.min(u.word_text.len())].join(" ");
                let rest = u.word_text[rw.min(u.word_text.len())..].join(" ");
                events += &format!("cut\t{:.3}\theard \"{heard}\"; resume \"{rest}\"\n", *t0 + c.end as f64 / SR as f64);
                // a push is shrugged off (resume); after whoa/oof/eep the rest is
                // dropped and reported: the dialogue manager re-plans (the next script line)
                if r == Reflex::Hup && !rest.trim().is_empty() {
                    queue.push_front((None, rest));
                } else if !rest.trim().is_empty() {
                    events += &format!("dropped\t{t:.3}\t{rest}\n");
                }
            }
            if r == Reflex::Eep {
                fallen = true;
                queue.clear();
            }
            let x = clips.iter().find(|c| c.0 == r).unwrap().1.clone();
            let g = if r == Reflex::Gasp { 0.6 + 0.4 * st } else { 1.0 };
            let dur = x.len() as f64 / SR as f64;
            refl = Some((x.iter().map(|v| v * g).collect(), 0));
            if r != Reflex::Gasp {
                next_free = next_free.max(t + dur + 0.25);
            }
            lat.push(((t - te) * 1e3, (t - onset) * 1e3));
            events += &format!(
                "reflex\t{te:.3}\t{} (strength {st:.2}); sound at {t:.3} s: +{:.1} ms after the bus sample, +{:.1} ms after onset; decision {:.1} µs\n",
                r.name(),
                (t - te) * 1e3,
                (t - onset) * 1e3,
                wall.elapsed().as_secs_f64() * 1e6
            );
        }
        // the current utterance is done when its playout is exhausted
        if cur.as_ref().is_some_and(|(u, po, _)| (po.cancelled() || u.final_done()) && po.played >= po.buf.len()) {
            let (u, _, t0) = cur.take().unwrap();
            for (s0, x) in &u.breaths {
                events += &format!("breath\t{:.3}\t{:.0} ms\n", t0 + *s0 as f64 / SR as f64, x.len() as f64 * 1e3 / SR as f64);
            }
            next_free = next_free.max(t + gap);
        }
        // start the next utterance
        if cur.is_none() && refl.is_none() && !fallen {
            if let Some((st, _)) = queue.front() {
                let start = st.unwrap_or(next_free).max(next_free);
                if t >= start {
                    let (_, text) = queue.pop_front().unwrap();
                    eng.cfg.body_t0 = t;
                    let mut u = Utt::new(eng.n_mels(), seed, eng.cfg.temp);
                    let words: Vec<&str> = text.split_whitespace().collect();
                    for (i, w) in words.iter().enumerate() {
                        eng.push_text(&mut u, &format!("{w} "), i + 1 == words.len());
                    }
                    events += &format!("say\t{t:.3}\t{text}\n");
                    cur = Some((u, Playout::new(block, fade), t));
                }
            }
        }
        // one device block: speech (just-in-time rendering) + reflex
        let mut blk = vec![0.0f32; block];
        if let Some((u, po, _)) = cur.as_mut() {
            while !u.cancelled && po.buf.len() < po.played + block + HOP * eng.cfg.chunk {
                let before = u.audio.len();
                if eng.pump_n(u, 1)?.is_empty() {
                    break;
                }
                po.push(&u.audio[before..]);
            }
            if let Some(b) = po.next_block() {
                blk[..b.len()].copy_from_slice(&b);
            }
        }
        if let Some((x, p)) = refl.as_mut() {
            let n = (x.len() - *p).min(block);
            for i in 0..n {
                blk[i] += x[*p + i];
            }
            *p += n;
            if *p >= x.len() {
                refl = None;
            }
        }
        out.extend_from_slice(&blk);
        let idle = cur.is_none() && refl.is_none() && (queue.is_empty() || fallen);
        if idle && (fired >= tl.reflexes.len() || t > t_end) {
            break;
        }
        if t > t_end + 60.0 {
            break;
        }
    }
    let secs = out.len() as f64 / SR as f64;
    ribt::data::write_wav(&out_dir.join("story.wav"), &out, SR as u32)?;
    std::fs::write(out_dir.join("events.tsv"), &events)?;
    let mut f = std::fs::File::create(out_dir.join("timeline.tsv"))?;
    writeln!(f, "t\teffort\theat\tfatigue\tbreathless\tslip\t{}\tpitch_st\tenergy_db\tpace\trange\tcatch_st\ttremor_st\ttilt\tbreathy\tpress\teffort_w\theat_c\ttilt_rate\tsoc", persona::MOODS.join("\t"))?;
    for s in tl.s.iter().filter(|s| s.t <= secs) {
        let st = persona::style_of(s, drama);
        let m: Vec<String> = s.mood.iter().map(|v| format!("{v:.3}")).collect();
        let o = |v: Option<f32>| v.map_or("nan".to_string(), |x| format!("{x:.2}"));
        writeln!(
            f,
            "{:.2}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{:.2}\t{:.2}\t{:.3}\t{:.3}\t{:.2}\t{:.2}\t{:.2}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}",
            s.t, s.effort, s.heat, s.fatigue, s.breathless, s.slip, m.join("\t"), st.tok.pitch_st, st.tok.energy_db, st.tok.pace, st.fx.range, st.fx.f0_st, st.fx.tremor_st, st.fx.tilt, st.fx.breathy, st.fx.press,
            o(s.raw.effort_w), o(s.raw.heat_c), o(s.raw.tilt_rate), o(s.raw.soc)
        )?;
    }
    let mut l0: Vec<f64> = lat.iter().map(|l| l.0).collect();
    let mut l1: Vec<f64> = lat.iter().map(|l| l.1).collect();
    l0.sort_by(|a, b| a.partial_cmp(b).unwrap());
    l1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let md = |v: &[f64]| v.get(v.len() / 2).copied().unwrap_or(f64::NAN);
    let mx = |v: &[f64]| v.last().copied().unwrap_or(f64::NAN);
    let line = format!(
        "story: {secs:.1} s, {} reflexes; sound after the bus sample median {:.1} max {:.1} ms; after the physical onset (50 Hz bus) median {:.1} max {:.1} ms; reflex clips prerendered in {clip_ms:.0} ms at startup",
        lat.len(),
        md(&l0),
        mx(&l0),
        md(&l1),
        mx(&l1)
    );
    eprintln!("{line}");
    std::fs::write(out_dir.join("summary.txt"), line + "\n")?;
    Ok(())
}

/// `ribt-tts surprise`: run the body timeline both ways on a trace, no models (§28.12).
///
///   ribt-tts surprise --body-tsv TRACK [--loop N] [--shoves T,T,..] [--shove-rad 0.9]
///                     [--out FILE.tsv]
///
/// `--loop` repeats the trace end to end (a longer walk); `--shoves` adds a
/// 0.2 s tilt-rate pulse at each time. Prints every reflex under the fixed
/// thresholds (v1) and under the surprise tower, and writes the per-step
/// tower state.
pub fn surprise(a: &Args) -> Result<()> {
    let base = BodyTrack::load_tsv(&std::fs::read_to_string(a.need("--body-tsv")?)?);
    let n: usize = a.get("--loop", 1);
    let span = base.t.last().copied().unwrap_or(0.0) - base.t[0] + 0.02;
    let mut tr = BodyTrack::default();
    for k in 0..n {
        for (t, s) in base.t.iter().zip(&base.s) {
            tr.t.push(t + k as f64 * span);
            tr.s.push(*s);
        }
    }
    let shoves: Vec<f64> = a.opt("--shoves").map_or(vec![], |s| s.split(',').filter_map(|x| x.parse().ok()).collect());
    let mag: f32 = a.get("--shove-rad", 0.9);
    for (t, s) in tr.t.iter().zip(tr.s.iter_mut()) {
        if shoves.iter().any(|&x| *t >= x && *t < x + 0.2) {
            s.tilt_rate = Some(s.tilt_rate.unwrap_or(0.0) + mag);
        }
    }
    let v1 = Timeline::build(&tr, 0.0);
    let st = Timeline::build_with(&tr, 0.0, true);
    println!("trace {:.1} s, {} shoves", tr.t.last().unwrap() - tr.t[0], shoves.len());
    for (name, tl) in [("v1", &v1), ("surprise", &st)] {
        let r: Vec<String> = tl.reflexes.iter().map(|(t, r, s)| format!("{t:.2}:{}@{s:.2}", r.name())).collect();
        println!("{name:9} reflexes {}: {}", r.len(), r.join(" "));
    }
    for (k, b) in ribt::surprise::BANDS.iter().enumerate() {
        let f: Vec<String> = st.s.iter().filter(|s| s.surprise.fired[k]).map(|s| format!("{:.2}@{:.2}", s.t, s.surprise.amp[k])).collect();
        println!("band {b:4} fires {}: {}", f.len(), f.join(" "));
    }
    let peak = |tl: &Timeline, i: usize| tl.s.iter().map(|s| s.mood[i]).fold(0.0f32, f32::max);
    println!("peak surprised v1 {:.2} tower {:.2}; peak worried v1 {:.2} tower {:.2}", peak(&v1, 0), peak(&st, 0), peak(&v1, 6), peak(&st, 6));
    if let Some(p) = a.opt("--out") {
        let mut f = std::fs::File::create(p)?;
        writeln!(f, "t\ttilt\ts_fast\ts_mid\ts_slow\tamp_fast\tamp_mid\tamp_slow\th_fast\tsurprised\tworried")?;
        for s in &st.s {
            let o = s.surprise;
            writeln!(f, "{:.3}\t{:.4}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}", s.t, s.raw.tilt_rate.unwrap_or(f32::NAN), o.s[0], o.s[1], o.s[2], o.amp[0], o.amp[1], o.amp[2], o.h[0], s.mood[0], s.mood[6])?;
        }
    }
    Ok(())
}
