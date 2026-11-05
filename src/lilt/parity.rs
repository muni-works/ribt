//! Golden-parity harness against misaki (see scripts/lilt_golden.py) plus
//! tagger training and a speed probe. Used by `ribt lilt ...` and the
//! `lilt_parity` integration test.

use super::g2p::{self, Ctx, MTok};
use super::{join, tagger, tokenize, Phonemizer, WordOut, UNK};
use std::collections::HashMap;
use std::time::Instant;

pub struct GWord {
    pub text: String,
    pub tag: String,
    pub ws: bool,
    /// None: misaki found nothing (espeak fallback territory)
    pub ps: Option<String>,
}

pub struct Sent {
    pub split: String,
    pub text: String,
    pub ps: String,
    /// spaCy tokens: (text, tag, ws)
    pub spacy: Vec<(String, String, bool)>,
    pub words: Vec<GWord>,
}

impl GWord {
    pub fn oov(&self) -> bool {
        self.ps.as_deref().is_none_or(|p| p.contains(UNK))
    }
}

pub fn load(path: &str) -> std::io::Result<Vec<Sent>> {
    let s = std::fs::read_to_string(path)?;
    let mut out: Vec<Sent> = Vec::new();
    for line in s.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "S" => out.push(Sent {
                split: f[1].into(),
                text: f[2].into(),
                ps: f.get(3).copied().unwrap_or("").into(),
                spacy: vec![],
                words: vec![],
            }),
            "T" => out
                .last_mut()
                .unwrap()
                .spacy
                .push((f[1].into(), f[2].into(), f.get(3).is_some_and(|w| !w.is_empty()))),
            "W" => {
                let ps = f.get(4).copied().unwrap_or("");
                out.last_mut().unwrap().words.push(GWord {
                    text: f[1].into(),
                    tag: f[2].into(),
                    ws: !f[3].is_empty(),
                    ps: if ps == "\u{1}" { None } else { Some(ps.into()) },
                })
            }
            _ => {}
        }
    }
    Ok(out)
}

/// misaki on given (spaCy) tokens and tags: isolates the G2P port from
/// the tokenizer and tagger.
pub fn phonemize_tagged(toks: &[(String, String, bool)]) -> Vec<WordOut> {
    let mt: Vec<MTok> = toks
        .iter()
        .enumerate()
        .map(|(i, (t, g, w))| MTok {
            text: t.clone(),
            tag: g.clone(),
            ws: *w,
            is_head: true,
            src: i,
            ..Default::default()
        })
        .collect();
    let mut ws = g2p::retokenize(&mt);
    g2p::resolve(&mut ws, Ctx::default());
    g2p::finalize(ws, &UNK.to_string()).iter().map(WordOut::from_mtok).collect()
}

/// Training data is spaCy's tags on the `train` split, plus typographic
/// variants the corpus barely contains (curly quotes, £/€), made by
/// rewriting sentences whose tokens spaCy tagged as quotes or currency.
pub fn train_tagger(sents: &[Sent], iters: usize) -> tagger::Model {
    let mut data: Vec<(Vec<String>, Vec<String>)> = sents
        .iter()
        .filter(|s| s.split == "train")
        .map(|s| {
            (
                s.spacy.iter().map(|t| t.0.clone()).collect(),
                s.spacy.iter().map(|t| t.1.clone()).collect(),
            )
        })
        .collect();
    let mut aug = Vec::new();
    for (k, (w, t)) in data.iter().enumerate() {
        let curly = |x: &str, tag: &str| -> Option<String> {
            Some(match (x, tag) {
                ("\"", "``") => "“".into(),
                ("\"", "''") => "”".into(),
                ("'", "``") => "‘".into(),
                ("'", "''") => "’".into(),
                ("$", "$") => {
                    if k % 2 == 0 {
                        "£".into()
                    } else {
                        "€".into()
                    }
                }
                _ => return None,
            })
        };
        if w.iter().zip(t).any(|(x, g)| curly(x, g).is_some()) {
            let w2 = w
                .iter()
                .zip(t)
                .map(|(x, g)| curly(x, g).unwrap_or_else(|| x.replace('\'', "’")))
                .collect();
            aug.push((w2, t.clone()));
        }
    }
    eprintln!("augmented {} sentences (curly quotes, £/€, ’ apostrophes)", aug.len());
    data.extend(aug);
    eprintln!(
        "training on {} sentences, {} tokens",
        data.len(),
        data.iter().map(|d| d.0.len()).sum::<usize>()
    );
    tagger::train(&data, iters, 1e-3)
}

/// Character spans of words laid out over the (normalised) sentence text.
fn spans<'a>(text: &str, words: impl Iterator<Item = &'a str>) -> Vec<Option<(usize, usize)>> {
    let mut pos = 0;
    words
        .map(|w| {
            let i = text[pos..].find(w)? + pos;
            pos = i + w.len();
            Some((i, pos))
        })
        .collect()
}

#[derive(Default)]
pub struct Report {
    pub sents: usize,
    pub sent_exact: usize,
    pub words: usize,
    pub oov: usize,
    pub word_ok: usize,
    pub oracle_words: usize,
    pub oracle_ok: usize,
    pub oracle_sent_exact: usize,
    pub tok_total: usize,
    pub tok_ok_sents: usize,
    pub tag_total: usize,
    pub tag_ok: usize,
    pub classes: HashMap<&'static str, usize>,
    pub examples: HashMap<(&'static str, String, String, String), usize>,
}

fn classify(g: &GWord, ours: Option<&WordOut>, oracle_ok: bool, tag_differs: bool, tok_same: bool) -> &'static str {
    match ours {
        None if tok_same => "POS tag changes word grouping",
        None => "tokenization (word boundary differs)",
        Some(_) if !oracle_ok => {
            if g.text.chars().any(|c| c.is_ascii_digit()) {
                "port: number"
            } else if !g.text.chars().any(|c| c.is_alphabetic()) {
                "port: punctuation"
            } else {
                "port: lexicon/stress"
            }
        }
        Some(_) if tag_differs => "POS tag differs from spaCy",
        Some(_) => "context (neighbour differs)",
    }
}

pub fn evaluate(sents: &[Sent], splits: &[&str]) -> Report {
    let mut r = Report::default();
    for s in sents.iter().filter(|s| splits.contains(&s.split.as_str())) {
        r.sents += 1;
        let (ps, ours) = super::phonemize(&s.text);
        if ps == s.ps {
            r.sent_exact += 1;
        }
        // tokenizer + tagger vs spaCy
        let toks = tokenize::tokenize(&s.text);
        r.tok_total += 1;
        let tok_same = toks.len() == s.spacy.len() && toks.iter().zip(&s.spacy).all(|(a, b)| a.text == b.0);
        if tok_same {
            r.tok_ok_sents += 1;
            let words: Vec<&str> = toks.iter().map(|t| t.text.as_str()).collect();
            for (a, b) in tagger::model().tag(&words).iter().zip(&s.spacy) {
                r.tag_total += 1;
                r.tag_ok += (*a == b.1) as usize;
            }
        }
        // oracle (spaCy tokens + tags)
        let oracle = phonemize_tagged(&s.spacy);
        if join(&oracle) == s.ps {
            r.oracle_sent_exact += 1;
        }
        let gsp = spans(&s.text, s.words.iter().map(|w| w.text.as_str()));
        let osp = spans(&s.text, ours.iter().map(|w| w.text.as_str()));
        let orsp = spans(&s.text, oracle.iter().map(|w| w.text.as_str()));
        let ours_by: HashMap<(usize, usize), &WordOut> = osp.iter().zip(&ours).filter_map(|(sp, w)| Some(((*sp)?, w))).collect();
        let oracle_by: HashMap<(usize, usize), &WordOut> = orsp.iter().zip(&oracle).filter_map(|(sp, w)| Some(((*sp)?, w))).collect();
        for (g, sp) in s.words.iter().zip(&gsp) {
            if g.oov() {
                r.oov += 1;
                continue;
            }
            let gps = g.ps.as_deref().unwrap();
            r.words += 1;
            let o = sp.and_then(|sp| ours_by.get(&sp).copied());
            let or = sp.and_then(|sp| oracle_by.get(&sp).copied());
            let or_ok = or.is_some_and(|w| w.phonemes == gps);
            r.oracle_words += 1;
            r.oracle_ok += or_ok as usize;
            if o.is_some_and(|w| w.phonemes == gps) {
                r.word_ok += 1;
                continue;
            }
            let cls = classify(g, o, or_ok, o.is_some_and(|w| w.prosody.pos != g.tag), tok_same);
            *r.classes.entry(cls).or_default() += 1;
            let ex = (
                cls,
                format!("{} [{}→{}]", g.text, g.tag, o.map_or("-".into(), |w| w.prosody.pos.clone())),
                o.map_or("∅".into(), |w| w.phonemes.clone()),
                gps.to_string(),
            );
            *r.examples.entry(ex).or_default() += 1;
        }
    }
    r
}

pub fn print_report(r: &Report, show: usize) {
    let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    println!("sentences            {}", r.sents);
    println!("sentence exact       {:.2}%  ({}/{})", pct(r.sent_exact, r.sents), r.sent_exact, r.sents);
    println!(
        "word parity (non-OOV) {:.3}%  ({}/{}; {} OOV words excluded)",
        pct(r.word_ok, r.words),
        r.word_ok,
        r.words,
        r.oov
    );
    println!("-- diagnostics --");
    println!(
        "oracle tok+tag words {:.3}%  (misaki port alone, spaCy tokens+tags given)",
        pct(r.oracle_ok, r.oracle_words)
    );
    println!("oracle sentence exact {:.2}%", pct(r.oracle_sent_exact, r.sents));
    println!("tokenizer = spaCy    {:.2}% of sentences", pct(r.tok_ok_sents, r.tok_total));
    println!(
        "tagger = spaCy       {:.2}% of tokens (on identically tokenized sentences)",
        pct(r.tag_ok, r.tag_total)
    );
    let mut cls: Vec<_> = r.classes.iter().collect();
    cls.sort_by(|a, b| b.1.cmp(a.1));
    println!("-- mismatch classes (words) --");
    for (c, n) in &cls {
        println!("  {:6}  {:5.2}%  {}", n, pct(**n, r.words), c);
    }
    let mut ex: Vec<_> = r.examples.iter().collect();
    ex.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (c, _) in &cls {
        println!("-- top: {c} --");
        for ((_, t, o, g), n) in ex.iter().filter(|e| e.0 .0 == **c).take(show) {
            println!("  {n:4}x  {t:30} ours {o:24} misaki {g}");
        }
    }
}

/// Batch and char-by-char streaming must agree exactly.
pub fn stream_check(sents: &[Sent], n: usize) -> (usize, usize) {
    let mut bad = 0;
    let mut tot = 0;
    for s in sents.iter().take(n) {
        let (batch, _) = super::phonemize(&s.text);
        let mut p = Phonemizer::new();
        let mut out = Vec::new();
        let mut buf = [0u8; 4];
        for c in s.text.chars() {
            out.extend(p.push(c.encode_utf8(&mut buf)));
        }
        out.extend(p.finish());
        tot += 1;
        if join(&out) != batch {
            bad += 1;
            if bad <= 5 {
                eprintln!("stream≠batch: {}\n  batch  {batch}\n  stream {}", s.text, join(&out));
            }
        }
    }
    (tot, bad)
}

pub struct Speed {
    pub words_per_s: f64,
    pub us_per_push_word: f64,
    pub mean_lag_words: f64,
    pub immediate_frac: f64,
}

/// Single-threaded throughput and streaming latency.
pub fn speed(sents: &[Sent]) -> Speed {
    let texts: Vec<&str> = sents.iter().map(|s| s.text.as_str()).collect();
    super::phonemize("warm up the tables."); // lexicon + tagger load
    let t0 = Instant::now();
    let mut nw = 0usize;
    for t in &texts {
        nw += super::phonemize(t).1.len();
    }
    let words_per_s = nw as f64 / t0.elapsed().as_secs_f64();
    // stream word by word (each push = one word + its space)
    let (mut pushes, mut lag_sum, mut lag_n, mut immediate) = (0usize, 0usize, 0usize, 0usize);
    let t1 = Instant::now();
    for t in &texts {
        let mut p = Phonemizer::new();
        let mut emitted = 0usize;
        let ins: Vec<&str> = t.split(' ').collect();
        for (i, w) in ins.iter().enumerate() {
            let piece = if i + 1 < ins.len() { format!("{w} ") } else { w.to_string() };
            let got: Vec<WordOut> = p.push(&piece).collect();
            pushes += 1;
            // lag: input whitespace-words so far minus output whitespace-words
            emitted += got.iter().filter(|w| w.ws).count();
            if i + 1 < ins.len() {
                lag_sum += (i + 1).saturating_sub(emitted);
                lag_n += 1;
                immediate += (emitted == i + 1) as usize;
            }
        }
        let _ = p.finish().count();
    }
    Speed {
        words_per_s,
        us_per_push_word: t1.elapsed().as_secs_f64() * 1e6 / pushes.max(1) as f64,
        mean_lag_words: lag_sum as f64 / lag_n.max(1) as f64,
        immediate_frac: immediate as f64 / lag_n.max(1) as f64,
    }
}
