//! Greedy averaged-perceptron POS tagger (Penn tags, spaCy's label set).
//!
//! The weights are a *distillation* of spaCy en_core_web_sm: trained by
//! `ribt lilt train-tagger` on spaCy's own tags over lilt's corpus
//! (WordNet glosses + this repo's docs), never on a treebank. See
//! the ribt paper for the licensing reasoning.
//!
//! Features follow Honnibal's tagger (word, affixes, shape, two previous
//! predicted tags, ±2 words). Lookahead is exactly two tokens: the tag of
//! token i is a function of tokens ≤ i+2 and tags < i, which is what lets
//! the streaming phonemizer decide when a word is final.

use std::collections::HashMap;
use std::sync::OnceLock;

static MODEL_BIN: &[u8] = include_bytes!("../../data/lilt/tagger.bin");

pub const LOOKAHEAD: usize = 2;

fn fnv(parts: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            h ^= 0x1f;
            h = h.wrapping_mul(0x100000001b3);
        }
        for b in p.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn norm(w: &str) -> String {
    if w.chars().all(|c| c.is_ascii_digit()) {
        return if w.len() == 4 { "!YEAR".into() } else { "!DIGITS".into() };
    }
    if w.chars().any(|c| c.is_ascii_digit()) && w.chars().all(|c| c.is_ascii_digit() || matches!(c, ',' | '.' | ':' | '/' | '-')) {
        return "!NUM".into();
    }
    w.to_lowercase()
}

fn shape(w: &str) -> String {
    let mut s = String::new();
    for c in w.chars() {
        let k = if c.is_uppercase() {
            'X'
        } else if c.is_lowercase() {
            'x'
        } else if c.is_ascii_digit() {
            'd'
        } else {
            c
        };
        if !s.ends_with(k) || s.chars().rev().take_while(|&x| x == k).count() < 2 {
            s.push(k);
        }
    }
    s
}

fn suffix(w: &str, n: usize) -> &str {
    let k = w.char_indices().rev().nth(n.saturating_sub(1)).map_or(0, |x| x.0);
    &w[k..]
}

fn prefix1(w: &str) -> &str {
    w.char_indices().nth(1).map_or(w, |(i, _)| &w[..i])
}

/// Feature hashes for position i. `words` are raw token texts, `norms`
/// their normalised forms; `p1`/`p2` the previous two predicted tags.
pub fn features(words: &[&str], norms: &[String], i: usize, p1: &str, p2: &str, out: &mut Vec<u64>) {
    out.clear();
    let w = norms[i].as_str();
    let get = |j: isize| -> &str {
        if j < 0 {
            "-START-"
        } else if (j as usize) < norms.len() {
            norms[j as usize].as_str()
        } else {
            "-END-"
        }
    };
    let ii = i as isize;
    let raw = words[i];
    out.push(fnv(&["bias"]));
    out.push(fnv(&["w", w]));
    out.push(fnv(&["raw", raw]));
    out.push(fnv(&["suf3", suffix(w, 3)]));
    out.push(fnv(&["suf2", suffix(w, 2)]));
    out.push(fnv(&["suf4", suffix(w, 4)]));
    out.push(fnv(&["pre1", prefix1(w)]));
    out.push(fnv(&["shape", &shape(raw)]));
    out.push(fnv(&["first", if i == 0 { "1" } else { "0" }, &shape(prefix1(raw))]));
    out.push(fnv(&["p1", p1]));
    out.push(fnv(&["p2", p2]));
    out.push(fnv(&["p1p2", p1, p2]));
    out.push(fnv(&["p1w", p1, w]));
    out.push(fnv(&["w-1", get(ii - 1)]));
    out.push(fnv(&["suf-1", suffix(get(ii - 1), 3)]));
    out.push(fnv(&["w-2", get(ii - 2)]));
    out.push(fnv(&["w+1", get(ii + 1)]));
    out.push(fnv(&["suf+1", suffix(get(ii + 1), 3)]));
    out.push(fnv(&["w+2", get(ii + 2)]));
    out.push(fnv(&["ww+1", w, get(ii + 1)]));
    out.push(fnv(&["w-1w", get(ii - 1), w]));
    let nxt_raw = words.get(i + 1).copied().unwrap_or("-END-");
    out.push(fnv(&["shape+1", &shape(prefix1(nxt_raw))]));
}

pub struct Model {
    pub tags: Vec<String>,
    /// feature hash -> (tag index, weight) list
    pub w: HashMap<u64, Vec<(u8, f32)>>,
}

impl Model {
    pub fn score(&self, feats: &[u64], scores: &mut [f32]) {
        scores.iter_mut().for_each(|s| *s = 0.0);
        for f in feats {
            if let Some(ws) = self.w.get(f) {
                for &(t, v) in ws {
                    scores[t as usize] += v;
                }
            }
        }
    }

    fn argmax(scores: &[f32]) -> usize {
        let mut b = 0;
        for (i, &s) in scores.iter().enumerate() {
            if s > scores[b] {
                b = i;
            }
        }
        b
    }

    /// Tag tokens `from..` greedily, keeping `tags[..from]` fixed.
    pub fn tag_from(&self, words: &[&str], tags: &mut Vec<usize>, from: usize) {
        tags.truncate(from);
        let norms: Vec<String> = words.iter().map(|w| norm(w)).collect();
        let mut feats = Vec::with_capacity(32);
        let mut scores = vec![0.0; self.tags.len()];
        for i in from..words.len() {
            let p1 = if i >= 1 { self.tags[tags[i - 1]].as_str() } else { "-START-" };
            let p2 = if i >= 2 { self.tags[tags[i - 2]].as_str() } else { "-START2-" };
            features(words, &norms, i, p1, p2, &mut feats);
            self.score(&feats, &mut scores);
            tags.push(Self::argmax(&scores));
        }
    }

    pub fn tag(&self, words: &[&str]) -> Vec<&str> {
        let mut t = Vec::new();
        self.tag_from(words, &mut t, 0);
        t.into_iter().map(|i| self.tags[i].as_str()).collect()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = b"LILTPOS1".to_vec();
        b.push(self.tags.len() as u8);
        for t in &self.tags {
            b.push(t.len() as u8);
            b.extend(t.as_bytes());
        }
        let mut keys: Vec<_> = self.w.keys().copied().collect();
        keys.sort();
        b.extend((keys.len() as u32).to_le_bytes());
        for k in keys {
            let ws = &self.w[&k];
            b.extend(k.to_le_bytes());
            b.push(ws.len() as u8);
            for &(t, v) in ws {
                b.push(t);
                b.extend(v.to_le_bytes());
            }
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Option<Model> {
        let mut p = 0usize;
        let mut take = |n: usize| -> Option<&[u8]> {
            let s = b.get(p..p + n)?;
            p += n;
            Some(s)
        };
        if take(8)? != b"LILTPOS1" {
            return None;
        }
        let nt = take(1)?[0] as usize;
        let mut tags = Vec::with_capacity(nt);
        for _ in 0..nt {
            let l = take(1)?[0] as usize;
            tags.push(String::from_utf8(take(l)?.to_vec()).ok()?);
        }
        let nf = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let mut w = HashMap::with_capacity(nf);
        for _ in 0..nf {
            let k = u64::from_le_bytes(take(8)?.try_into().ok()?);
            let n = take(1)?[0] as usize;
            let mut ws = Vec::with_capacity(n);
            for _ in 0..n {
                let t = take(1)?[0];
                let v = f32::from_le_bytes(take(4)?.try_into().ok()?);
                ws.push((t, v));
            }
            w.insert(k, ws);
        }
        Some(Model { tags, w })
    }
}

/// The embedded model. An empty `tagger.bin` (bootstrap, before the first
/// `train-tagger`) yields a model that tags everything `NN`.
pub fn model() -> &'static Model {
    static M: OnceLock<Model> = OnceLock::new();
    M.get_or_init(|| {
        Model::from_bytes(MODEL_BIN).unwrap_or(Model {
            tags: vec!["NN".into()],
            w: HashMap::new(),
        })
    })
}

/// Averaged-perceptron training on (words, tags) sentences.
pub fn train(sents: &[(Vec<String>, Vec<String>)], iters: usize, prune: f32) -> Model {
    let mut tags: Vec<String> = sents
        .iter()
        .flat_map(|s| s.1.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    tags.sort();
    let tix: HashMap<&str, usize> = tags.iter().enumerate().map(|(i, t)| (t.as_str(), i)).collect();
    let nt = tags.len();
    // dense per-feature rows while training
    let mut w: HashMap<u64, Vec<f32>> = HashMap::new();
    let mut tot: HashMap<u64, Vec<f64>> = HashMap::new();
    let mut ts: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut step: u64 = 0;
    let mut order: Vec<usize> = (0..sents.len()).collect();
    let mut rng: u64 = 0x9e3779b97f4a7c15;
    let mut feats = Vec::new();
    let mut scores = vec![0f32; nt];
    for it in 0..iters {
        for i in (1..order.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        let (mut c, mut n) = (0usize, 0usize);
        for &si in &order {
            let (ws, gold) = &sents[si];
            let words: Vec<&str> = ws.iter().map(String::as_str).collect();
            let norms: Vec<String> = words.iter().map(|w| norm(w)).collect();
            let mut pred: Vec<usize> = Vec::with_capacity(words.len());
            for i in 0..words.len() {
                let p1 = if i >= 1 { tags[pred[i - 1]].as_str() } else { "-START-" };
                let p2 = if i >= 2 { tags[pred[i - 2]].as_str() } else { "-START2-" };
                features(&words, &norms, i, p1, p2, &mut feats);
                scores.iter_mut().for_each(|s| *s = 0.0);
                for f in &feats {
                    if let Some(r) = w.get(f) {
                        for (s, v) in scores.iter_mut().zip(r) {
                            *s += v;
                        }
                    }
                }
                let guess = Model::argmax(&scores);
                let truth = tix[gold[i].as_str()];
                step += 1;
                if guess != truth {
                    for f in &feats {
                        let r = w.entry(*f).or_insert_with(|| vec![0.0; nt]);
                        let tt = tot.entry(*f).or_insert_with(|| vec![0.0; nt]);
                        let st = ts.entry(*f).or_insert_with(|| vec![0; nt]);
                        for (cls, d) in [(truth, 1.0f32), (guess, -1.0)] {
                            tt[cls] += (step - st[cls]) as f64 * r[cls] as f64;
                            st[cls] = step;
                            r[cls] += d;
                        }
                    }
                } else {
                    c += 1;
                }
                n += 1;
                pred.push(guess);
            }
        }
        eprintln!("  iter {it}: train acc {:.4}", c as f64 / n as f64);
    }
    // average
    let mut out = HashMap::new();
    for (f, r) in &w {
        let tt = &tot[f];
        let st = &ts[f];
        let mut row = Vec::new();
        for cls in 0..nt {
            let total = tt[cls] + (step - st[cls]) as f64 * r[cls] as f64;
            let avg = (total / step as f64) as f32;
            if avg.abs() >= prune {
                row.push((cls as u8, avg));
            }
        }
        if !row.is_empty() {
            out.insert(*f, row);
        }
    }
    Model { tags, w: out }
}
