//! lilt: English text → Kokoro phonemes (+ per-word prosody hints).
//!
//! A Rust port of hexgrad's misaki 0.9.4 `en.G2P` (US), golden-parity
//! tested against it. Zero runtime dependencies: the spaCy tokenizer rules,
//! a distilled averaged-perceptron POS tagger and misaki's gold/silver
//! lexicons are embedded in the binary. Out-of-vocabulary words (where
//! misaki would call espeak-ng, GPL) go down `oov`'s ladder instead: times
//! and addresses, morphology, initialisms, a letter-context G2P fitted on the
//! lexicon, spelling (§27). Only text with nothing speakable comes back `oov`.
//! See the ribt paper
//!
//! ```no_run
//! use ribt::lilt::Phonemizer;
//! let mut p = Phonemizer::new();
//! for w in p.push("I read the ") { print!("{}", w.phonemes); }
//! for w in p.push("book.") { print!("{}", w.phonemes); }
//! for w in p.finish() { print!("{}", w.phonemes); }
//! ```

pub mod accent;
pub mod g2p;
pub mod lexicon;
pub mod num;
pub mod oov;
pub mod parity;
pub mod tagger;
pub mod tokenize;
pub mod vocab;

use g2p::{Ctx, MTok, Word};

/// Per-word prosody hints for the acoustic model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prosody {
    /// Penn tag of the word (spaCy label set)
    pub pos: String,
    /// carries a primary stress mark
    pub stressed: bool,
    /// the word is punctuation that closes a clause/sentence: , ; : . ! ? — …
    pub boundary: Option<char>,
    /// read out from digits
    pub is_number: bool,
    /// no pronunciation found even by the §27 fallback (or the fallback is off)
    pub oov: bool,
}

/// One output word, in source order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordOut {
    pub text: String,
    /// Kokoro phoneme string ('' for OOV or silent tokens)
    pub phonemes: String,
    /// followed by a space in the phoneme stream
    pub ws: bool,
    pub prosody: Prosody,
}

impl WordOut {
    fn from_mtok(t: &MTok) -> WordOut {
        let oov = t.phonemes.is_none() || t.phonemes.as_deref().is_some_and(|p| p.contains(UNK));
        let phonemes = t.phonemes.as_deref().unwrap_or("").replace(UNK, "");
        let is_punct = !phonemes.is_empty() && phonemes.chars().all(|c| ",;:.!?—…\"“”()".contains(c));
        let boundary = if is_punct {
            phonemes.chars().rev().find(|c| ",;:.!?—…".contains(*c))
        } else {
            None
        };
        WordOut {
            text: t.text.clone(),
            ws: t.ws,
            prosody: Prosody {
                pos: t.tag.clone(),
                stressed: phonemes.contains(g2p::PRIMARY),
                boundary,
                is_number: t.text.chars().any(|c| c.is_ascii_digit()),
                oov,
            },
            phonemes,
        }
    }
}

pub(crate) const UNK: char = '❓';

/// Join words the way misaki does (`phonemes + whitespace`).
pub fn join(words: &[WordOut]) -> String {
    let mut s = String::new();
    for w in words {
        s.push_str(&w.phonemes);
        if w.ws {
            s.push(' ');
        }
    }
    s
}

/// One-shot phonemization (same code path as the stream).
pub fn phonemize(text: &str) -> (String, Vec<WordOut>) {
    let mut p = Phonemizer::new();
    p.feed(text);
    let out: Vec<WordOut> = p.finish().collect();
    (join(&out), out)
}

/// Streaming phonemizer. `push` text in any pieces; each call yields the
/// words that are now final. A word is held back only while something to its
/// right could still change it: its POS tag (the tagger looks two tokens
/// ahead), or the next word's first phoneme / "to"-ness (misaki's
/// right-to-left context), or an unfinished whitespace chunk. Words whose
/// pronunciation is invariant to all of those (most words) are emitted as
/// soon as the space after them arrives.
pub struct Phonemizer {
    /// text after the last whitespace (an incomplete chunk)
    buf: String,
    /// whether any text has been seen (misaki lstrips)
    started: bool,
    /// complete spaCy tokens in the window
    toks: Vec<tokenize::Tok>,
    /// global index of toks[0]
    off: usize,
    /// tags (model indices) for toks[..tags.len()] that are final
    tags: Vec<usize>,
    /// global index of the next token not yet emitted
    base: usize,
}

impl Default for Phonemizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Phonemizer {
    pub fn new() -> Self {
        Phonemizer {
            buf: String::new(),
            started: false,
            toks: Vec::new(),
            off: 0,
            tags: Vec::new(),
            base: 0,
        }
    }

    pub fn push(&mut self, text: &str) -> std::vec::IntoIter<WordOut> {
        self.feed(text);
        self.step(false).into_iter()
    }

    /// Append text without resolving anything yet.
    fn feed(&mut self, text: &str) {
        let mut text = text;
        if !self.started {
            text = text.trim_start();
            self.started = !text.is_empty();
        }
        self.buf.push_str(text);
        // split off complete chunks (followed by whitespace)
        if let Some(last_ws) = self.buf.rfind(char::is_whitespace) {
            let ws_len = self.buf[last_ws..].chars().next().unwrap().len_utf8();
            let done: String = self.buf[..last_ws + ws_len].to_string();
            self.buf = self.buf[last_ws + ws_len..].to_string();
            for chunk in done.split_whitespace() {
                for t in tokenize::tokenize_chunk(chunk) {
                    self.toks.push(tokenize::Tok { text: t, ws: false });
                }
                if let Some(t) = self.toks.last_mut() {
                    t.ws = true;
                }
            }
        }
    }

    pub fn finish(&mut self) -> std::vec::IntoIter<WordOut> {
        let rest = std::mem::take(&mut self.buf);
        if !rest.is_empty() {
            for t in tokenize::tokenize_chunk(&rest) {
                self.toks.push(tokenize::Tok { text: t, ws: false });
            }
        }
        let out = self.step(true);
        *self = Phonemizer::new();
        out.into_iter()
    }

    fn step(&mut self, eof: bool) -> Vec<WordOut> {
        let m = tagger::model();
        let n = self.toks.len();
        let words: Vec<&str> = self.toks.iter().map(|t| t.text.as_str()).collect();
        let mut tags = self.tags.clone();
        let fixed = tags.len();
        m.tag_from(&words, &mut tags, fixed);
        // tags final where the lookahead window is complete
        let n_final_tags = if eof { n } else { n.saturating_sub(tagger::LOOKAHEAD) };
        let mtoks: Vec<MTok> = self
            .toks
            .iter()
            .zip(&tags)
            .enumerate()
            .map(|(i, (t, &g))| MTok {
                text: t.text.clone(),
                tag: m.tags[g].clone(),
                ws: t.ws,
                is_head: true,
                src: self.off + i,
                ..Default::default()
            })
            .collect();
        let mut ws = g2p::retokenize(&mtoks);
        let src_last = |w: &Word| match w {
            Word::One(t) => t.src,
            Word::Group(g) => g.last().unwrap().src,
        };
        let src_first = |w: &Word| match w {
            Word::One(t) => t.src,
            Word::Group(g) => g[0].src,
        };
        // the last group may still grow if it ends the window without whitespace
        g2p::resolve(&mut ws, Ctx::default());
        let nw = ws.len();
        // stability, right to left
        let mut stable = vec![eof; nw];
        if !eof {
            // +1: retokenize reads the next token's tag (currency + CD)
            let tags_final = |w: &Word| src_last(w) + 1 < self.off + n_final_tags;
            for i in (0..nw).rev() {
                let inv = invariant(&ws[i], &mtoks, self.off);
                stable[i] = inv || (tags_final(&ws[i]) && i + 1 < nw && stable[i + 1]);
            }
        }
        let mut out = Vec::new();
        let mut new_base = self.base;
        let finals = g2p::finalize(ws.clone(), &UNK.to_string());
        for (i, w) in ws.iter().enumerate() {
            if src_first(w) < self.base {
                continue;
            }
            if !stable[i] {
                break;
            }
            let mut wo = WordOut::from_mtok(&finals[i]);
            wo.phonemes = accent::apply(&wo.phonemes);
            out.push(wo);
            new_base = src_last(w) + 1;
        }
        self.base = new_base;
        // keep finalized tags; slide the window (2 tokens of left context)
        let keep_tags = n_final_tags.min(tags.len());
        self.tags = tags[..keep_tags].to_vec();
        // keep 3 tokens before `base` (tagger words/tags, currency sign) and
        // 2 before the first token whose tag is still open
        let lo = self
            .base
            .saturating_sub(3)
            .min((self.off + self.tags.len()).saturating_sub(2))
            .max(self.off);
        let drop = lo - self.off;
        if drop > 0 {
            self.toks.drain(..drop);
            self.tags.drain(..drop.min(self.tags.len()));
            self.off = lo;
        }
        out
    }
}

/// Is a word's pronunciation independent of its tag and its right context?
/// Probed by re-running the lexicon under every tag and context that could
/// change it. Only plain alphabetic single tokens qualify.
fn invariant(w: &Word, mtoks: &[MTok], off: usize) -> bool {
    let Word::One(t) = w else { return false };
    // grouping with neighbours depends on their tags unless this word is a
    // whole whitespace chunk on its own; and it must be a real word
    let prev_ws = t.src == 0 || t.src <= off || mtoks.get(t.src - off - 1).is_some_and(|p| p.ws);
    if !t.ws || !prev_ws || !t.text.bytes().any(|b| b.is_ascii_alphabetic()) {
        return false;
    }
    // fast path: a lowercase word with a plain (untagged) entry and no
    // special-case handling cannot change with tag or context
    const SPECIAL: &[&str] = &["a", "am", "an", "by", "to", "in", "the", "vs", "used", "i"];
    if t.alias.is_none() && t.currency.is_none() && t.text.bytes().all(|b| b.is_ascii_lowercase()) && !SPECIAL.contains(&t.text.as_str()) {
        let plain = |e: Option<lexicon::Entry>| matches!(e, Some(lexicon::Entry::Plain(_)));
        let g = lexicon::gold().get(&t.text);
        if plain(g) || (g.is_none() && plain(lexicon::silver().get(&t.text))) {
            return t.src == 0 || !mtoks.get(t.src - off - 1).is_some_and(|p| matches!(p.text.as_str(), "$" | "£" | "€"));
        }
    }
    // ASCII letters only: then no tag can route it through the punctuation branch
    if t.alias.is_some() || t.currency.is_some() || t.text.is_empty() || !t.text.chars().all(|c| c.is_ascii_alphabetic() || c == '\'') {
        return false;
    }
    // a preceding currency sign could attach to this token if it were CD
    if t.src > off && mtoks.get(t.src - off - 1).is_some_and(|p| matches!(p.text.as_str(), "$" | "£" | "€")) {
        return false;
    }
    const TAGS: &[&str] = &[
        "NN", "NNS", "NNP", "VB", "VBD", "VBN", "VBP", "VBZ", "VBG", "JJ", "RB", "IN", "DT", "CD", "PRP", "TO", "CC", "MD", "UH", "ADD", "WDT",
        "WRB", "POS", "RP",
    ];
    let mut first: Option<Option<String>> = None;
    for tag in TAGS {
        for fv in [None, Some(false), Some(true)] {
            for ft in [false, true] {
                let mut probe = t.clone();
                probe.tag = (*tag).into();
                probe.phonemes = None;
                let r = g2p::lexicon_call(
                    &probe,
                    Ctx {
                        future_vowel: fv,
                        future_to: ft,
                    },
                )
                .map(|x| x.0);
                match &first {
                    None => first = Some(r),
                    Some(f) if *f != r => return false,
                    _ => {}
                }
            }
        }
    }
    true
}
