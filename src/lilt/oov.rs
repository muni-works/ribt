//! Unknown words (the ribt paper): what lilt says when misaki's
//! dictionaries have no entry. misaki hands these to espeak-ng (GPL, never
//! ported); lilt used to drop them silently. The ladder, first hit wins:
//!
//! 1. `normalise`: clock times, emails, URLs, @handles, #tags.
//! 2. morphology: a known stem plus a known affix (talk+eth, un+…, …+less),
//!    an unknown stem's plural/possessive, and known+known compounds.
//! 3. initialisms: all-caps or vowelless short tokens are spelled.
//! 4. a learned grapheme→phoneme model (`G2p`): a pruned letter-context tree
//!    (IGTree-style) fitted on lilt's own lexicon, deterministic, ~µs/word.
//! 5. spell it letter by letter. Nothing speakable is ever dropped.
//!
//! `RIBT_LILT_FALLBACK=0` (or `set_enabled(false)`) restores the old
//! behaviour; misaki parity is measured with it off, as misaki's goldens are
//! recorded with `fallback=None`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use super::g2p::{self, PRIMARY, SECONDARY};
use super::lexicon::{gold, silver, Entry};
use super::num;

static G2P_BIN: &[u8] = include_bytes!("../../data/lilt/g2p.bin");
static ON: AtomicBool = AtomicBool::new(true);

pub fn set_enabled(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    static ENV_OFF: OnceLock<bool> = OnceLock::new();
    !*ENV_OFF.get_or_init(|| std::env::var("RIBT_LILT_FALLBACK").is_ok_and(|v| v == "0")) && ON.load(Ordering::Relaxed)
}

fn is_vowel(c: char) -> bool {
    "AIOWYaiuæɑɒɔəɛɜɪʊʌᵻ".contains(c)
}

/// A dictionary word's phonemes (lexicon, overrides, misaki's -s/-ed/-ing stemming), or None.
fn known(w: &str) -> Option<String> {
    g2p::lexicon_word(w).filter(|p| !p.is_empty())
}

/// A word, or a number, read by the full ladder short of spelling.
fn word_or_number(w: &str) -> Option<String> {
    if w.is_empty() {
        return None;
    }
    if w.bytes().all(|b| b.is_ascii_digit()) {
        return number(w);
    }
    known(w).or_else(|| guess(w))
}

fn number(d: &str) -> Option<String> {
    let v: i128 = d.parse().ok()?;
    let ws = num::cardinal(v)?;
    let ps: Vec<String> = ws.iter().filter(|w| !w.is_empty() && *w != "and").filter_map(|w| known(w)).collect();
    (!ps.is_empty()).then(|| ps.join(" "))
}

fn join(parts: Vec<String>) -> Option<String> {
    let v: Vec<String> = parts.into_iter().filter(|p| !p.is_empty()).collect();
    (!v.is_empty()).then(|| v.join(" "))
}

/// Clock times, emails, URLs, handles and hashtags: whole-token patterns
/// misaki has no reading for (it sends them to espeak).
pub fn normalise(text: &str) -> Option<String> {
    if let Some(ps) = clock(text) {
        return Some(ps);
    }
    let lw = text.to_ascii_lowercase();
    let is_url = lw.starts_with("http://") || lw.starts_with("https://") || lw.starts_with("www.");
    let is_email = !is_url && text.matches('@').count() == 1 && !text.starts_with('@') && text.contains('.');
    if is_url || is_email {
        let body = lw.trim_start_matches("https://").trim_start_matches("http://").trim_end_matches('/');
        return read_address(body);
    }
    if let Some(h) = text.strip_prefix('@').filter(|h| !h.is_empty() && h.chars().all(|c| c.is_alphanumeric() || c == '_')) {
        return join(vec![known("at")?, read_address(h)?]);
    }
    if let Some(h) = text.strip_prefix('#').filter(|h| !h.is_empty() && h.chars().all(char::is_alphanumeric)) {
        return join(vec![known("hashtag").or_else(|| known("hash"))?, read_address(h)?]);
    }
    None
}

/// H:MM (and H:MM:SS): "four fifteen", "four oh five", "twelve o'clock".
fn clock(t: &str) -> Option<String> {
    let parts: Vec<&str> = t.split(':').collect();
    if !(2..=3).contains(&parts.len()) || !parts.iter().all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())) {
        return None;
    }
    let h: u32 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    if parts[0].len() > 2 || parts[1].len() != 2 || h > 24 || m > 59 {
        return None;
    }
    let mut out = vec![number(&h.to_string())?];
    if m == 0 {
        out.push(known("o'clock")?);
    } else if m < 10 {
        out.push(known("oh")?);
        out.push(number(&m.to_string())?);
    } else {
        out.push(number(&m.to_string())?);
    }
    if let Some(s) = parts.get(2) {
        let s: u32 = s.parse().ok()?;
        out.push(known("and")?);
        out.push(number(&s.to_string())?);
        out.push(known(if s == 1 { "second" } else { "seconds" })?);
    }
    join(out)
}

/// host.tld/path, user@host: words between separators, separators named.
fn read_address(s: &str) -> Option<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.is_empty() {
            // digit runs and letter runs are read separately (b2b, x86)
            let mut run = String::new();
            for c in cur.chars() {
                if !run.is_empty() && run.chars().last().unwrap().is_ascii_digit() != c.is_ascii_digit() {
                    out.extend(word_or_number(&run).or_else(|| spell(&run)));
                    run.clear();
                }
                run.push(c);
            }
            out.extend(word_or_number(&run).or_else(|| spell(&run)));
            cur.clear();
        }
    };
    for c in s.chars() {
        let sep = match c {
            '.' => Some("dot"),
            '/' => Some("slash"),
            '@' => Some("at"),
            '-' => Some("dash"),
            '_' => Some("underscore"),
            ':' => Some("colon"),
            '?' | '=' | '&' | '#' | '~' | '+' | '%' => Some(""),
            _ => None,
        };
        match sep {
            Some(w) => {
                flush(&mut cur, &mut out);
                if !w.is_empty() {
                    out.extend(known(w));
                }
            }
            None => cur.push(c),
        }
    }
    flush(&mut cur, &mut out);
    join(out)
}

// ---------------------------------------------------------------- ladder

/// Phonemes for a token that every dictionary missed: morphology, then
/// initialism, then the learned G2P, then spelling. `None` only when the
/// token has nothing speakable in it (an emoji, a stray symbol).
pub fn guess(word: &str) -> Option<String> {
    // ASCII letters (accents folded, case kept), digits and apostrophes only
    let w: String = word
        .chars()
        .filter_map(|c| match c {
            '’' | '\'' => Some('\''),
            _ if c.is_ascii_digit() => Some(c),
            _ => deaccent(c).map(|b| if c.is_uppercase() { b.to_ascii_uppercase() } else { b }),
        })
        .collect();
    if w.trim_matches('\'').is_empty() {
        return spell(word);
    }
    if w.bytes().any(|b| b.is_ascii_digit()) {
        // digits and letters misaki did not split (a1b2): read each run
        let mut runs: Vec<String> = Vec::new();
        for c in w.chars().filter(|&c| c != '\'') {
            match runs.last_mut() {
                Some(r) if r.chars().last().unwrap().is_ascii_digit() == c.is_ascii_digit() => r.push(c),
                _ => runs.push(c.to_string()),
            }
        }
        return join(runs.iter().filter_map(|r| if r.as_bytes()[0].is_ascii_digit() { number(r).or_else(|| spell(r)) } else { known(r).or_else(|| guess(r)) }).collect());
    }
    if initialism(&w) {
        return spell(&w);
    }
    let parts = camel(&w);
    if parts.len() > 1 {
        return join(parts.iter().filter_map(|p| known(p).or_else(|| guess(p))).collect());
    }
    let lw = w.to_ascii_lowercase();
    if let Some(ps) = morph(&lw, 2) {
        return Some(ps);
    }
    if let Some(ps) = model().map(|m| m.predict(&lw)).filter(|p| !p.is_empty()) {
        return Some(ps);
    }
    spell(&w)
}

/// Case-boundary pieces: "STFTNet" → STFT, Net; "kNN" → k, NN. Title case is one piece.
fn camel(w: &str) -> Vec<String> {
    let cs: Vec<char> = w.chars().collect();
    let mut out = vec![String::new()];
    for i in 0..cs.len() {
        let up = cs[i].is_uppercase();
        let brk = i > 0 && up && (cs[i - 1].is_lowercase() || cs.get(i + 1).is_some_and(|c| c.is_lowercase()));
        if brk && !out.last().unwrap().is_empty() {
            out.push(String::new());
        }
        out.last_mut().unwrap().push(cs[i]);
    }
    out
}

/// A token with no letter or digit that nothing could read (a stray `^`, `?`
/// or bracket inside a run) is silent, not unknown: no word was lost.
pub fn silent(text: &str) -> Option<String> {
    (!text.chars().any(char::is_alphanumeric)).then(String::new)
}

/// All caps (2–5 letters), or no vowel letter at all, and short: say the letters.
fn initialism(w: &str) -> bool {
    let n = w.chars().filter(|c| c.is_alphabetic()).count();
    let caps = (2..=5).contains(&n) && w.chars().all(|c| !c.is_alphabetic() || c.is_uppercase());
    let vowelless = (2..=5).contains(&n) && !w.to_ascii_lowercase().chars().any(|c| "aeiouy".contains(c));
    caps || vowelless
}

const SUFFIXES: &[(&str, &str)] = &[
    ("fulness", "fəlnəs"),
    ("lessness", "ləsnəs"),
    ("ment", "mənt"),
    ("ness", "nəs"),
    ("less", "ləs"),
    ("ful", "fəl"),
    ("eth", "ɪθ"),
    ("est", "ɪst"),
    ("ish", "ɪʃ"),
    ("ify", "ɪfˌI"),
    ("ism", "ˌɪzəm"),
    ("ist", "ɪst"),
    ("ize", "ˌIz"),
    ("able", "əbᵊl"),
    ("ly", "li"),
    ("er", "əɹ"),
];
const PREFIXES: &[(&str, &str)] = &[
    ("over", "ˌOvəɹ"),
    ("under", "ˌʌndəɹ"),
    ("non", "nˌɑn"),
    ("pre", "pɹˌi"),
    ("mis", "mˌɪs"),
    ("dis", "dˌɪs"),
    ("un", "ˌʌn"),
    ("re", "ɹˌi"),
];

fn demote(ps: &str) -> String {
    ps.replace(PRIMARY, &SECONDARY.to_string())
}

/// A known word inside affixes, or two known words run together.
fn morph(w: &str, depth: u8) -> Option<String> {
    let n = w.len();
    let stem_known = |s: &str| -> Option<String> {
        if s.len() < 3 {
            return None;
        }
        known(s).or_else(|| if depth > 1 { morph(s, depth - 1) } else { None })
    };
    // possessive / plural of something the ladder can read
    if let Some(s) = w.strip_suffix("'s").or_else(|| w.strip_suffix('\'')) {
        if let Some(st) = known(s).or_else(|| guess(s)) {
            return g2p::s_suffix(&st);
        }
    }
    for (suf, sp) in SUFFIXES {
        let Some(s) = w.strip_suffix(suf) else { continue };
        // talk+eth, bamboozl(e)+ment, happ(y→i)+ness, stop(p)+er
        let mut cands = vec![s.to_string(), format!("{s}e")];
        if let Some(t) = s.strip_suffix('i') {
            cands.push(format!("{t}y"));
        }
        let b = s.as_bytes();
        if b.len() >= 2 && b[b.len() - 1] == b[b.len() - 2] {
            cands.push(s[..s.len() - 1].to_string());
        }
        if let Some(st) = cands.iter().find_map(|c| stem_known(c)) {
            return Some(format!("{st}{sp}"));
        }
    }
    for (pre, pp) in PREFIXES {
        if let Some(st) = w.strip_prefix(pre).and_then(&stem_known) {
            return Some(format!("{pp}{st}"));
        }
    }
    // compounds: both halves in gold (silver is too permissive: "hob"+"son"),
    // at least 3 letters each; the longest known head wins
    if depth > 1 {
        for i in (3..=n.saturating_sub(3)).rev() {
            let (a, b) = (&w[..i], &w[i..]);
            let in_gold = |x: &str| gold().contains(x) || super::lexicon::override_get(x).is_some();
            if !in_gold(a) || !(in_gold(b) || b.strip_suffix('s').is_some_and(in_gold)) {
                continue;
            }
            if let (Some(pa), Some(pb)) = (known(a), known(b)) {
                return Some(format!("{pa}{}", demote(&pb)));
            }
        }
    }
    None
}

/// Letter by letter: letter names, digits as numbers, a few symbol names.
pub fn spell(w: &str) -> Option<String> {
    let mut out: Vec<String> = Vec::new();
    let mut letters = String::new();
    let mut digits = String::new();
    let letter = |c: char| match gold().get(&c.to_uppercase().to_string()) {
        Some(Entry::Plain(p)) => Some(p.to_string()),
        _ => None,
    };
    let flush_l = |letters: &mut String, out: &mut Vec<String>| {
        if !letters.is_empty() {
            let ps = g2p::apply_stress(letters, Some(0.0));
            out.push(match ps.rfind(SECONDARY) {
                Some(i) => format!("{}{}{}", &ps[..i], PRIMARY, &ps[i + SECONDARY.len_utf8()..]),
                None => ps,
            });
            letters.clear();
        }
    };
    for c in w.chars() {
        if c.is_ascii_digit() {
            flush_l(&mut letters, &mut out);
            digits.push(c);
            continue;
        }
        if !digits.is_empty() {
            out.extend(number(&digits));
            digits.clear();
        }
        let base = deaccent(c);
        if let Some(p) = base.and_then(letter) {
            letters.push_str(&p);
            continue;
        }
        flush_l(&mut letters, &mut out);
        if let Some(g) = greek(c) {
            out.extend(known(g));
            continue;
        }
        let sym = match c {
            '&' => "and",
            '+' => "plus",
            '@' => "at",
            '%' => "percent",
            '#' => "hash",
            '=' => "equals",
            '/' => "slash",
            '*' => "star",
            '$' => "dollar",
            '<' => "less",
            '>' => "greater",
            '~' => "tilde",
            _ => "",
        };
        if !sym.is_empty() {
            out.extend(known(sym));
        }
    }
    flush_l(&mut letters, &mut out);
    if !digits.is_empty() {
        out.extend(number(&digits));
    }
    join(out)
}

/// Greek letter names (σ, Σ, μ… show up in technical text).
fn greek(c: char) -> Option<&'static str> {
    const N: [&str; 24] = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau",
        "upsilon", "phi", "chi", "psi", "omega",
    ];
    let l = c.to_lowercase().next()?;
    let i = match l {
        'α'..='ρ' => l as u32 - 'α' as u32,
        'ς' => 17,
        'σ'..='ω' => l as u32 - 'α' as u32 - 1,
        _ => return None,
    };
    N.get(i as usize).copied()
}

/// The ASCII letter under a Latin accented letter (é→e, ñ→n, ø→o).
fn deaccent(c: char) -> Option<char> {
    if c.is_ascii_alphabetic() {
        return Some(c);
    }
    let l = c.to_lowercase().next()?;
    let b = match l {
        'à'..='å' | 'ā' | 'ă' | 'ą' => 'a',
        'æ' => 'a',
        'ç' | 'ć' | 'č' => 'c',
        'ď' | 'đ' | 'ð' => 'd',
        'è'..='ë' | 'ē' | 'ė' | 'ę' | 'ě' => 'e',
        'ì'..='ï' | 'ī' | 'į' | 'ı' => 'i',
        'ł' | 'ľ' => 'l',
        'ñ' | 'ń' | 'ň' => 'n',
        'ò'..='ö' | 'ø' | 'ō' | 'ő' => 'o',
        'ř' => 'r',
        'ś' | 'š' | 'ş' | 'ß' => 's',
        'ť' | 'ţ' | 'þ' => 't',
        'ù'..='ü' | 'ū' | 'ů' | 'ű' => 'u',
        'ý' | 'ÿ' => 'y',
        'ź' | 'ż' | 'ž' => 'z',
        _ => return None,
    };
    Some(b)
}

// ---------------------------------------------------------------- learned G2P

/// Letters of context each side, at most 5 so a key packs into 60 bits.
pub const MAX_HALF: usize = 5;

fn code(c: u8) -> u64 {
    match c {
        b'a'..=b'z' => (c - b'a' + 1) as u64,
        b'\'' => 27,
        _ => 0,
    }
}

/// Features, nearest first: c0, r1, l1, r2, l2, … (± `half`).
fn feats(w: &[u8], i: usize, half: usize) -> Vec<u64> {
    let at = |j: isize| if j < 0 || j as usize >= w.len() { 0 } else { code(w[j as usize]) };
    let ii = i as isize;
    let mut f = vec![at(ii)];
    for k in 1..=half as isize {
        f.extend([at(ii + k), at(ii - k)]);
    }
    f
}

fn key(f: &[u64], d: usize) -> u64 {
    let mut k = (d as u64) << 60;
    for (j, x) in f.iter().take(d + 1).enumerate() {
        k |= x << (5 * j);
    }
    k
}

/// A pruned letter-context tree: each letter's phoneme chunk is the class of
/// the deepest stored context that matches. A node is stored only when it
/// disagrees with its parent or has a descendant that does (IGTree).
pub struct G2p {
    half: usize,
    classes: Vec<String>,
    nodes: Vec<(u64, u16)>,
}

impl G2p {
    pub fn predict(&self, w: &str) -> String {
        let b = w.as_bytes();
        // (chunk, depth of the context that chose it)
        let mut chunks: Vec<(&str, usize)> = Vec::with_capacity(b.len());
        // per letter: the deepest non-empty class on its context path (for the digraph back-off)
        let mut loud: Vec<Option<(&str, usize)>> = Vec::with_capacity(b.len());
        for i in 0..b.len() {
            let f = feats(b, i, self.half);
            let mut cls = None;
            let mut nz = None;
            for d in 0..f.len() {
                match self.nodes.binary_search_by_key(&key(&f, d), |n| n.0) {
                    Ok(j) => {
                        cls = Some((self.nodes[j].1, d));
                        let c = self.classes[self.nodes[j].1 as usize].as_str();
                        if !c.is_empty() {
                            nz = Some((c, d));
                        }
                    }
                    Err(_) => break,
                }
            }
            if let Some((c, d)) = cls {
                chunks.push((&self.classes[c as usize], d));
                loud.push(nz);
            }
        }
        // §29.2: each letter picks its chunk alone, so a consonant digraph can lose both halves
        // ("Lacheneur": c → "" because h carries ʧ, h → "" because c carries ʃ). When two adjacent
        // c/s/t/p + h letters are both silent, the one with the more specific non-empty context speaks.
        // (Only these: "gh" really is silent in "night"; the general consonant rule cost 1.2 pt held-out.)
        if chunks.len() == b.len() {
            for i in 0..b.len().saturating_sub(1) {
                if chunks[i].0.is_empty() && chunks[i + 1].0.is_empty() && b"cstp".contains(&b[i]) && b[i + 1] == b'h' {
                    let j = match (loud[i], loud[i + 1]) {
                        (Some((_, da)), Some((_, db))) => if db >= da { i + 1 } else { i },
                        (None, Some(_)) => i + 1,
                        (Some(_), None) => i,
                        (None, None) => continue,
                    };
                    chunks[j] = loud[j].unwrap();
                }
            }
        }
        // several primaries: keep the one chosen by the most specific context
        let best = chunks.iter().enumerate().filter(|(_, (c, _))| c.contains(PRIMARY)).max_by_key(|(i, (_, d))| (*d, usize::MAX - i)).map(|x| x.0);
        let mut out = String::new();
        for (i, (c, _)) in chunks.iter().enumerate() {
            let c = if best.is_some_and(|b| b != i) { c.replace(PRIMARY, &SECONDARY.to_string()) } else { c.to_string() };
            // one letter per sound: "ng-i" must not become ŋŋ
            let dup = c.chars().next().is_some_and(|f| !is_vowel(f) && f != PRIMARY && f != SECONDARY && out.ends_with(f));
            out.push_str(if dup { &c[c.chars().next().unwrap().len_utf8()..] } else { &c });
        }
        // misaki writes a stressed schwa as ʌ
        fix_stress(&out).replace("ˈə", "ˈʌ").replace("ˌə", "ˌʌ")
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = b"LG2P2".to_vec();
        b.push(self.half as u8);
        b.extend((self.classes.len() as u32).to_le_bytes());
        for c in &self.classes {
            b.push(c.len() as u8);
            b.extend(c.as_bytes());
        }
        b.extend((self.nodes.len() as u32).to_le_bytes());
        for (k, c) in &self.nodes {
            b.extend(k.to_le_bytes());
            b.extend(c.to_le_bytes());
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Option<G2p> {
        let (&half, rest) = b.strip_prefix(b"LG2P2")?.split_first()?;
        let half = half as usize;
        let u32at = |b: &[u8], i: usize| -> Option<u32> { Some(u32::from_le_bytes(b.get(i..i + 4)?.try_into().ok()?)) };
        let s8 = |b: &[u8], p: &mut usize| -> Option<String> {
            let l = *b.get(*p)? as usize;
            let s = std::str::from_utf8(b.get(*p + 1..*p + 1 + l)?).ok()?.to_string();
            *p += 1 + l;
            Some(s)
        };
        let nc = u32at(rest, 0)? as usize;
        let mut p = 4;
        let classes = (0..nc).map(|_| s8(rest, &mut p)).collect::<Option<Vec<_>>>()?;
        let nn = u32at(rest, p)? as usize;
        p += 4;
        let mut nodes = Vec::with_capacity(nn);
        for i in 0..nn {
            let q = p + i * 10;
            let k = u64::from_le_bytes(rest.get(q..q + 8)?.try_into().ok()?);
            let c = u16::from_le_bytes(rest.get(q + 8..q + 10)?.try_into().ok()?);
            nodes.push((k, c));
        }
        Some(G2p { half, classes, nodes })
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

pub fn model() -> Option<&'static G2p> {
    static M: OnceLock<Option<G2p>> = OnceLock::new();
    M.get_or_init(|| G2p::from_bytes(G2P_BIN)).as_ref()
}

/// Exactly one primary stress: keep the first ˈ (demote the rest), or promote
/// the first ˌ, or stress the first vowel.
fn fix_stress(ps: &str) -> String {
    let mut out = String::with_capacity(ps.len());
    let mut seen = false;
    for c in ps.chars() {
        if c == PRIMARY {
            out.push(if seen { SECONDARY } else { PRIMARY });
            seen = true;
        } else {
            out.push(c);
        }
    }
    if seen {
        return out;
    }
    if let Some(i) = out.find(SECONDARY) {
        return format!("{}{}{}", &out[..i], PRIMARY, &out[i + SECONDARY.len_utf8()..]);
    }
    match out.char_indices().find(|(_, c)| is_vowel(*c)) {
        Some((i, _)) => format!("{}{}{}", &out[..i], PRIMARY, &out[i..]),
        None => out,
    }
}

// ---------------------------------------------------------------- fitting

/// Phoneme units: a stress mark rides on the symbol after it.
fn units(ps: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut pend = String::new();
    for c in ps.chars() {
        if c == PRIMARY || c == SECONDARY {
            pend.push(c);
        } else {
            pend.push(c);
            out.push(std::mem::take(&mut pend));
        }
    }
    out
}

fn held_out(w: &str) -> bool {
    // FNV-1a; 1 word in 20
    let h = w.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    h % 20 == 0
}

/// Lowercase alphabetic lexicon words with a plain pronunciation, gold before silver.
pub fn lexicon_pairs() -> Vec<(String, String)> {
    let mut m: std::collections::BTreeMap<String, String> = Default::default();
    for d in [silver(), gold()] {
        for (k, v) in d.rows() {
            if k.len() >= 2 && k.bytes().all(|b| b.is_ascii_lowercase() || b == b'\'') && !v.starts_with('{') && !v.contains(' ') && !v.is_empty() {
                m.insert(k.to_string(), v.to_string());
            }
        }
    }
    m.into_iter().collect()
}

/// Viterbi-EM alignment: each letter emits 0, 1 or 2 phoneme units.
fn align(data: &[(Vec<u8>, Vec<String>)], iters: usize) -> Vec<Vec<String>> {
    let mut logp: HashMap<(u8, String), f64> = HashMap::new();
    let mut aligned: Vec<Vec<String>> = Vec::new();
    for it in 0..iters {
        let lp = |l: u8, ch: &str, k: usize| -> f64 {
            if it == 0 {
                return if k == 1 { 0.0 } else { -1.0 };
            }
            *logp.get(&(l, ch.to_string())).unwrap_or(&-30.0)
        };
        let mut counts: HashMap<(u8, String), f64> = HashMap::new();
        aligned.clear();
        for (w, u) in data {
            let (n, m) = (w.len(), u.len());
            let mut dp = vec![vec![f64::NEG_INFINITY; m + 1]; n + 1];
            let mut bk = vec![vec![0usize; m + 1]; n + 1];
            dp[0][0] = 0.0;
            for i in 0..n {
                for j in 0..=m {
                    if dp[i][j] == f64::NEG_INFINITY {
                        continue;
                    }
                    for k in 0..=2 {
                        if j + k > m {
                            break;
                        }
                        let s = dp[i][j] + lp(w[i], &u[j..j + k].concat(), k);
                        if s > dp[i + 1][j + k] {
                            dp[i + 1][j + k] = s;
                            bk[i + 1][j + k] = k;
                        }
                    }
                }
            }
            if dp[n][m] == f64::NEG_INFINITY {
                aligned.push(Vec::new());
                continue;
            }
            let mut chunks = vec![String::new(); n];
            let mut j = m;
            for i in (1..=n).rev() {
                let k = bk[i][j];
                chunks[i - 1] = u[j - k..j].concat();
                j -= k;
            }
            for (i, ch) in chunks.iter().enumerate() {
                *counts.entry((w[i], ch.clone())).or_default() += 1.0;
            }
            aligned.push(chunks);
        }
        let mut tot: HashMap<u8, f64> = HashMap::new();
        for ((l, _), c) in &counts {
            *tot.entry(*l).or_default() += c;
        }
        logp = counts.into_iter().map(|((l, ch), c)| ((l, ch), (c / tot[&l]).ln())).collect();
    }
    aligned
}

/// Fit the tree on `pairs`. `min_count`: contexts seen fewer times are pruned.
pub fn fit(pairs: &[(String, String)], min_count: u32, iters: usize, half: usize) -> G2p {
    let half = half.clamp(1, MAX_HALF);
    let depth = 2 * half + 1;
    let data: Vec<(Vec<u8>, Vec<String>)> = pairs.iter().map(|(w, p)| (w.as_bytes().to_vec(), units(p))).filter(|(w, u)| u.len() <= 2 * w.len()).collect();
    let al = align(&data, iters);
    let mut cls_id: HashMap<String, u16> = HashMap::new();
    let mut classes: Vec<String> = Vec::new();
    // per depth: (key, class) occurrences
    let mut occ: Vec<Vec<(u64, u16)>> = vec![Vec::new(); depth];
    for ((w, _), chunks) in data.iter().zip(&al) {
        if chunks.is_empty() {
            continue;
        }
        for (i, ch) in chunks.iter().enumerate() {
            let c = *cls_id.entry(ch.clone()).or_insert_with(|| {
                classes.push(ch.clone());
                (classes.len() - 1) as u16
            });
            let f = feats(w, i, half);
            for (d, o) in occ.iter_mut().enumerate() {
                o.push((key(&f, d), c));
            }
        }
    }
    // majority class per key and depth
    let mut maj: Vec<HashMap<u64, u16>> = Vec::with_capacity(depth);
    for o in occ.iter_mut() {
        o.sort_unstable();
        let mut m = HashMap::new();
        let mut i = 0;
        while i < o.len() {
            let k = o[i].0;
            let (mut best, mut bc, mut tot) = (0u16, 0u32, 0u32);
            let mut j = i;
            while j < o.len() && o[j].0 == k {
                let c = o[j].1;
                let mut e = j;
                while e < o.len() && o[e] == (k, c) {
                    e += 1;
                }
                let n = (e - j) as u32;
                tot += n;
                if n > bc {
                    (best, bc) = (c, n);
                }
                j = e;
            }
            if tot >= min_count {
                m.insert(k, best);
            }
            i = j;
        }
        maj.push(m);
        o.clear();
        o.shrink_to_fit();
    }
    let parent = |k: u64, d: usize| -> u64 { (((d - 1) as u64) << 60) | (k & ((1u64 << (5 * d)) - 1)) };
    // keep a node if it disagrees with its parent or a kept child needs it
    let mut keep: Vec<std::collections::HashSet<u64>> = vec![Default::default(); depth];
    for d in (1..depth).rev() {
        let kd: Vec<u64> = maj[d].keys().copied().collect();
        for k in kd {
            let p = parent(k, d);
            let Some(&pc) = maj[d - 1].get(&p) else { continue };
            if keep[d].contains(&k) || maj[d][&k] != pc {
                keep[d].insert(k);
                keep[d - 1].insert(p);
            }
        }
    }
    let mut nodes: Vec<(u64, u16)> = maj[0].iter().map(|(k, c)| (*k, *c)).collect();
    for d in 1..depth {
        nodes.extend(keep[d].iter().map(|k| (*k, maj[d][k])));
    }
    nodes.sort_unstable();
    G2p { half, classes, nodes }
}

/// Held-out scores: word exact match (with stress), word match ignoring
/// stress, phoneme error rate over units (stress included), µs per word.
pub struct Score {
    pub words: usize,
    pub exact: usize,
    pub exact_nostress: usize,
    pub per: f64,
    pub us_per_word: f64,
}

fn edit(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + (a[i - 1] != b[j - 1]) as usize);
        }
        prev = cur;
    }
    prev[b.len()]
}

pub fn score(m: &G2p, pairs: &[(String, String)]) -> Score {
    let t0 = std::time::Instant::now();
    let preds: Vec<String> = pairs.iter().map(|(w, _)| m.predict(w)).collect();
    let us = t0.elapsed().as_secs_f64() * 1e6 / pairs.len().max(1) as f64;
    let strip = |s: &str| s.replace([PRIMARY, SECONDARY], "");
    let (mut ex, mut exn, mut errs, mut tot) = (0, 0, 0usize, 0usize);
    for ((_, gold), p) in pairs.iter().zip(&preds) {
        ex += (p == gold) as usize;
        exn += (strip(p) == strip(gold)) as usize;
        let (g, q) = (units(gold), units(p));
        errs += edit(&g, &q);
        tot += g.len();
    }
    Score {
        words: pairs.len(),
        exact: ex,
        exact_nostress: exn,
        per: errs as f64 / tot.max(1) as f64,
        us_per_word: us,
    }
}

/// (train, held-out) split of the lexicon, 1 word in 20 held out by hash.
pub type Pairs = Vec<(String, String)>;

pub fn split() -> (Pairs, Pairs) {
    lexicon_pairs().into_iter().partition(|(w, _)| !held_out(w))
}
