//! misaki 0.9.4 `en.G2P` (US) ported line for line: retokenize into
//! subtokens, resolve right-to-left with a one-word lookahead context
//! (`future_vowel`, `future_to`), lexicon lookup gold → silver with
//! POS-keyed heteronyms, -s/-ed/-ing stemming, numbers, stress rules.
//! Function names mirror misaki's so the two can be read side by side.

use super::lexicon::{gold, gold_plain, silver, Entry};
use super::num;
use super::oov;

pub const PRIMARY: char = 'ˈ';
pub const SECONDARY: char = 'ˌ';

fn is_vowel(c: char) -> bool {
    "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ".contains(c)
}
fn is_consonant(c: char) -> bool {
    "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ".contains(c)
}
const PUNCTS: &str = ";:,.!?—…\"“”";
fn is_non_quote_punct(c: char) -> bool {
    ";:,.!?—…".contains(c)
}
const SUBTOKEN_JUNKS: &str = "',-._‘’/";
const PUNCT_TAGS: &[&str] = &[".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"];
const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";
const CURRENCIES: &[(&str, &str, &str)] = &[("$", "dollar", "cent"), ("£", "pound", "pence"), ("€", "euro", "cent")];
const ORDINALS: &[&str] = &["st", "nd", "rd", "th"];

fn currency(c: &str) -> Option<(&'static str, &'static str)> {
    CURRENCIES.iter().find(|x| x.0 == c).map(|x| (x.1, x.2))
}

fn lexicon_ord(c: char) -> bool {
    c == '\'' || c == '-' || c.is_ascii_alphabetic()
}

// Python str predicates (Unicode-aware, matching CPython closely enough).
fn py_isalpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char::is_alphabetic)
}
fn is_digit(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}
fn lower(s: &str) -> String {
    s.to_lowercase()
}
fn upper(s: &str) -> String {
    s.to_uppercase()
}
fn tail(s: &str) -> &str {
    s.char_indices().nth(1).map_or("", |(i, _)| &s[i..])
}

#[derive(Clone, Debug, Default)]
pub struct MTok {
    pub text: String,
    pub tag: String,
    pub ws: bool,
    pub phonemes: Option<String>,
    pub is_head: bool,
    pub alias: Option<String>,
    pub stress: Option<f64>,
    pub currency: Option<String>,
    pub prespace: bool,
    pub rating: Option<i32>,
    /// index of the spaCy-level token this came from (streaming bookkeeping)
    pub src: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx {
    pub future_vowel: Option<bool>,
    pub future_to: bool,
}

// ---------------------------------------------------------------- stress

fn restress(ps: &str) -> String {
    let cs: Vec<char> = ps.chars().collect();
    let mut ips: Vec<(f64, char)> = cs.iter().enumerate().map(|(i, &c)| (i as f64, c)).collect();
    for i in 0..cs.len() {
        if cs[i] == PRIMARY || cs[i] == SECONDARY {
            if let Some(j) = (i..cs.len()).find(|&j| is_vowel(cs[j])) {
                ips[i].0 = j as f64 - 0.5;
            }
        }
    }
    ips.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    ips.into_iter().map(|x| x.1).collect()
}

pub fn apply_stress(ps: &str, stress: Option<f64>) -> String {
    let Some(s) = stress else { return ps.to_string() };
    let has_p = ps.contains(PRIMARY);
    let has_s = ps.contains(SECONDARY);
    let has_v = ps.chars().any(is_vowel);
    if s < -1.0 {
        ps.replace([PRIMARY, SECONDARY], "")
    } else if s == -1.0 || ((s == 0.0 || s == -0.5) && has_p) {
        ps.replace(SECONDARY, "").replace(PRIMARY, &SECONDARY.to_string())
    } else if (s == 0.0 || s == 0.5 || s == 1.0) && !has_p && !has_s {
        if !has_v {
            return ps.to_string();
        }
        restress(&format!("{SECONDARY}{ps}"))
    } else if s >= 1.0 && !has_p && has_s {
        ps.replace(SECONDARY, &PRIMARY.to_string())
    } else if s > 1.0 && !has_p && !has_s {
        if !has_v {
            return ps.to_string();
        }
        restress(&format!("{PRIMARY}{ps}"))
    } else {
        ps.to_string()
    }
}

fn stress_weight(ps: &str) -> usize {
    ps.chars().map(|c| if "AIOQWYʤʧ".contains(c) { 2 } else { 1 }).sum()
}

// ---------------------------------------------------------------- lexicon

type R = Option<(String, i32)>;

fn parent_tag(tag: Option<&str>) -> Option<String> {
    let t = tag?;
    Some(if t.starts_with("VB") {
        "VERB".into()
    } else if t.starts_with("NN") {
        "NOUN".into()
    } else if t.starts_with("ADV") || t.starts_with("RB") {
        "ADV".into()
    } else if t.starts_with("ADJ") || t.starts_with("JJ") {
        "ADJ".into()
    } else {
        t.into()
    })
}

fn get_nnp(word: &str) -> R {
    let mut ps = String::new();
    for c in word.chars().filter(|c| c.is_alphabetic()) {
        match gold().get(&upper(&c.to_string())) {
            Some(Entry::Plain(p)) => ps.push_str(p),
            _ => return None,
        }
    }
    let ps = apply_stress(&ps, Some(0.0));
    let out = match ps.rfind(SECONDARY) {
        Some(i) => format!("{}{}{}", &ps[..i], PRIMARY, &ps[i + SECONDARY.len_utf8()..]),
        None => ps,
    };
    Some((out, 3))
}

fn get_special_case(word: &str, tag: &str, stress: Option<f64>, ctx: Ctx) -> R {
    let ctx_some = Some(ctx);
    if tag == "ADD" && (word == "." || word == "/") {
        return lookup(if word == "." { "dot" } else { "slash" }, None, Some(-0.5), ctx_some);
    }
    let sym = match word {
        "%" => Some("percent"),
        "&" => Some("and"),
        "+" => Some("plus"),
        "@" => Some("at"),
        _ => None,
    };
    if let Some(s) = sym {
        return lookup(s, None, None, ctx_some);
    }
    let stripped = word.trim_matches('.');
    if stripped.contains('.') && py_isalpha(&word.replace('.', "")) && word.split('.').map(|p| p.chars().count()).max().unwrap_or(0) < 3 {
        return get_nnp(word);
    }
    match word {
        "a" | "A" => return Some((if tag == "DT" { "ɐ".into() } else { "ˈA".into() }, 4)),
        "am" | "Am" | "AM" => {
            if tag.starts_with("NN") {
                return get_nnp(word);
            } else if ctx.future_vowel.is_none() || word != "am" || stress.is_some_and(|s| s > 0.0) {
                return Some((gold_plain("am").into(), 4));
            }
            return Some(("ɐm".into(), 4));
        }
        "an" | "An" | "AN" => {
            if word == "AN" && tag.starts_with("NN") {
                return get_nnp(word);
            }
            return Some(("ɐn".into(), 4));
        }
        _ => {}
    }
    if word == "I" && tag == "PRP" {
        return Some((format!("{SECONDARY}I"), 4));
    }
    if matches!(word, "by" | "By" | "BY") && parent_tag(Some(tag)).as_deref() == Some("ADV") {
        return Some(("bˈI".into(), 4));
    }
    if matches!(word, "to" | "To") || (word == "TO" && matches!(tag, "TO" | "IN")) {
        return Some((
            match ctx.future_vowel {
                None => gold_plain("to").into(),
                Some(false) => "tə".into(),
                Some(true) => "tʊ".into(),
            },
            4,
        ));
    }
    if matches!(word, "in" | "In") || (word == "IN" && tag != "NNP") {
        let s = if ctx.future_vowel.is_none() || tag != "IN" { "ˈ" } else { "" };
        return Some((format!("{s}ɪn"), 4));
    }
    if matches!(word, "the" | "The") || (word == "THE" && tag == "DT") {
        return Some((if ctx.future_vowel == Some(true) { "ði".into() } else { "ðə".into() }, 4));
    }
    if tag == "IN" && (word.eq_ignore_ascii_case("vs") || word.eq_ignore_ascii_case("vs.")) {
        return lookup("versus", None, None, ctx_some);
    }
    if matches!(word, "used" | "Used" | "USED") {
        let e = gold().get("used").unwrap();
        let key = if matches!(tag, "VBD" | "JJ") && ctx.future_to {
            "VBD"
        } else {
            "DEFAULT"
        };
        return Some((e.table_get(key).flatten().unwrap_or("").into(), 4));
    }
    None
}

fn is_known(word: &str, _tag: Option<&str>) -> bool {
    if gold().contains(word) || matches!(word, "%" | "&" | "+" | "@") || silver().contains(word) {
        return true;
    }
    if !py_isalpha(word) || !word.chars().all(lexicon_ord) {
        return false;
    }
    if word.chars().count() == 1 {
        return true;
    }
    if word == upper(word) && gold().contains(&lower(word)) {
        return true;
    }
    let t = tail(word);
    t == upper(t)
}

pub fn lookup(word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<Ctx>) -> R {
    let mut word = word.to_string();
    let mut is_nnp: Option<bool> = None;
    if word == upper(&word) && !gold().contains(&word) {
        word = lower(&word);
        is_nnp = Some(tag == Some("NNP"));
    }
    let (mut e, mut rating) = (gold().get(&word), 4);
    if e.is_none() && is_nnp != Some(true) {
        e = silver().get(&word);
        rating = 3;
    }
    let ps: Option<&str> = match e {
        None => None,
        Some(Entry::Plain(p)) => Some(p),
        Some(t @ Entry::Table(_)) => {
            let key: Option<String> = if ctx.is_some_and(|c| c.future_vowel.is_none()) && t.table_get("None").is_some() {
                Some("None".into())
            } else if tag.is_some_and(|tg| t.table_get(tg).is_some()) {
                tag.map(String::from)
            } else {
                parent_tag(tag)
            };
            match key.as_deref().and_then(|k| t.table_get(k)) {
                Some(v) => v,
                None => t.table_get("DEFAULT").flatten(),
            }
        }
    };
    if ps.is_none() || (is_nnp == Some(true) && !ps.unwrap().contains(PRIMARY)) {
        if let Some(r) = get_nnp(&word) {
            return Some(r);
        }
    }
    let ps = ps?;
    Some((apply_stress(ps, stress), rating))
}

pub(crate) fn s_suffix(stem: &str) -> Option<String> {
    let last = stem.chars().last()?;
    Some(if "ptkfθ".contains(last) {
        format!("{stem}s")
    } else if "szʃʒʧʤ".contains(last) {
        format!("{stem}ᵻz")
    } else {
        format!("{stem}z")
    })
}

fn stem_s(word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<Ctx>) -> R {
    let n = word.chars().count();
    if n < 3 || !word.ends_with('s') {
        return None;
    }
    let stem = if !word.ends_with("ss") && is_known(&word[..word.len() - 1], tag) {
        word[..word.len() - 1].to_string()
    } else if (word.ends_with("'s") || (n > 4 && word.ends_with("es") && !word.ends_with("ies"))) && is_known(&word[..word.len() - 2], tag) {
        word[..word.len() - 2].to_string()
    } else if n > 4 && word.ends_with("ies") && is_known(&format!("{}y", &word[..word.len() - 3]), tag) {
        format!("{}y", &word[..word.len() - 3])
    } else {
        return None;
    };
    let (st, r) = lookup(&stem, tag, stress, ctx)?;
    Some((s_suffix(&st)?, r))
}

fn ed_suffix(stem: &str) -> Option<String> {
    let cs: Vec<char> = stem.chars().collect();
    let last = *cs.last()?;
    Some(if "pkfθʃsʧ".contains(last) {
        format!("{stem}t")
    } else if last == 'd' {
        format!("{stem}ᵻd")
    } else if last != 't' {
        format!("{stem}d")
    } else if cs.len() < 2 {
        format!("{stem}ɪd")
    } else if US_TAUS.contains(cs[cs.len() - 2]) {
        format!("{}ɾᵻd", &stem[..stem.len() - 1])
    } else {
        format!("{stem}ᵻd")
    })
}

fn stem_ed(word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<Ctx>) -> R {
    let n = word.chars().count();
    if n < 4 || !word.ends_with('d') {
        return None;
    }
    let stem = if !word.ends_with("dd") && is_known(&word[..word.len() - 1], tag) {
        &word[..word.len() - 1]
    } else if n > 4 && word.ends_with("ed") && !word.ends_with("eed") && is_known(&word[..word.len() - 2], tag) {
        &word[..word.len() - 2]
    } else {
        return None;
    };
    let (st, r) = lookup(stem, tag, stress, ctx)?;
    Some((ed_suffix(&st)?, r))
}

fn ing_suffix(stem: &str) -> Option<String> {
    let cs: Vec<char> = stem.chars().collect();
    if cs.is_empty() {
        return None;
    }
    if cs.len() > 1 && cs[cs.len() - 1] == 't' && US_TAUS.contains(cs[cs.len() - 2]) {
        return Some(format!("{}ɾɪŋ", &stem[..stem.len() - 1]));
    }
    Some(format!("{stem}ɪŋ"))
}

fn doubled_ing(word: &str) -> bool {
    // ([bcdgklmnprstvxz])\1ing$|cking$
    if word.ends_with("cking") {
        return true;
    }
    let b = word.as_bytes();
    let n = b.len();
    n >= 5 && word.ends_with("ing") && b[n - 4] == b[n - 5] && b"bcdgklmnprstvxz".contains(&b[n - 4])
}

fn stem_ing(word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<Ctx>) -> R {
    let n = word.chars().count();
    if n < 5 || !word.ends_with("ing") {
        return None;
    }
    let base = &word[..word.len() - 3];
    let stem = if n > 5 && is_known(base, tag) {
        base.to_string()
    } else if is_known(&format!("{base}e"), tag) {
        format!("{base}e")
    } else if n > 5 && doubled_ing(word) && is_known(&word[..word.len() - 4], tag) {
        word[..word.len() - 4].to_string()
    } else {
        return None;
    };
    let (st, r) = lookup(&stem, tag, stress, ctx)?;
    Some((ing_suffix(&st)?, r))
}

fn get_word(word: &str, tag: &str, stress: Option<f64>, ctx: Ctx) -> R {
    if let Some(ps) = super::lexicon::override_get(word) {
        return Some((ps.into(), 4));
    }
    if let Some(r) = get_special_case(word, tag, stress, ctx) {
        return Some(r);
    }
    let tg = Some(tag);
    let c = Some(ctx);
    let wl = lower(word);
    let mut word = word.to_string();
    if word.chars().count() > 1
        && py_isalpha(&word.replace('\'', ""))
        && word != wl
        && (tag != "NNP" || word.chars().count() > 7)
        && !gold().contains(&word)
        && !silver().contains(&word)
        && (word == upper(&word) || tail(&word) == lower(tail(&word)))
        && (gold().contains(&wl)
            || silver().contains(&wl)
            || stem_s(&wl, tg, stress, c).is_some_and(|r| !r.0.is_empty())
            || stem_ed(&wl, tg, stress, c).is_some_and(|r| !r.0.is_empty())
            || stem_ing(&wl, tg, stress, c).is_some_and(|r| !r.0.is_empty()))
    {
        word = wl;
    }
    if is_known(&word, tg) {
        return lookup(&word, tg, stress, c);
    }
    if word.ends_with("s'") && is_known(&format!("{}'s", &word[..word.len() - 2]), tg) {
        return lookup(&format!("{}'s", &word[..word.len() - 2]), tg, stress, c);
    }
    if word.ends_with('\'') && is_known(&word[..word.len() - 1], tg) {
        return lookup(&word[..word.len() - 1], tg, stress, c);
    }
    if let Some(r) = stem_s(&word, tg, stress, c) {
        return Some(r);
    }
    if let Some(r) = stem_ed(&word, tg, stress, c) {
        return Some(r);
    }
    stem_ing(&word, tg, Some(stress.unwrap_or(0.5)), c)
}

/// A bare word through the lexicon (overrides, gold, silver, -s/-ed/-ing),
/// as a noun with no right context; the OOV ladder's "is this known?".
pub(crate) fn lexicon_word(word: &str) -> Option<String> {
    let stress = if word == lower(word) { None } else if word == upper(word) { Some(2.0) } else { Some(0.5) };
    get_word(word, "NN", stress, Ctx::default()).map(|r| r.0)
}

fn is_currency_amount(word: &str) -> bool {
    match word.matches('.').count() {
        0 => true,
        1 => word.split('.').nth(1).unwrap().chars().count() < 3,
        _ => false,
    }
}

fn get_number(word: &str, cur: Option<&str>, is_head: bool) -> R {
    let sfx_start = word
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_ascii_lowercase() || *c == '\'')
        .last()
        .map(|x| x.0);
    let (mut word, suffix) = match sfx_start {
        Some(i) => (&word[..i], Some(&word[i..])),
        None => (word, None),
    };
    let mut result: Vec<(String, i32)> = Vec::new();
    if let Some(w) = word.strip_prefix('-') {
        result.push(lookup("minus", None, None, None)?);
        word = w;
    }
    let extend = |words: Vec<String>, result: &mut Vec<(String, i32)>| -> Option<()> {
        for w in words.into_iter().filter(|w| !w.is_empty()) {
            if w != "and" {
                result.push(lookup(&w, None, if w == "point" { Some(-2.0) } else { None }, None)?);
            }
        }
        Some(())
    };
    let card = |s: &str| -> Option<Vec<String>> { num::cardinal(s.parse().ok()?) };
    let is_ord = suffix.is_some_and(|s| ORDINALS.contains(&s));
    let cur_known = cur.and_then(currency);
    if is_digit(word) && is_ord {
        extend(num::ordinal(word.parse().ok()?)?, &mut result)?;
    } else if result.is_empty() && word.len() == 4 && cur_known.is_none() && is_digit(word) {
        extend(num::year(word.parse().ok()?)?, &mut result)?;
    } else if !is_head && !word.contains('.') {
        let n = word.replace(',', "");
        let nb = n.as_bytes();
        if nb.first() == Some(&b'0') || n.len() > 3 {
            for d in n.chars() {
                extend(card(&d.to_string())?, &mut result)?;
            }
        } else if n.len() == 3 && !n.ends_with("00") {
            extend(card(&n[..1])?, &mut result)?;
            if nb[1] == b'0' {
                result.push(lookup("O", None, Some(-2.0), None)?);
                extend(card(&n[2..])?, &mut result)?;
            } else {
                extend(card(&n[1..])?, &mut result)?;
            }
        } else {
            extend(card(&n)?, &mut result)?;
        }
    } else if word.matches('.').count() > 1 || !is_head {
        for n in word.replace(',', "").split('.') {
            if n.is_empty() {
                continue;
            }
            if n.starts_with('0') || (n.len() != 2 && n[1..].bytes().any(|b| b != b'0')) {
                for d in n.chars() {
                    extend(card(&d.to_string())?, &mut result)?;
                }
            } else {
                extend(card(n)?, &mut result)?;
            }
        }
    } else if let (Some((unit, sub)), true) = (cur_known, is_currency_amount(word)) {
        let w = word.replace(',', "");
        let mut pairs: Vec<(i128, &str)> = w
            .split('.')
            .zip([unit, sub])
            .map(|(n, u)| (if n.is_empty() { 0 } else { n.parse().unwrap_or(0) }, u))
            .collect();
        if pairs.len() > 1 {
            if pairs[1].0 == 0 {
                pairs.truncate(1);
            } else if pairs[0].0 == 0 {
                pairs.remove(0);
            }
        }
        for (i, (n, u)) in pairs.into_iter().enumerate() {
            if i > 0 {
                result.push(lookup("and", None, None, None)?);
            }
            extend(num::cardinal(n)?, &mut result)?;
            result.push(if n.abs() != 1 && u != "pence" {
                stem_s(&format!("{u}s"), None, None, None)?
            } else {
                lookup(u, None, None, None)?
            });
        }
    } else {
        let words = if is_digit(word) {
            card(word)?
        } else if !word.contains('.') {
            let v: i128 = word.replace(',', "").parse().ok()?;
            if is_ord {
                num::ordinal(v)?
            } else {
                num::cardinal(v)?
            }
        } else {
            let w = word.replace(',', "");
            if let Some(frac) = w.strip_prefix('.') {
                let mut v = vec!["point".to_string()];
                for d in frac.chars() {
                    v.extend(card(&d.to_string())?);
                }
                v
            } else {
                num::float(&w)?
            }
        };
        extend(words, &mut result)?;
    }
    if result.is_empty() {
        return None;
    }
    let rating = result.iter().map(|r| r.1).min()?;
    let joined = result.into_iter().map(|r| r.0).collect::<Vec<_>>().join(" ");
    match suffix {
        Some("s" | "'s") => Some((s_suffix(&joined)?, rating)),
        Some("ed" | "'d") => Some((ed_suffix(&joined)?, rating)),
        Some("ing") => Some((ing_suffix(&joined)?, rating)),
        _ => Some((joined, rating)),
    }
}

fn append_currency(ps: String, cur: Option<&str>) -> String {
    match cur.and_then(currency) {
        Some((u, _)) => match stem_s(&format!("{u}s"), None, None, None) {
            Some((c, _)) => format!("{ps} {c}"),
            None => ps,
        },
        None => ps,
    }
}

fn is_number(word: &str, is_head: bool) -> bool {
    if !word.bytes().any(|b| b.is_ascii_digit()) {
        return false;
    }
    let mut w = word;
    for s in ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"] {
        if let Some(x) = w.strip_suffix(s) {
            w = x;
            break;
        }
    }
    w.chars()
        .enumerate()
        .all(|(i, c)| c.is_ascii_digit() || c == ',' || c == '.' || (is_head && i == 0 && c == '-'))
}

/// The tiny slice of NFKC + `unicodedata.numeric` misaki's lexicon applies.
fn normalize_word(w: &str) -> String {
    let mut out = String::with_capacity(w.len());
    for c in w.chars() {
        let c = match c {
            '‘' | '’' => '\'',
            '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0).unwrap(),
            '²' => '2',
            '³' => '3',
            '¹' => '1',
            '⁰' => '0',
            '⁴'..='⁹' => char::from_u32(c as u32 - 0x2074 + '4' as u32).unwrap(),
            '\u{FB01}' => {
                out.push_str("fi");
                continue;
            }
            '\u{FB02}' => {
                out.push_str("fl");
                continue;
            }
            '…' => {
                out.push_str("...");
                continue;
            }
            '\u{A0}' => ' ',
            _ => c,
        };
        out.push(c);
    }
    out
}

/// misaki `Lexicon.__call__`.
pub fn lexicon_call(tk: &MTok, ctx: Ctx) -> R {
    let word = normalize_word(tk.alias.as_deref().unwrap_or(&tk.text));
    let stress = if word == lower(&word) {
        None
    } else if word == upper(&word) {
        Some(2.0)
    } else {
        Some(0.5)
    };
    if let Some((ps, r)) = get_word(&word, &tk.tag, stress, ctx) {
        return Some((apply_stress(&append_currency(ps, tk.currency.as_deref()), tk.stress), r));
    }
    if is_number(&word, tk.is_head) {
        let (ps, r) = get_number(&word, tk.currency.as_deref(), tk.is_head)?;
        return Some((apply_stress(&ps, tk.stress), r));
    }
    None
}

// ---------------------------------------------------------------- subtokenize

fn is_apos(c: char) -> bool {
    matches!(c, '\'' | '‘' | '’')
}

/// misaki's SUBTOKEN_REGEX findall, hand-compiled.
pub fn subtokenize(word: &str) -> Vec<String> {
    let cs: Vec<char> = word.chars().collect();
    let n = cs.len();
    let mut out = Vec::new();
    let mut p = 0;
    let lu = |i: usize| i < n && cs[i].is_uppercase();
    let ll = |i: usize| i < n && cs[i].is_lowercase();
    let l = |i: usize| i < n && cs[i].is_alphabetic();
    let d = |i: usize| i < n && cs[i].is_ascii_digit();
    while p < n {
        let m: Option<usize> = (|| {
            // ^['‘’]+
            if p == 0 && is_apos(cs[0]) {
                return Some(cs.iter().take_while(|&&c| is_apos(c)).count());
            }
            // \p{Lu}(?=\p{Lu}\p{Ll})
            if lu(p) && lu(p + 1) && ll(p + 2) {
                return Some(1);
            }
            // (?:^-)?(?:\d?[,.]?\d)+
            {
                let mut q = p;
                if p == 0 && cs[0] == '-' {
                    q = 1;
                }
                let start_units = q;
                loop {
                    let sep = |i: usize| i < n && (cs[i] == ',' || cs[i] == '.');
                    let next = if d(q) && sep(q + 1) && d(q + 2) {
                        Some(q + 3)
                    } else if d(q) && d(q + 1) {
                        Some(q + 2)
                    } else if sep(q) && d(q + 1) {
                        Some(q + 2)
                    } else if d(q) {
                        Some(q + 1)
                    } else {
                        None
                    };
                    match next {
                        Some(x) => q = x,
                        None => break,
                    }
                }
                if q > start_units {
                    return Some(q - p);
                }
            }
            // [-_]+
            if cs[p] == '-' || cs[p] == '_' {
                return Some(cs[p..].iter().take_while(|&&c| c == '-' || c == '_').count());
            }
            // ['‘’]{2,}
            if is_apos(cs[p]) {
                let k = cs[p..].iter().take_while(|&&c| is_apos(c)).count();
                if k >= 2 {
                    return Some(k);
                }
            }
            // \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})
            if l(p) {
                let mut q = p;
                let mut seen_apos = false;
                while q < n {
                    if ll(q) && lu(q + 1) {
                        return Some(q + 1 - p);
                    }
                    if !seen_apos && l(q) {
                        q += 1;
                    } else if is_apos(cs[q]) && l(q + 1) {
                        seen_apos = true;
                        if ll(q + 1) && lu(q + 2) {
                            return Some(q + 2 - p);
                        }
                        q += 2;
                    } else {
                        break;
                    }
                }
            }
            // \p{L}+(?:['‘’]\p{L})*
            if l(p) {
                let mut q = p;
                while l(q) {
                    q += 1;
                }
                while q < n && is_apos(cs[q]) && l(q + 1) {
                    q += 2;
                }
                return Some(q - p);
            }
            // [^-_\p{L}'‘’\d]
            if !(cs[p] == '-' || cs[p] == '_' || l(p) || is_apos(cs[p]) || d(p)) {
                return Some(1);
            }
            // ['‘’]+$
            if is_apos(cs[p]) && cs[p..].iter().all(|&c| is_apos(c)) {
                return Some(n - p);
            }
            None
        })();
        match m {
            Some(k) if k > 0 => {
                out.push(cs[p..p + k].iter().collect());
                p += k;
            }
            _ => p += 1,
        }
    }
    out
}

// ---------------------------------------------------------------- G2P

#[derive(Clone, Debug)]
pub enum Word {
    One(MTok),
    Group(Vec<MTok>),
}

fn all_az(s: &str) -> bool {
    s.chars().all(|c| {
        let l: Vec<char> = c.to_lowercase().collect();
        l.len() == 1 && l[0].is_ascii_lowercase()
    })
}

pub fn retokenize(tokens: &[MTok]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut cur: Option<String> = None;
    for (i, token) in tokens.iter().enumerate() {
        let mut tks: Vec<MTok> = if token.alias.is_none() && token.phonemes.is_none() {
            subtokenize(&token.text)
                .into_iter()
                .map(|t| MTok {
                    text: t,
                    tag: token.tag.clone(),
                    ws: false,
                    is_head: true,
                    stress: token.stress,
                    src: token.src,
                    ..Default::default()
                })
                .collect()
        } else {
            vec![token.clone()]
        };
        if tks.is_empty() {
            continue; // findall found nothing (e.g. a lone apostrophe run mid-word)
        }
        tks.last_mut().unwrap().ws = token.ws;
        let nt = tks.len();
        for j in 0..nt {
            let (before, after) = tks.split_at_mut(j);
            let next_text0 = after.get(1).and_then(|x| x.text.chars().next());
            let tk = &mut after[0];
            if tk.alias.is_some() || tk.phonemes.is_some() {
            } else if tk.tag == "$" && currency(&tk.text).is_some() {
                cur = Some(tk.text.clone());
                tk.phonemes = Some(String::new());
                tk.rating = Some(4);
            } else if tk.tag == ":" && (tk.text == "-" || tk.text == "–") {
                tk.phonemes = Some("—".into());
                tk.rating = Some(3);
            // §27: a mis-tagged "86" (tag '.') is not punctuation; let the lexicon/fallback read it
            } else if PUNCT_TAGS.contains(&tk.tag.as_str()) && !all_az(&tk.text) && !(oov::enabled() && tk.text.chars().any(char::is_alphanumeric)) {
                tk.phonemes = Some(match tk.tag.as_str() {
                    "-LRB-" => "(".into(),
                    "-RRB-" => ")".into(),
                    "``" => "“".into(),
                    "\"\"" | "''" => "”".into(),
                    _ => tk.text.chars().filter(|&c| PUNCTS.contains(c)).collect(),
                });
                tk.rating = Some(4);
            } else if cur.is_some() {
                if tk.tag != "CD" {
                    cur = None;
                } else if j + 1 == nt && (i + 1 == tokens.len() || tokens[i + 1].tag != "CD") {
                    tk.currency = cur.clone();
                }
            } else if 0 < j && j < nt - 1 && tk.text == "2" {
                let a = before[j - 1].text.chars().last();
                if let (Some(a), Some(b)) = (a, next_text0) {
                    if a.is_alphabetic() && b.is_alphabetic() {
                        tk.alias = Some("to".into());
                    }
                }
            }
        }
        for mut tk in tks {
            if tk.alias.is_some() || tk.phonemes.is_some() {
                words.push(Word::One(tk));
            } else if let Some(Word::Group(g)) = words.last_mut().filter(|w| matches!(w, Word::Group(g) if !g.last().unwrap().ws)) {
                tk.is_head = false;
                g.push(tk);
            } else if tk.ws {
                words.push(Word::One(tk));
            } else {
                words.push(Word::Group(vec![tk]));
            }
        }
    }
    words
        .into_iter()
        .map(|w| match w {
            Word::Group(mut g) if g.len() == 1 => Word::One(g.pop().unwrap()),
            w => w,
        })
        .collect()
}

pub fn merge_tokens(tokens: &[MTok], unk: Option<&str>) -> MTok {
    let stresses: Vec<f64> = {
        let mut v: Vec<f64> = tokens.iter().filter_map(|t| t.stress).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup();
        v
    };
    let currency = tokens.iter().filter_map(|t| t.currency.clone()).max();
    let phonemes = unk.map(|unk| {
        let mut ps = String::new();
        for tk in tokens {
            if tk.prespace && !ps.is_empty() && !ps.chars().last().unwrap().is_whitespace() && tk.phonemes.as_deref().is_some_and(|p| !p.is_empty()) {
                ps.push(' ');
            }
            ps.push_str(tk.phonemes.as_deref().unwrap_or(unk));
        }
        ps
    });
    let mut text = String::new();
    for (i, t) in tokens.iter().enumerate() {
        text.push_str(&t.text);
        if i + 1 < tokens.len() && t.ws {
            text.push(' ');
        }
    }
    let weight = |t: &MTok| {
        t.text
            .chars()
            .map(|c| if lower(&c.to_string()) == c.to_string() { 1 } else { 2 })
            .sum::<usize>()
    };
    let mut best = &tokens[0];
    for t in &tokens[1..] {
        if weight(t) > weight(best) {
            best = t;
        }
    }
    let ratings: Vec<Option<i32>> = tokens.iter().map(|t| t.rating).collect();
    MTok {
        text,
        tag: best.tag.clone(),
        ws: tokens.last().unwrap().ws,
        phonemes,
        is_head: tokens[0].is_head,
        alias: None,
        stress: if stresses.len() == 1 { Some(stresses[0]) } else { None },
        currency,
        prespace: tokens[0].prespace,
        rating: if ratings.contains(&None) {
            None
        } else {
            ratings.into_iter().flatten().min()
        },
        src: tokens[0].src,
    }
}

pub fn token_context(ctx: Ctx, ps: Option<&str>, token: &MTok) -> Ctx {
    let mut vowel = ctx.future_vowel;
    if let Some(ps) = ps.filter(|p| !p.is_empty()) {
        if let Some(c) = ps.chars().find(|&c| is_vowel(c) || is_consonant(c) || is_non_quote_punct(c)) {
            vowel = if is_non_quote_punct(c) { None } else { Some(is_vowel(c)) };
        }
    }
    let future_to = matches!(token.text.as_str(), "to" | "To") || (token.text == "TO" && matches!(token.tag.as_str(), "TO" | "IN"));
    Ctx {
        future_vowel: vowel,
        future_to,
    }
}

fn resolve_tokens(tokens: &mut [MTok]) {
    let mut text = String::new();
    for (i, t) in tokens.iter().enumerate() {
        text.push_str(&t.text);
        if i + 1 < tokens.len() && t.ws {
            text.push(' ');
        }
    }
    let classes: std::collections::BTreeSet<u8> = text
        .chars()
        .filter(|&c| !SUBTOKEN_JUNKS.contains(c))
        .map(|c| {
            if c.is_alphabetic() {
                0
            } else if c.is_ascii_digit() {
                1
            } else {
                2
            }
        })
        .collect();
    let prespace = text.contains(' ') || text.contains('/') || classes.len() > 1;
    let n = tokens.len();
    for (i, tk) in tokens.iter_mut().enumerate() {
        if tk.phonemes.is_none() {
            if i == n - 1 && tk.text.chars().count() == 1 && is_non_quote_punct(tk.text.chars().next().unwrap()) {
                tk.phonemes = Some(tk.text.clone());
                tk.rating = Some(3);
            } else if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                tk.phonemes = Some(String::new());
                tk.rating = Some(3);
            }
        } else if i > 0 {
            tk.prespace = prespace;
        }
    }
    if prespace {
        return;
    }
    let mut idx: Vec<(bool, usize, usize)> = tokens
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            t.phonemes
                .as_deref()
                .filter(|p| !p.is_empty())
                .map(|p| (p.contains(PRIMARY), stress_weight(p), i))
        })
        .collect();
    if idx.len() == 2 && tokens[idx[0].2].text.chars().count() == 1 {
        let i = idx[1].2;
        tokens[i].phonemes = Some(apply_stress(tokens[i].phonemes.as_deref().unwrap(), Some(-0.5)));
        return;
    }
    let primaries = idx.iter().filter(|x| x.0).count();
    if idx.len() < 2 || primaries <= (idx.len() + 1) / 2 {
        return;
    }
    idx.sort();
    let half = idx.len() / 2;
    for &(_, _, i) in &idx[..half] {
        tokens[i].phonemes = Some(apply_stress(tokens[i].phonemes.as_deref().unwrap(), Some(-0.5)));
    }
}

/// Resolve words `[..end]` right to left starting from context `ctx`
/// (the context contributed by word `end` and beyond). Returns the context
/// after processing word 0 (unused) — callers use per-word results.
pub fn resolve(words: &mut [Word], mut ctx: Ctx) {
    for w in words.iter_mut().rev() {
        match w {
            Word::One(t) => {
                if t.phonemes.is_none() {
                    if let Some((p, r)) = lexicon_call(t, ctx) {
                        t.phonemes = Some(p);
                        t.rating = Some(r);
                    } else if oov::enabled() {
                        // misaki: `self.fallback(tk)` (espeak); lilt: docs §27
                        t.phonemes = oov::normalise(&t.text).or_else(|| oov::guess(&t.text)).or_else(|| oov::silent(&t.text));
                        t.rating = t.phonemes.as_ref().map(|_| 1);
                    }
                }
                ctx = token_context(ctx, t.phonemes.as_deref(), t);
            }
            Word::Group(g) => {
                let (mut left, mut right) = (0usize, g.len());
                while left < right {
                    let blocked = g[left..right].iter().any(|t| t.alias.is_some() || t.phonemes.is_some());
                    let tk = if blocked { None } else { Some(merge_tokens(&g[left..right], None)) };
                    let res = tk.as_ref().and_then(|tk| lexicon_call(tk, ctx));
                    if let (Some((ps, r)), Some(tk)) = (res, tk.as_ref()) {
                        ctx = token_context(ctx, Some(&ps), tk);
                        g[left].phonemes = Some(ps);
                        g[left].rating = Some(r);
                        for x in &mut g[left + 1..right] {
                            x.phonemes = Some(String::new());
                        }
                        right = left;
                        left = 0;
                    } else if left + 1 < right {
                        left += 1;
                    } else {
                        right -= 1;
                        let t = &mut g[right];
                        if t.phonemes.is_none() && t.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                            t.phonemes = Some(String::new());
                            t.rating = Some(3);
                        }
                        left = 0;
                    }
                }
                if oov::enabled() && g.iter().any(|t| t.phonemes.is_none()) {
                    // a whole-run pattern (4:15, an email) first, else each unknown part
                    let text: String = g.iter().map(|t| t.text.as_str()).collect();
                    if let Some(ps) = oov::normalise(&text) {
                        for x in g.iter_mut() {
                            x.phonemes = Some(String::new());
                            x.rating = Some(1);
                        }
                        g[0].phonemes = Some(ps);
                    } else {
                        for x in g.iter_mut().filter(|x| x.phonemes.is_none()) {
                            x.phonemes = oov::guess(&x.text).or_else(|| oov::silent(&x.text));
                            x.rating = x.phonemes.as_ref().map(|_| 1);
                        }
                    }
                }
                resolve_tokens(g);
            }
        }
    }
}

/// Final per-word token (misaki's merged MToken) after the version-1.0
/// replacements (ɾ→T, ʔ→t). `phonemes == None` means OOV (espeak in misaki);
/// a group with an unknown part carries `unk` in its place.
pub fn finalize(words: Vec<Word>, unk: &str) -> Vec<MTok> {
    words
        .into_iter()
        .map(|w| {
            let mut t = match w {
                Word::One(t) => t,
                Word::Group(g) => merge_tokens(&g, Some(unk)),
            };
            if let Some(p) = &t.phonemes {
                t.phonemes = Some(p.replace('ɾ', "T").replace('ʔ', "t"));
            }
            t
        })
        .collect()
}
