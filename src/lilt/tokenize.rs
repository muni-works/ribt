//! spaCy 3.8 English tokenizer, re-implemented without regexes.
//!
//! misaki runs on spaCy's tokens (and its tagger runs on them too), so token
//! boundaries are part of the parity contract. The algorithm is spaCy's
//! `Tokenizer._split_affixes` + `_attach_tokens` verbatim: split on spaces,
//! then peel prefixes/suffixes (stopping at special cases), then split the
//! core on infixes, unless it is a special case or a URL. The prefix/suffix/
//! infix regexes of `spacy.lang.en` are hand-compiled into predicates below;
//! the special-case table (contractions, abbreviations, emoticons; 1343
//! entries) is exported from spaCy (MIT) into `data/lilt/tok_rules.tsv`.
//!
//! Known gaps: the `faster_heuristics` special-case matcher pass (special
//! cases spanning several affix-split tokens), runs of whitespace (spaCy
//! emits `_SP` tokens; we treat any whitespace run as one separator), and
//! non-Latin unit/currency suffixes.

use std::collections::HashMap;
use std::sync::OnceLock;

static RULES_TSV: &str = include_str!("../../data/lilt/tok_rules.tsv");

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tok {
    pub text: String,
    /// followed by a space in the source
    pub ws: bool,
}

fn specials() -> &'static HashMap<&'static str, Vec<&'static str>> {
    static S: OnceLock<HashMap<&'static str, Vec<&'static str>>> = OnceLock::new();
    S.get_or_init(|| {
        RULES_TSV
            .lines()
            .filter_map(|l| {
                let mut it = l.split('\t');
                let k = it.next()?;
                Some((k, it.collect()))
            })
            .collect()
    })
}

fn is_special(s: &str) -> bool {
    specials().contains_key(s)
}

// ---- character classes (spacy/lang/char_classes.py) ----

const QUOTES: &[char] = &[
    '\'', '"', '”', '“', '`', '‘', '´', '’', '‚', ',', '„', '»', '«', '「', '」', '『', '』', '（', '）', '〔', '〕', '【', '】', '《', '》', '〈',
    '〉', '⟦', '⟧',
];
const PUNCT: &[char] = &[
    '…', ',', ':', ';', '!', '?', '¿', '؟', '¡', '(', ')', '[', ']', '{', '}', '<', '>', '_', '#', '*', '&', '。', '？', '！', '，', '、', '；',
    '：', '～', '·', '।', '،', '۔', '؛', '٪',
];
const CURRENCY: &[&str] = &[
    "$", "£", "€", "¥", "฿", "US$", "C$", "A$", "₽", "﷼", "₴", "₠", "₡", "₢", "₣", "₤", "₥", "₦", "₧", "₨", "₩", "₪", "₫", "₭", "₮", "₯", "₰", "₱",
    "₲", "₳", "₵", "₶", "₷", "₸", "₹", "₺", "₻", "₼", "₾", "₿",
];
const UNITS: &[&str] = &[
    "km", "km²", "km³", "m", "m²", "m³", "dm", "dm²", "dm³", "cm", "cm²", "cm³", "mm", "mm²", "mm³", "ha", "µm", "nm", "yd", "in", "ft", "kg", "g",
    "mg", "µg", "t", "lb", "oz", "m/s", "km/h", "kmh", "mph", "hPa", "Pa", "mbar", "mb", "MB", "kb", "KB", "gb", "GB", "tb", "TB", "T", "G", "M",
    "K", "%",
];

/// spaCy's "other symbol" class (So), approximated by the blocks that matter.
fn is_symbol(c: char) -> bool {
    matches!(c, '¦' | '©' | '®' | '°' | '҂' | '℀' | '℁' | '℃'..='℆' | '℈' | '℉' | '℔' | '№' | '℗' | '℞'..='℣' | '℥' | '℧' | '℩' | '℮' | '℺' | '℻' | '⅊' | '⅌' | '⅍' | '⅏')
        || ('\u{2195}'..='\u{2199}').contains(&c)
        || ('\u{219C}'..='\u{21F3}').contains(&c)
            && !matches!(
                c,
                '\u{21A0}' | '\u{21A3}' | '\u{21A6}' | '\u{21AE}' | '\u{21CE}' | '\u{21CF}' | '\u{21D2}' | '\u{21D4}'
            )
        || ('\u{2300}'..='\u{2307}').contains(&c)
        || ('\u{230C}'..='\u{231F}').contains(&c)
        || ('\u{2322}'..='\u{2328}').contains(&c)
        || ('\u{232B}'..='\u{237B}').contains(&c)
        || ('\u{237D}'..='\u{239A}').contains(&c)
        || ('\u{23B4}'..='\u{23DB}').contains(&c)
        || ('\u{23E2}'..='\u{2426}').contains(&c)
        || ('\u{2440}'..='\u{244A}').contains(&c)
        || ('\u{249C}'..='\u{24E9}').contains(&c)
        || ('\u{2500}'..='\u{25B6}').contains(&c)
        || ('\u{25B8}'..='\u{25C0}').contains(&c)
        || ('\u{25C2}'..='\u{25F7}').contains(&c)
        || ('\u{2600}'..='\u{266E}').contains(&c)
        || ('\u{2670}'..='\u{2767}').contains(&c)
        || ('\u{2794}'..='\u{27BF}').contains(&c)
        || ('\u{2800}'..='\u{28FF}').contains(&c)
        || ('\u{2B00}'..='\u{2B2F}').contains(&c)
        || ('\u{1F000}'..='\u{1FA6D}').contains(&c)
}

fn alpha(c: char) -> bool {
    c.is_alphabetic()
}

// ---- prefix ----

fn find_prefix(s: &str) -> usize {
    let mut cs = s.chars();
    let Some(c) = cs.next() else { return 0 };
    let n = c.len_utf8();
    match c {
        '+' => {
            return if cs.next().is_some_and(|d| d.is_ascii_digit()) { 0 } else { 1 };
        }
        '.' => {
            let k = s.chars().take_while(|&d| d == '.').count();
            return if k >= 2 { k } else { 0 };
        }
        '§' | '%' | '=' | '—' | '–' => return n,
        _ => {}
    }
    if PUNCT.contains(&c) || QUOTES.contains(&c) || is_symbol(c) {
        return n;
    }
    for cur in CURRENCY {
        if s.starts_with(cur) {
            return cur.len();
        }
    }
    0
}

// ---- suffix: re.search over `alt$`, i.e. the leftmost start whose tail matches ----

fn suffix_tail_matches(prev2: Option<char>, prev: Option<char>, t: &str) -> bool {
    let mut cs = t.chars();
    let c0 = cs.next().unwrap();
    let single = cs.next().is_none();
    let pdigit = prev.is_some_and(|p| p.is_ascii_digit());
    if single && (PUNCT.contains(&c0) || QUOTES.contains(&c0) || is_symbol(c0) || c0 == '—' || c0 == '–') {
        return true;
    }
    if t == "……" || (t.len() >= 2 && t.chars().all(|d| d == '.')) {
        return true;
    }
    if matches!(t, "'s" | "'S" | "’s" | "’S") {
        return true;
    }
    if t == "+" && pdigit {
        return true;
    }
    if pdigit && (CURRENCY.contains(&t) || UNITS.contains(&t)) {
        return true;
    }
    if t == "." {
        if let (Some(p2), Some(p)) = (prev2, prev) {
            if p2 == '°' && matches!(p, 'F' | 'f' | 'C' | 'c' | 'K' | 'k') {
                return true;
            }
            if p2.is_uppercase() && p.is_uppercase() {
                return true;
            }
        }
        if let Some(p) = prev {
            if p.is_ascii_digit() || p.is_lowercase() || matches!(p, '%' | '²' | '-' | '+') || PUNCT.contains(&p) || QUOTES.contains(&p) {
                return true;
            }
        }
    }
    false
}

fn find_suffix(s: &str) -> usize {
    let idx: Vec<(usize, char)> = s.char_indices().collect();
    for (k, &(b, _)) in idx.iter().enumerate() {
        let prev = if k > 0 { Some(idx[k - 1].1) } else { None };
        let prev2 = if k > 1 { Some(idx[k - 2].1) } else { None };
        if suffix_tail_matches(prev2, prev, &s[b..]) {
            return s.len() - b;
        }
    }
    0
}

// ---- infix: re.finditer ----

const HYPHENS: &[&str] = &["-", "–", "—", "--", "---", "——", "~"];

fn infix_at(chars: &[(usize, char)], s: &str, k: usize) -> Option<usize> {
    let (b, c) = chars[k];
    let prev = if k > 0 { Some(chars[k - 1].1) } else { None };
    // \.\.+
    if c == '.' {
        let n = chars[k..].iter().take_while(|x| x.1 == '.').count();
        if n >= 2 {
            return Some(n);
        }
    }
    if c == '…' || is_symbol(c) {
        return Some(1);
    }
    let next_char_at = |j: usize| chars.get(j).map(|x| x.1);
    // (?<=[0-9])[+\-\*^](?=[0-9-])
    if prev.is_some_and(|p| p.is_ascii_digit())
        && matches!(c, '+' | '-' | '*' | '^')
        && next_char_at(k + 1).is_some_and(|n| n.is_ascii_digit() || n == '-')
    {
        return Some(1);
    }
    // (?<=[lower quotes])\.(?=[upper quotes])
    if c == '.'
        && prev.is_some_and(|p| p.is_lowercase() || QUOTES.contains(&p))
        && next_char_at(k + 1).is_some_and(|n| n.is_uppercase() || QUOTES.contains(&n))
    {
        return Some(1);
    }
    // (?<=[alpha]),(?=[alpha])
    if c == ',' && prev.is_some_and(alpha) && next_char_at(k + 1).is_some_and(alpha) {
        return Some(1);
    }
    let prev_an = prev.is_some_and(|p| alpha(p) || p.is_ascii_digit());
    if prev_an {
        for h in HYPHENS {
            if s[b..].starts_with(h) {
                let n = h.chars().count();
                if next_char_at(k + n).is_some_and(alpha) {
                    return Some(n);
                }
            }
        }
        if matches!(c, ':' | '<' | '>' | '=' | '/') && next_char_at(k + 1).is_some_and(alpha) {
            return Some(1);
        }
    }
    None
}

fn find_infixes(s: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut out = Vec::new();
    let mut k = 0;
    while k < chars.len() {
        if let Some(n) = infix_at(&chars, s, k) {
            let start = chars[k].0;
            let end = chars.get(k + n).map_or(s.len(), |x| x.0);
            out.push((start, end));
            k += n;
        } else {
            k += 1;
        }
    }
    out
}

// ---- url_match ----

fn url_match(s: &str) -> bool {
    let mut rest = s;
    if let Some(i) = rest.find("://") {
        let sch = &rest[..i];
        if sch.chars().count() >= 2 && sch.chars().all(|c| c.is_alphanumeric() || c == '_' || matches!(c, '+' | '-' | '.')) {
            rest = &rest[i + 3..];
        }
    }
    // optional userinfo: \S+(?::\S*)?@  (greedy: take the last '@')
    if let Some(i) = rest.rfind('@') {
        if i > 0 {
            rest = &rest[i + 1..];
        }
    }
    // host: (label\.)+ tld ; then (:port)? ([/?#]\S*)?
    let host_end = rest.find(|c| matches!(c, ':' | '/' | '?' | '#')).unwrap_or(rest.len());
    let (host, tail) = rest.split_at(host_end);
    let labels: Vec<&str> = host.split('.').collect();
    if labels.len() < 2 {
        return false;
    }
    let lab_ok = |l: &str| {
        let ok = |c: char| c.is_ascii_alphanumeric() || ('\u{a1}'..='\u{ffff}').contains(&c);
        !l.is_empty()
            && l.chars().count() <= 64
            && l.chars().next().is_some_and(ok)
            && l.chars().last().is_some_and(ok)
            && l.chars().all(|c| ok(c) || c == '_' || c == '-')
    };
    let tld = labels[labels.len() - 1];
    let tn = tld.chars().count();
    if !(2..=63).contains(&tn) || !tld.chars().all(|c| c.is_lowercase()) {
        return false;
    }
    if !labels[..labels.len() - 1].iter().all(|l| lab_ok(l)) {
        return false;
    }
    let mut t = tail;
    if let Some(p) = t.strip_prefix(':') {
        let n = p.chars().take_while(|c| c.is_ascii_digit()).count();
        if !(2..=5).contains(&n) {
            return false;
        }
        t = &p[n..];
    }
    t.is_empty() || t.starts_with(['/', '?', '#'])
}

// ---- the algorithm ----

fn split_affixes(mut s: &str, prefixes: &mut Vec<String>, suffixes: &mut Vec<String>, sp: bool) -> String {
    let mut last = usize::MAX;
    while !s.is_empty() && s.len() != last {
        if sp && is_special(s) {
            break;
        }
        last = s.len();
        let pre = find_prefix(s);
        if pre != 0 {
            let minus = &s[pre..];
            if sp && !minus.is_empty() && is_special(minus) {
                prefixes.push(s[..pre].into());
                s = minus;
                break;
            }
        }
        let suf = find_suffix(&s[pre..]);
        if suf != 0 {
            let minus = &s[..s.len() - suf];
            if sp && !minus.is_empty() && is_special(minus) {
                suffixes.push(s[s.len() - suf..].into());
                s = minus;
                break;
            }
        }
        if pre != 0 && suf != 0 && pre + suf <= s.len() {
            prefixes.push(s[..pre].into());
            suffixes.push(s[s.len() - suf..].into());
            s = &s[pre..s.len() - suf];
        } else if pre != 0 {
            prefixes.push(s[..pre].into());
            s = &s[pre..];
        } else if suf != 0 {
            suffixes.push(s[s.len() - suf..].into());
            s = &s[..s.len() - suf];
        }
    }
    s.to_string()
}

fn push_special_or(out: &mut Vec<String>, s: &str) {
    match specials().get(s) {
        Some(v) => out.extend(v.iter().map(|x| x.to_string())),
        None => out.push(s.into()),
    }
}

/// Special cases spaCy also matches *across* affix-split tokens
/// (`faster_heuristics`): each key that contains an affix, as the token
/// sequence it tokenizes to without special cases. Keyed by first token.
fn matcher() -> &'static HashMap<String, Vec<(Vec<String>, &'static str)>> {
    static M: OnceLock<HashMap<String, Vec<(Vec<String>, &'static str)>>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m: HashMap<String, Vec<(Vec<String>, &'static str)>> = HashMap::new();
        for &k in specials().keys() {
            if find_prefix(k) > 0 || find_suffix(k) > 0 || !find_infixes(k).is_empty() {
                let pat = tokenize_raw(k, false);
                m.entry(pat[0].clone()).or_default().push((pat, k));
            }
        }
        for v in m.values_mut() {
            v.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(b.1)));
        }
        m
    })
}

/// Tokenize one whitespace-free chunk.
pub fn tokenize_chunk(span: &str) -> Vec<String> {
    let toks = tokenize_raw(span, true);
    // special-case matcher pass: longest match, left to right
    let m = matcher();
    let mut out = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        let hit = m
            .get(&toks[i])
            .and_then(|cands| cands.iter().find(|(pat, _)| pat.len() > 1 && toks[i..].starts_with(pat)));
        match hit {
            Some((pat, k)) => {
                out.extend(specials()[k].iter().map(|x| x.to_string()));
                i += pat.len();
            }
            None => {
                out.push(toks[i].clone());
                i += 1;
            }
        }
    }
    out
}

fn tokenize_raw(span: &str, sp: bool) -> Vec<String> {
    let mut out = Vec::new();
    if sp {
        if let Some(v) = specials().get(span) {
            return v.iter().map(|x| x.to_string()).collect();
        }
    }
    let (mut pre, mut suf) = (Vec::new(), Vec::new());
    let core = split_affixes(span, &mut pre, &mut suf, sp);
    out.extend(pre);
    if !core.is_empty() {
        if sp && is_special(&core) {
            push_special_or(&mut out, &core);
        } else if url_match(&core) {
            out.push(core);
        } else {
            let m = find_infixes(&core);
            if m.is_empty() {
                out.push(core);
            } else {
                let mut start = 0;
                for (a, b) in m {
                    if a == 0 {
                        continue; // infix_start == start_before_infixes
                    }
                    if a != start {
                        out.push(core[start..a].into());
                    }
                    if a != b {
                        out.push(core[a..b].into());
                    }
                    start = b;
                }
                if start < core.len() {
                    out.push(core[start..].into());
                }
            }
        }
    }
    out.extend(suf.into_iter().rev());
    out
}

/// Tokenize text. Whitespace runs separate chunks; `ws` marks a token that
/// was followed by whitespace.
pub fn tokenize(text: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut it = text.split_whitespace().peekable();
    let trailing_ws = text.ends_with(char::is_whitespace);
    while let Some(chunk) = it.next() {
        let toks = tokenize_chunk(chunk);
        let n = toks.len();
        let last_chunk = it.peek().is_none();
        for (i, t) in toks.into_iter().enumerate() {
            let ws = i + 1 == n && (!last_chunk || trailing_ws);
            out.push(Tok { text: t, ws });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(s: &str) -> Vec<String> {
        tokenize(s).into_iter().map(|t| t.text).collect()
    }
    #[test]
    fn spacy_like() {
        assert_eq!(t("Don't stop!"), ["Do", "n't", "stop", "!"]);
        assert_eq!(t("Dr. Smith's car."), ["Dr.", "Smith", "'s", "car", "."]);
        assert_eq!(t("$5.50 (approx)"), ["$", "5.50", "(", "approx", ")"]);
        assert_eq!(t("state-of-the-art"), ["state", "-", "of", "-", "the", "-", "art"]);
        assert_eq!(t("20kg"), ["20", "kg"]);
        assert_eq!(t("https://example.com/path."), ["https://example.com/path", "."]);
        assert_eq!(t("U.S."), ["U.S."]);
        assert_eq!(t("Wait..."), ["Wait", "..."]);
    }
}
