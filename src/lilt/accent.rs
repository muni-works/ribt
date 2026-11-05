//! Opt-in accent rules applied to lilt's phoneme output (docs §26.7).
//!
//! `midwest`: Upper-Midwest pre-velar raising, /æ/ before /ɡ/ → misaki `A`
//! (eɪ): bag bˈæɡ → bˈAɡ, dragon, wagon, magnet. Sub-flag `ng` extends it to
//! /æ/ before /ŋ/ (bang, thank, hanging). Stress marks sit before the vowel,
//! so they stay in place; a length mark between vowel and velar is skipped.
//! Off by default, so misaki parity is unchanged. Set with `set` or
//! `RIBT_LILT_ACCENT=midwest[,ng][,o]` or `minnesota`.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

const MIDWEST: u8 = 1;
const NG: u8 = 2;
const O: u8 = 4;
/// 0xFF = not set by code; fall back to the environment
static SET: AtomicU8 = AtomicU8::new(0xFF);

/// Parse "neutral" | "midwest" | "midwest,ng".
pub fn parse(s: &str) -> Result<u8, String> {
    let mut f = 0;
    for p in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        f |= match p {
            "neutral" | "none" => 0,
            "midwest" => MIDWEST,
            "ng" => NG,
            "o" => O,
            "minnesota" => MIDWEST | O,
            _ => return Err(format!("unknown accent {p:?} (neutral | midwest[,ng][,o] | minnesota)")),
        };
    }
    if f & (NG | O) != 0 && f & MIDWEST == 0 {
        return Err("accent flags ng / o need midwest".into());
    }
    Ok(f)
}

/// Select the accent for this process ("neutral" switches it off).
pub fn set(s: &str) -> Result<(), String> {
    SET.store(parse(s)?, Ordering::Relaxed);
    Ok(())
}

fn flags() -> u8 {
    match SET.load(Ordering::Relaxed) {
        0xFF => {
            static ENV: OnceLock<u8> = OnceLock::new();
            *ENV.get_or_init(|| std::env::var("RIBT_LILT_ACCENT").ok().and_then(|v| parse(&v).ok()).unwrap_or(0))
        }
        f => f,
    }
}

/// Whether a Midwest accent is active (persona picks the Minnesotan reflexes).
pub fn midwest() -> bool {
    flags() & MIDWEST != 0
}

/// `ps` with the active accent's rewrites (unchanged when neutral).
pub fn apply(ps: &str) -> String {
    rewrite(ps, flags())
}

pub fn rewrite(ps: &str, f: u8) -> String {
    if f & MIDWEST == 0 || !(ps.contains('æ') || (f & O != 0 && ps.contains('O'))) {
        return ps.to_string();
    }
    let c: Vec<char> = ps.chars().collect();
    let mut out = String::with_capacity(ps.len());
    for (i, &ch) in c.iter().enumerate() {
        let mut j = i + 1;
        while c.get(j) == Some(&'ː') {
            j += 1;
        }
        let next = c.get(j).copied();
        let raise = ch == 'æ' && (next == Some('ɡ') || (f & NG != 0 && next == Some('ŋ')));
        if f & O != 0 && ch == 'O' {
            out.push('O');
            if c.get(i + 1) != Some(&'ː') {
                out.push('ː');
            }
            continue;
        }
        out.push(if raise { 'A' } else { ch });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rules() {
        let mw = parse("midwest").unwrap();
        let ng = parse("midwest,ng").unwrap();
        assert_eq!(rewrite("bˈæɡ", mw), "bˈAɡ");
        assert_eq!(rewrite("dɹˈæɡən", mw), "dɹˈAɡən");
        assert_eq!(rewrite("bˈæːɡ", mw), "bˈAːɡ");
        assert_eq!(rewrite("bˈæɡ", 0), "bˈæɡ");
        assert_eq!(rewrite("bˈæŋ", mw), "bˈæŋ");
        assert_eq!(rewrite("bˈæŋ", ng), "bˈAŋ");
        assert_eq!(rewrite("kˈæt sˈæd", ng), "kˈæt sˈæd");
        let mn = parse("minnesota").unwrap();
        assert_eq!(mn, parse("midwest,o").unwrap());
        assert_eq!(rewrite("ɡˈO", mn), "ɡˈOː");
        assert_eq!(rewrite("bˈOt bˈæɡ", mn), "bˈOːt bˈAɡ");
        assert_eq!(rewrite("ɡˈO", mw), "ɡˈO");
        assert_eq!(rewrite("ɡˈOː", mn), "ɡˈOː");
        assert!(parse("o").is_err());
        assert!(parse("ng").is_err() && parse("texan").is_err());
    }
}
