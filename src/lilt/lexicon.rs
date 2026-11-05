//! misaki's US gold/silver dictionaries, embedded as sorted TSV and searched
//! with a binary search over a borrowed index (no allocation per lookup).
//!
//! The TSVs are written by `scripts/lilt_golden.py export` *after* misaki's
//! `grow_dictionary` (capitalised/lowercased aliases), so a lookup here is
//! exactly misaki's `dict.get`. Values are either a phoneme string or a
//! POS-keyed table `{TAG=ps|TAG=ps|...` where `~` is Python `None`.

use std::sync::OnceLock;

static GOLD_TSV: &str = include_str!("../../data/lilt/gold.tsv");
static SILVER_TSV: &str = include_str!("../../data/lilt/silver.tsv");
/// RIBT's own pronunciation fixes (docs §26): words Kokoro/misaki get wrong.
/// Consulted before misaki's dictionaries; the same table is injected into
/// Kokoro's misaki when re-rendering teacher audio, so student and frontend agree.
static OVERRIDE_TSV: &str = include_str!("../../data/lilt/override.tsv");

/// A dictionary value: a plain pronunciation or a tag-keyed table.
#[derive(Clone, Copy, Debug)]
pub enum Entry {
    Plain(&'static str),
    Table(&'static str),
}

impl Entry {
    /// Look a tag up in a table entry: `Some(None)` is an explicit Python `None`.
    pub fn table_get(&self, tag: &str) -> Option<Option<&'static str>> {
        let Entry::Table(t) = self else { return None };
        for kv in t.split('|') {
            let (k, v) = kv.split_once('=')?;
            if k == tag {
                return Some(if v == "~" { None } else { Some(v) });
            }
        }
        None
    }
}

pub struct Dict {
    rows: Vec<(&'static str, &'static str)>,
}

impl Dict {
    fn parse(tsv: &'static str) -> Dict {
        let rows: Vec<_> = tsv.lines().filter_map(|l| l.split_once('\t')).collect();
        debug_assert!(rows.windows(2).all(|w| w[0].0 < w[1].0), "lexicon TSV must be byte-sorted");
        Dict { rows }
    }
    pub fn get(&self, k: &str) -> Option<Entry> {
        let i = self.rows.binary_search_by(|(r, _)| (*r).cmp(k)).ok()?;
        let v = self.rows[i].1;
        Some(match v.strip_prefix('{') {
            Some(t) => Entry::Table(t),
            None => Entry::Plain(v),
        })
    }
    pub fn contains(&self, k: &str) -> bool {
        self.rows.binary_search_by(|(r, _)| (*r).cmp(k)).is_ok()
    }
    pub fn rows(&self) -> &[(&'static str, &'static str)] {
        &self.rows
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

pub fn gold() -> &'static Dict {
    static D: OnceLock<Dict> = OnceLock::new();
    D.get_or_init(|| Dict::parse(GOLD_TSV))
}

static OVERRIDES_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Switch the override table off (misaki parity measurement) or back on (default).
/// `RIBT_LILT_OVERRIDES=0` in the environment also switches it off (A/B of the fixes).
pub fn set_overrides(on: bool) {
    OVERRIDES_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn overrides() -> &'static Dict {
    static D: OnceLock<Dict> = OnceLock::new();
    D.get_or_init(|| Dict::parse(OVERRIDE_TSV))
}

/// The override pronunciation for `word`: exact key, else the lowercased word
/// (so "Phyz" at a sentence start still hits "phyz"); short words never fall
/// back, so the title "Ms" is not read as the unit "ms".
pub fn override_get(word: &str) -> Option<&'static str> {
    static ENV_OFF: OnceLock<bool> = OnceLock::new();
    let env_off = *ENV_OFF.get_or_init(|| std::env::var("RIBT_LILT_OVERRIDES").is_ok_and(|v| v == "0"));
    if env_off || !OVERRIDES_ON.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let d = overrides();
    let lower = || (word.chars().count() > 3).then(|| d.get(&word.to_lowercase())).flatten();
    match d.get(word).or_else(lower) {
        Some(Entry::Plain(p)) => Some(p),
        _ => None,
    }
}

pub fn silver() -> &'static Dict {
    static D: OnceLock<Dict> = OnceLock::new();
    D.get_or_init(|| Dict::parse(SILVER_TSV))
}

/// gold["x"] for a key known to hold a plain string.
pub fn gold_plain(k: &str) -> &'static str {
    match gold().get(k) {
        Some(Entry::Plain(p)) => p,
        _ => "",
    }
}
