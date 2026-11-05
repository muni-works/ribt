//! lilt's pronunciation overrides (docs §26). Own test binary: lilt_parity
//! switches the (process-global) table off.

use ribt::lilt;

#[test]
fn overrides_win() {
    // docs §26: RIBT's pronunciation fixes beat misaki (which spells RIBT out, reads IMU as "eemu").
    let (ps, _) = lilt::phonemize("The RIBT voice reads the IMU on phyz.");
    assert!(ps.contains("ɹˈɪbɪt"), "{ps}");
    assert!(ps.contains("ˌIˌɛmjˈu"), "{ps}");
    assert!(ps.contains("fˈɪz"), "{ps}");
}

#[test]
fn short_words_need_exact_case() {
    let (ps, _) = lilt::phonemize("Ms Smith ran for 80 ms.");
    assert!(!ps.starts_with("mˈɪləsˌɛkəndz"), "{ps}");
    assert!(ps.contains("mˈɪləsˌɛkəndz"), "{ps}");
}

#[test]
fn interjections() {
    let (ps, _) = lilt::phonemize("Uh-huh, the ipse repo.");
    assert!(ps.contains("ˌʌhˈʌ"), "{ps}");
    assert!(ps.contains("ˈɪpsi"), "{ps}");
}

/// `Dict::get` binary-searches, so an unsorted row is silently unreachable.
#[test]
fn override_table_is_byte_sorted() {
    let t = include_str!("../data/lilt/override.tsv");
    let keys: Vec<&str> = t.lines().filter_map(|l| l.split_once('\t')).map(|(k, _)| k).collect();
    assert!(keys.windows(2).all(|w| w[0] < w[1]), "override.tsv must be LC_ALL=C sorted with unique keys");
}
