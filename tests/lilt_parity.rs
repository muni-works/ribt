//! lilt vs misaki golden parity (data/lilt/golden_eval.tsv: held-out test
//! split + hand-built edge cases). CPU only, a few seconds in release-ish
//! test profile.

use ribt::lilt::{self, parity, Phonemizer};

fn golden() -> Vec<parity::Sent> {
    // every test in this binary runs as misaki with fallback=None (§27); the
    // switch is process-global, so set it before any test reads it
    lilt::oov::set_enabled(false);
    parity::load(concat!(env!("CARGO_MANIFEST_DIR"), "/data/lilt/golden_eval.tsv")).expect("golden_eval.tsv")
}

#[test]
fn word_parity_vs_misaki() {
    let g = golden();
    lilt::lexicon::set_overrides(false); // parity is vs stock misaki; §26 fixes diverge on purpose
    let r = parity::evaluate(&g, &["test", "edge"]);
    parity::print_report(&r, 5);
    let word = r.word_ok as f64 / r.words as f64;
    let oracle = r.oracle_ok as f64 / r.oracle_words as f64;
    // the port itself is exact given spaCy's tokens and tags
    assert_eq!(r.oracle_ok, r.oracle_words, "misaki port drifted: oracle {oracle:.5}");
    // end to end (own tokenizer + distilled tagger): measured 99.85%
    assert!(word >= 0.995, "word parity {word:.5} < 0.995");
}

#[test]
fn stream_equals_batch() {
    let g = golden();
    let (n, bad) = parity::stream_check(&g, 1000);
    assert_eq!(bad, 0, "{bad}/{n} sentences differ between char-by-char stream and batch");
}

#[test]
fn heteronyms_and_numbers() {
    // expected strings are misaki 0.9.4 output
    let p = |t: &str| lilt::phonemize(t).0;
    assert_eq!(p("I read the book yesterday."), "ˌI ɹˈɛd ðə bˈʊk jˈɛstəɹdˌA.");
    assert_eq!(p("It costs $5.50."), "ˌɪt kˈɔsts fˈIv dˈɑləɹz ænd fˈɪfti sˈɛnts.");
    assert_eq!(p("She was born in 1984."), "ʃˌi wʌz bˈɔɹn ɪn nˌIntˈin ˈATi fˈɔɹ.");
}

#[test]
fn stream_emits_early() {
    let mut s = Phonemizer::new();
    let first: Vec<_> = s.push("Hello world ").collect();
    assert_eq!(first.len(), 2, "plain words go out as soon as their space arrives");
    // "read" waits for the tagger's lookahead
    let held: Vec<_> = s.push("I read ").collect();
    assert!(held.iter().all(|w| w.text != "read"));
    let rest: Vec<_> = s.push("it.").chain(s.finish()).collect();
    assert!(rest.iter().any(|w| w.text == "read"));
}

#[test]
fn kokoro_ids() {
    let (ps, _) = lilt::phonemize("Hello.");
    let ids = lilt::vocab::ids(&ps);
    assert_eq!(ids.len(), ps.chars().count(), "every lilt symbol is in Kokoro's vocab");
}
