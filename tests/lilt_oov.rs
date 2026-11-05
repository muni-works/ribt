//! lilt's unknown-word ladder (docs §27): no word is silently dropped.
//! Own test binary, since lilt_parity switches the fallback off process-wide.

use ribt::lilt;

fn ps(t: &str) -> String {
    lilt::phonemize(t).0
}

fn lost(t: &str) -> Vec<String> {
    lilt::phonemize(t).1.into_iter().filter(|w| w.prosody.oov || (w.phonemes.is_empty() && w.text.chars().any(char::is_alphanumeric))).map(|w| w.text).collect()
}

#[test]
fn fun_set_has_no_deletions() {
    let t = include_str!("../data/lilt/oov_fun.tsv");
    for l in t.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let c: Vec<&str> = l.split('\t').collect();
        assert!(lost(c[0]).is_empty(), "word {:?} dropped", c[0]);
        assert!(lost(c[3]).is_empty(), "sentence {:?} drops {:?}", c[3], lost(c[3]));
    }
}

#[test]
fn clock_times() {
    assert_eq!(ps("4:15"), "fˈɔɹ fˌɪftˈin");
    assert!(ps("at 4:05").contains("fˈɔɹ ˈO fˈIv"), "{}", ps("at 4:05"));
    assert!(ps("12:00").contains("əklˈɑk"), "{}", ps("12:00"));
}

#[test]
fn morphology_and_compounds() {
    assert!(ps("He talketh.").starts_with("hˌi tˈɔkɪθ"), "{}", ps("He talketh."));
    assert!(ps("softmax").starts_with("sˈɔft"), "{}", ps("softmax"));
}

#[test]
fn initialisms_spelled_lexicon_first() {
    // NASA stays a word (lexicon), an unknown all-caps token is spelled
    assert_eq!(ps("NASA"), "nˈæsə");
    assert!(ps("QZX").contains("kjˌu"), "{}", ps("QZX"));
}

#[test]
fn addresses_are_read() {
    let p = ps("cam@ipse.dev");
    assert!(p.contains(" æt ") && p.contains("dˈɑt"), "{p}");
}

#[test]
fn g2p_model_loads_and_is_fast() {
    let m = lilt::oov::model().expect("embedded g2p.bin");
    assert!(!m.is_empty());
    let t0 = std::time::Instant::now();
    for _ in 0..1000 {
        std::hint::black_box(m.predict("flibbertigibbet"));
    }
    let us = t0.elapsed().as_secs_f64() * 1e3; // ms per 1000 = us per word
    assert!(us < 1000.0, "{us:.1} us/word");
}

#[test]
fn greek_letters_named() {
    assert!(ps("at 10 N σ").ends_with("sˈɪɡmə"), "{}", ps("at 10 N σ"));
    assert!(ps("Ω").contains("Omˈ"), "{}", ps("Ω"));
}

#[test]
fn stream_equals_batch_with_fallback() {
    let g = ribt::lilt::parity::load(concat!(env!("CARGO_MANIFEST_DIR"), "/data/lilt/golden_eval.tsv")).expect("golden_eval.tsv");
    let (n, bad) = ribt::lilt::parity::stream_check(&g, 1000);
    assert_eq!(bad, 0, "{bad}/{n} sentences differ between stream and batch with the OOV fallback on");
}

#[test]
fn digraph_never_loses_both_letters() {
    // §29.2: "Lacheneur" used to read lˌæənˈɜɹ (c and h each chose an empty chunk)
    let p = ps("Lacheneur");
    assert!(p.contains('ʧ') || p.contains('ʃ'), "{p}");
}
