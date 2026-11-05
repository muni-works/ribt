//! lilt `--accent midwest[,ng]` (docs §26.7). One test: the accent is process-global.
use ribt::lilt::{self, accent, Phonemizer};

fn stream(text: &str) -> String {
    let mut p = Phonemizer::new();
    let mut w: Vec<_> = Vec::new();
    for piece in text.split_inclusive(' ') {
        w.extend(p.push(piece));
    }
    w.extend(p.finish());
    lilt::join(&w)
}

#[test]
fn midwest_raising() {
    let s = "Grab the bag from the wagon and drag it by the flag. Thanks for hanging the dragon magnet.";
    let control = "The cat sat on the mat and read a book.";
    accent::set("neutral").unwrap();
    let (n, _) = lilt::phonemize(s);
    let (nc, _) = lilt::phonemize(control);
    assert!(n.contains("bˈæɡ") && n.contains("θˈæŋ"), "{n}");

    accent::set("midwest").unwrap();
    let (m, _) = lilt::phonemize(s);
    assert!(m.contains("bˈAɡ") && m.contains("wˈAɡᵊn") && m.contains("dɹˈAɡ"), "{m}");
    assert!(m.contains("θˈæŋ"), "ŋ untouched without the sub-flag: {m}");
    assert_eq!(m.chars().filter(|&c| c == 'ˈ').count(), n.chars().filter(|&c| c == 'ˈ').count());
    assert_eq!(lilt::phonemize(control).0, nc, "no æɡ, no change");
    assert_eq!(stream(s), m, "stream == batch");

    accent::set("midwest,ng").unwrap();
    let (g, _) = lilt::phonemize(s);
    assert!(g.contains("θˈAŋ") && g.contains("bˈAɡ"), "{g}");
    assert_eq!(stream(s), g);

    accent::set("neutral").unwrap();
    assert_eq!(lilt::phonemize(s).0, n);

    // §26.8 long o: O (oʊ) -> oː, stress kept, stream == batch; midwest alone leaves O
    let o = "Ope! Oh geez, I know, we go home to Minnesota now.";
    let (on, _) = lilt::phonemize(o);
    assert!(on.contains("ˈOp") && on.contains('O'), "{on}");
    accent::set("midwest").unwrap();
    assert_eq!(lilt::phonemize(o).0, on);
    accent::set("minnesota").unwrap();
    let (om, _) = lilt::phonemize(o);
    let ch: Vec<char> = om.chars().collect();
    assert!(om.contains("ˈOːp") && ch.windows(2).all(|w| w[0] != 'O' || w[1] == 'ː'), "{om}");
    assert_eq!(om.chars().filter(|&c| c == 'ˈ').count(), on.chars().filter(|&c| c == 'ˈ').count());
    assert_eq!(stream(o), om);
    assert_eq!(lilt::phonemize(control).0, nc);
    assert!(ribt::persona::Reflex::Hup.text() == "Ope!" && ribt::persona::Reflex::Whoa.text() == "Oh geez!");
    accent::set("neutral").unwrap();
    assert_eq!(ribt::persona::Reflex::Whoa.text(), "Whoa!");
}
