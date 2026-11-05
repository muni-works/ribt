//! lilt for the browser: `phonemize(text)` → JSON with the phoneme string,
//! RIBT's input tokens (each phoneme with its Kokoro id) and the words.
//! The lilt sources are ribt's own (`src/lilt`), unchanged.

use wasm_bindgen::prelude::*;

#[path = "../../../src/lilt/mod.rs"]
pub mod lilt;

fn js(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// `neutral`, `midwest`, `minnesota`, … (lilt::accent::set).
#[wasm_bindgen]
pub fn set_accent(name: &str) -> Result<(), JsValue> {
    lilt::accent::set(name).map_err(|e| JsValue::from_str(&e))
}

/// `{"ps": "…", "toks": [["h", 50], …], "words": [{"t": "Hello", "p": "həlˈO", "oov": false, "ws": true}, …]}`.
/// `toks` holds only the phonemes that have a vocabulary id (what the model reads);
/// the model's input is `[0, ids…, 0]`.
#[wasm_bindgen]
pub fn phonemize(text: &str) -> String {
    let (ps, words) = lilt::phonemize(text);
    let toks: Vec<String> = ps.chars().filter_map(|c| lilt::vocab::id(c).map(|id| format!("[{},{id}]", js(&c.to_string())))).collect();
    let ws: Vec<String> = words
        .iter()
        .map(|w| format!("{{\"t\":{},\"p\":{},\"oov\":{},\"ws\":{}}}", js(&w.text), js(&w.phonemes), w.prosody.oov, w.ws))
        .collect();
    format!("{{\"ps\":{},\"toks\":[{}],\"words\":[{}]}}", js(&ps), toks.join(","), ws.join(","))
}
