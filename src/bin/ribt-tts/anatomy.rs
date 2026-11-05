//! `ribt-tts anatomy`: run the rung-1 synth path once per (line, speaker) and
//! dump every intermediate tensor the network computed, plus "tang data"
//! (per-stage wall time, kernels, params, storage precision), for
//! `scripts/anatomy_video.py`.
//!
//!   ribt-tts anatomy --run DIR [--ckpt step….safetensors] --voc VOC.safetensors
//!                    --ref A.wav[,B.wav…] (--lines FILE | --text "…") --dump DIR [--pace 1.0]
//!                    [--flow F.safetensors --nfe 1 --shortcut --temp 0.5]   rung-2 mel instead of rung 1's
//!
//! Per take, DIR/<name>_<spk>/ holds meta.json and npy files:
//!   h [d,N] encoder states, ld/pitch/energy [N], durs [N], tok_of_frame [T],
//!   mel [100,T], f0hz [T], voiced_logit [T], hid [dd,T], spk_dec [dd],
//!   audio.wav (raw vocoder output, not leveled).
//! Stage timings are the second (warm) of two runs at the same shape.

use super::*;
use ribt::lilt::Phonemizer;

fn js(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn precision_of(p: &Program) -> String {
    let mut tally: HashMap<String, usize> = HashMap::new();
    for k in 0..8192 {
        if let Some(i) = p.exe.input_slot(k) {
            let (len, f) = p.exe.spec(i);
            *tally.entry(format!("{f:?}")).or_default() += len;
        }
    }
    let mut v: Vec<_> = tally.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    format!("{{{}}}", v.iter().map(|(k, n)| format!("{}:{n}", js(k))).collect::<Vec<_>>().join(","))
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

pub fn anatomy(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let run = PathBuf::from(a.need("--run")?);
    let ckpt_name = a.opt("--ckpt");
    let (params, nm, step) = load_run(&run, &tc, ckpt_name.clone())?;
    let voc_path = PathBuf::from(a.need("--voc")?);
    let pace: f64 = a.get("--pace", 1.0);
    let refs_l: Vec<String> = a.need("--ref")?.split(',').map(String::from).collect();
    let jobs: Vec<(String, String)> = match (a.opt("--lines"), a.opt("--text")) {
        (Some(f), _) => std::fs::read_to_string(f)?
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .filter_map(|l| l.split_once('\t').map(|(n, t)| (n.to_string(), t.split('\t').next().unwrap().to_string())))
            .collect(),
        (None, Some(t)) => vec![("out".into(), t)],
        _ => return Err(anyhow!("--text or --lines")),
    };
    let dump = PathBuf::from(a.need("--dump")?);
    std::fs::create_dir_all(&dump)?;
    let dev = Device::new();
    // parameter counts by top-level name ("enc", "ref", "dur", "dec", …)
    let mut groups: Vec<(String, usize)> = Vec::new();
    for (s, v) in tc.params().iter().zip(&params) {
        let g = s.name.split('.').next().unwrap_or("?").to_string();
        match groups.iter_mut().find(|(k, _)| *k == g) {
            Some(e) => e.1 += v.len(),
            None => groups.push((g, v.len())),
        }
    }
    let vcfg = if ckpt::has(&voc_path, "src.w") { Config::by_name("v1f0").unwrap() } else { Config::v1() };
    let voc_params: usize = ckpt::load(&voc_path, &vcfg)?.params.iter().map(|p| p.len()).sum();

    // optional rung-2 flow decoder (`--flow F [--nfe N --shortcut --temp τ --seed S]`)
    let fc = ribt::flow::FConfig::by_name(&a.opt("--fcfg").unwrap_or_else(|| "base".into())).ok_or_else(|| anyhow!("fcfg"))?;
    let flow = match a.opt("--flow") {
        Some(f) => {
            let p = PathBuf::from(f);
            Some((ckpt::load_specs(&p, &fc.params(), false)?.params, super::flow_cmd::FNorm::load(&p)?))
        }
        None => None,
    };
    let nfe: usize = a.get("--nfe", 1);
    let shortcut = a.0.iter().any(|x| x == "--shortcut");
    let temp: f64 = a.get("--temp", 0.5);
    let seed: u64 = a.get("--seed", 0);
    let mut fcache: HashMap<usize, super::flow_cmd::Vel> = HashMap::new();
    let model_tag = if flow.is_some() { format!("v2 rung2 flow nfe{nfe}") } else { "v2a rung1".into() };
    let mut fe_ref = Frontend::new(&dev, 1, REF_T);
    let mut cache: HashMap<(usize, usize), Tts> = HashMap::new();
    let mut vcache: HashMap<usize, Vocoder> = HashMap::new();
    for (name, text) in &jobs {
        // warm run: lilt builds lexicon/tagger state lazily, so time the second pass
        let _ = ribt::lilt::phonemize(text);
        // lilt, streamed one whitespace chunk at a time: when did each word come out?
        let t_l = Instant::now();
        let mut ph = Phonemizer::new();
        let mut words: Vec<(ribt::lilt::WordOut, usize, f64)> = Vec::new();
        let mut fed = 0usize;
        let mut chunks: Vec<&str> = Vec::new();
        let mut last = 0;
        for (i, c) in text.char_indices() {
            if c == ' ' {
                chunks.push(&text[last..=i]);
                last = i + 1;
            }
        }
        chunks.push(&text[last..]);
        for ch in chunks {
            fed += ch.len();
            for w in ph.push(ch) {
                words.push((w, fed, t_l.elapsed().as_secs_f64() * 1e6));
            }
        }
        for w in ph.finish() {
            words.push((w, fed, t_l.elapsed().as_secs_f64() * 1e6));
        }
        let lilt_ms = ms(t_l);
        // tokens with their word index (-1: inter-word space / pad)
        let mut toks: Vec<u16> = vec![0];
        let mut tok_word: Vec<i64> = vec![-1];
        let mut tok_sym: Vec<String> = vec!["$".into()];
        for (wi, (w, _, _)) in words.iter().enumerate() {
            for c in w.phonemes.chars() {
                if let Some(id) = ribt::lilt::vocab::id(c) {
                    toks.push(id as u16);
                    tok_word.push(wi as i64);
                    tok_sym.push(c.to_string());
                }
            }
            if w.ws {
                if let Some(id) = ribt::lilt::vocab::id(' ') {
                    toks.push(id as u16);
                    tok_word.push(-1);
                    tok_sym.push(" ".into());
                }
            }
        }
        toks.push(0);
        tok_word.push(-1);
        tok_sym.push("$".into());
        let ps: String = ribt::lilt::join(&words.iter().map(|w| w.0.clone()).collect::<Vec<_>>());
        let ntok = toks.len();
        let n = ntok.div_ceil(32) * 32;
        for r in &refs_l {
            let rp = match r.strip_prefix("~/") {
                Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
                None => PathBuf::from(r),
            };
            let tag = rp.file_stem().unwrap().to_string_lossy().split('_').next().unwrap_or("spk").to_string();
            let (x, _) = ribt::data::read_wav(&rp)?;
            let fr = x.len() / HOP;
            let off = if fr > REF_T + 30 { 30 } else { 0 };
            let len = (fr - off).min(REF_T);
            let t_r = Instant::now();
            let refmel = fe_ref.run(&dev, &[x[off * HOP..(off + len) * HOP].to_vec()]);
            let ref_ms = ms(t_r);
            let mut rw = vec![0.0f32; REF_T];
            rw.iter_mut().take(len).for_each(|v| *v = 1.0 / len as f32);

            let (oh, tm) = onehot_rows(&tc, &[&toks], n);
            let tts = cache.entry((n, 0)).or_insert_with(|| Tts::new(&dev, &tc, &params, Dims { batch: 1, n, t: 64, r: REF_T }));
            let enc_kernels = tts.enc.kernels();
            let enc_prec = precision_of(&tts.enc);
            let _ = tts.encode(&dev, &oh, &tm, &refmel, &rw);
            let t_e = Instant::now();
            let [h, ld, pp, ep, spk] = tts.encode(&dev, &oh, &tm, &refmel, &rw);
            let enc_ms = ms(t_e);
            let t_v = Instant::now();
            let durs: Vec<u32> = (0..ntok).map(|i| dur_of(ld[i], pace)).collect();
            let frames: usize = durs.iter().sum::<u32>() as usize;
            let t = frames.div_ceil(64) * 64;
            let al = acoustic::align_matrix(&[&durs], n, t);
            let mut fm = vec![0.0f32; t];
            fm[..frames].iter_mut().for_each(|v| *v = 1.0);
            let lr_ms = ms(t_v);
            let tts = cache.entry((n, t)).or_insert_with(|| Tts::new(&dev, &tc, &params, Dims { batch: 1, n, t, r: REF_T }));
            let dec_kernels = tts.dec.kernels();
            let dec_prec = precision_of(&tts.dec);
            let _ = tts.decode(&dev, &h, &pp, &ep, &tm, &al, &fm, &spk);
            let t_d = Instant::now();
            let (mut mel, f0, hid) = tts.decode(&dev, &h, &pp, &ep, &tm, &al, &fm, &spk);
            if let Some((fp, fnm)) = &flow {
                // rung 2: the flow decoder replaces rung 1's mel (its μ and hidden are the conditioning)
                let vel = fcache.entry(t).or_insert_with(|| super::flow_cmd::Vel::new(&dev, &fc, fp, ribt::flow::FDims { batch: 1, t }));
                let mun = fnm.norm(&mel, &fm, 1, t);
                vel.set_cond(&dev, &mun, &hid, &fm, &ribt::flow::attn_bias(&[frames], t, None));
                let x0 = super::flow_cmd::noise(&mut Rng::new(seed), &fm, 1, tc.n_mels, t, temp);
                let (x1, _) = vel.sample(&dev, &x0, nfe, shortcut);
                mel = fnm.denorm(&x1);
            }
            let dec_ms = ms(t_d);
            let mel_c: Vec<f32> = (0..tc.n_mels).flat_map(|c| mel[c * t..c * t + frames].to_vec()).collect();
            let f0hz: Vec<f32> = (0..frames)
                .map(|f| if f0[t + f] > 0.0 { ((f0[f] as f64) * nm.f0_sd + nm.f0_mu).exp() as f32 } else { 0.0 })
                .collect();
            let vlog: Vec<f32> = f0[t..t + frames].to_vec();
            let voc = match vcache.entry(frames) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => e.insert(Vocoder::new(&dev, &voc_path, frames)?),
            };
            let voc_kernels = voc.prog.kernels();
            let voc_prec = precision_of(&voc.prog);
            let _ = voc.run(&dev, &mel_c, &f0hz, frames);
            let t_o = Instant::now();
            let y = voc.run(&dev, &mel_c, &f0hz, frames);
            let voc_ms = ms(t_o);

            let od = dump.join(format!("{name}_{tag}"));
            std::fs::create_dir_all(&od)?;
            let d = tc.d;
            let h_c: Vec<f32> = (0..d).flat_map(|c| h[c * n..c * n + ntok].to_vec()).collect();
            write_npy(&od.join("h.npy"), &[d, ntok], &h_c)?;
            write_npy(&od.join("ld.npy"), &[ntok], &ld[..ntok])?;
            write_npy(&od.join("pitch.npy"), &[ntok], &pp[..ntok])?;
            write_npy(&od.join("energy.npy"), &[ntok], &ep[..ntok])?;
            write_npy(&od.join("durs.npy"), &[ntok], &durs.iter().map(|&v| v as f32).collect::<Vec<_>>())?;
            let tof: Vec<f32> = durs.iter().enumerate().flat_map(|(i, &k)| std::iter::repeat(i as f32).take(k as usize)).collect();
            write_npy(&od.join("tok_of_frame.npy"), &[frames], &tof)?;
            write_npy(&od.join("mel.npy"), &[tc.n_mels, frames], &mel_c)?;
            write_npy(&od.join("f0hz.npy"), &[frames], &f0hz)?;
            write_npy(&od.join("voiced_logit.npy"), &[frames], &vlog)?;
            if hid.len() == tc.dd * t {
                let hc: Vec<f32> = (0..tc.dd).flat_map(|c| hid[c * t..c * t + frames].to_vec()).collect();
                write_npy(&od.join("hid.npy"), &[tc.dd, frames], &hc)?;
            }
            write_npy(&od.join("spk_dec.npy"), &[spk.len()], &spk)?;
            ribt::data::write_wav(&od.join("audio.wav"), &y, SR as u32)?;

            let secs = y.len() as f64 / SR as f64;
            let total = lilt_ms + ref_ms + enc_ms + lr_ms + dec_ms + voc_ms;
            let wj: Vec<String> = words
                .iter()
                .map(|(w, at, us)| format!("{{\"text\":{},\"ph\":{},\"ws\":{},\"stressed\":{},\"boundary\":{},\"oov\":{},\"emitted_at_char\":{at},\"emitted_us\":{us:.1}}}",
                    js(&w.text), js(&w.phonemes), w.ws, w.prosody.stressed, w.prosody.boundary.map_or("null".into(), |c| js(&c.to_string())), w.prosody.oov))
                .collect();
            let gj: Vec<String> = groups.iter().map(|(k, v)| format!("{}:{v}", js(k))).collect();
            let meta = format!(
                "{{\"model\":{},\"ckpt\":{},\"step\":{step},\"vocoder\":{},\"name\":{},\"speaker\":{},\"ref\":{},\"text\":{},\"phonemes\":{},\
\"tokens\":[{}],\"tok_sym\":[{}],\"tok_word\":[{}],\"words\":[{}],\"n_pad\":{n},\"t_pad\":{t},\"frames\":{frames},\"hop\":{HOP},\"sr\":{SR},\"seconds\":{secs:.4},\"pace\":{pace},\
\"ms\":{{\"lilt\":{lilt_ms:.3},\"ref\":{ref_ms:.3},\"encoder\":{enc_ms:.3},\"regulator\":{lr_ms:.3},\"decoder\":{dec_ms:.3},\"vocoder\":{voc_ms:.3},\"total\":{total:.3}}},\"rtf\":{:.5},\
\"kernels\":{{\"encoder\":{enc_kernels},\"decoder\":{dec_kernels},\"vocoder\":{voc_kernels}}},\"precision\":{{\"encoder\":{enc_prec},\"decoder\":{dec_prec},\"vocoder\":{voc_prec}}},\
\"params\":{{\"acoustic\":{{{}}},\"acoustic_total\":{},\"vocoder\":{voc_params}}},\"device\":{},\"f0_mu\":{},\"f0_sd\":{}}}\n",
                js(&model_tag),
                js(&ckpt_name.clone().unwrap_or_else(|| "last.safetensors".into())),
                js(&voc_path.file_name().unwrap().to_string_lossy()),
                js(name),
                js(&tag),
                js(r),
                js(text),
                js(&ps),
                toks.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
                tok_sym.iter().map(|s| js(s)).collect::<Vec<_>>().join(","),
                tok_word.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
                wj.join(","),
                total / 1e3 / secs,
                gj.join(","),
                tc.n_params(),
                js(&dev.name()),
                nm.f0_mu,
                nm.f0_sd,
            );
            std::fs::write(od.join("meta.json"), meta)?;
            eprintln!("{name}_{tag}: {ntok} tok → {frames} fr ({secs:.2} s); lilt {lilt_ms:.2} enc {enc_ms:.2} dec {dec_ms:.2} voc {voc_ms:.2} ms; RTF {:.4}", total / 1e3 / secs);
        }
    }
    Ok(())
}
