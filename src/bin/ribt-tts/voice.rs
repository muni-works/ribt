//! Voices as small files (spike). A voice is the two speaker vectors the
//! acoustic model adds (`voice.se` [d] after the text encoder, `voice.sd`
//! [dd] at the decoder input; 640 f32, ~2.6 KB as safetensors) plus default
//! settings in the safetensors metadata (`name`, `pitch`, `rate`, `accent`,
//! and the flow decoder's `nfe`, `temp`).
//! The weights stay shared; every voice runs the same acoustic/flow/vocoder.
//!
//!   ribt-tts synth … --ref A.wav+B.wav --save-voice V.voice [--name N]   zero-shot: the reference
//!                    encoder's vectors for those clips (rung 1 of the ladder)
//!   ribt-tts synth … --voice V.voice     speak with a voice file instead of --ref (its pitch / rate /
//!                    accent are defaults; --pitch / --rate / --accent override)
//!   ribt-tts voice-blend --a A.voice --b B.voice --t 0.3 --out C.voice   (1−t)·A + t·B (defaults from A)
//!   ribt-tts voice-fit --run DIR [--ckpt C] --voice INIT.voice --lines FIT.tsv --wav-dir DIR
//!                    [--phon P.phon.tsv --durs D.durs.tsv] [--held HELD.tsv] [--steps 300] [--lr 3e-3]
//!                    [--batch 8] [--rank 0] [--check 25] [--no-f0-loss] --out V.voice
//!                    embedding fit (rung 2): every weight frozen, AdamW on the 640 voice numbers
//!                    against the target clips with the training losses (mel L1, duration, pitch,
//!                    energy, frame F0, voicing). Rows are `id<TAB>…<TAB>text` (text = last column),
//!                    audio DIR/id.wav at 24 kHz. Durations: with `--phon`/`--durs` (`ribt-align durs`
//!                    output) the aligner's tokens and durations are used as-is — that is the good
//!                    path; tokens come from the phon file so they always match the durations.
//!                    Without them each clip is DTW-aligned once against the init voice's own decoder
//!                    mel, which demonstrably misaligns (it starved a /d/, "derived" → "arrived").
//!                    --rank R (rung 3) also trains a rank-R adapter `w + b·a` on each decoder
//!                    block's two pointwise layers; it is stored in the voice file and folded into
//!                    the weights at load, so synthesis runs the same graphs at the same cost.
//!                    Rank is the main quality knob, and it costs voice-file size, not speed:
//!                    on speaker 7127 (ECAPA cosine to his held-out centroid, 14 sentences, and
//!                    Whisper base.en WER over 419 words) rank 0 / 4 / 8 / 16 / 32 / 64 / 128 gave
//!                    0.441 / 0.643 / 0.664 / 0.694 / 0.713 / 0.738 / 0.756 at 4.53 / 4.06 / 4.53 /
//!                    4.30 / 4.30 / 4.30 / 3.82 % WER (his real clips: 0.789). Voice files are
//!                    3 KB, 0.5, 1.0, 2.0, 3.9, 7.9 and 15.7 MB; decoder time is unchanged.
//!   ribt-tts voice-resynth --voc VOC --lines ROWS.tsv --wav-dir DIR --out-dir OUT [--n 4]
//!                    copy-synthesis: each clip's real log-mel and YIN F0 through the vocoder, i.e.
//!                    the vocoder's ceiling on that speaker (what is left is the acoustic model's)
//!   ribt-tts voice-probe --run DIR [--ckpt C] --voice V --voc VOC [--flow F --nfe 4 --shortcut --temp T]
//!                    --lines ROWS.tsv --wav-dir DIR --out-dir OUT [--n 4]
//!                    blame split on real clips (DTW durations, so outputs line up with the recording):
//!                    vocodes real mel + predicted F0, predicted mel (decoder, and flow) + real F0,
//!                    and predicted mel + predicted F0

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use ribt::acoustic::{self, Dims, TConfig, N_LOSS};
use ribt::model::Rng;
use ribt::runtime::{Device, Program};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use super::{flow_cmd, Vocoder, dur_of, frame_feats, host_batch, input_map, load_run, lr_at, onehot_rows, refs, Args, Frontend, Tts, Utt, HOP, REF_T, SR};

#[derive(Clone, Debug)]
pub struct Voice {
    pub se: Vec<f32>,
    pub sd: Vec<f32>,
    /// optional decoder adapter (rung 3): `(name, shape, data)` for the
    /// `voice.lora.<weight>.a` `[r, c]` / `.b` `[o, r]` pairs of `acoustic::adapter_names`
    pub lora: Vec<(String, Vec<usize>, Vec<f32>)>,
    /// name, pitch (semitones), rate, accent, source, …
    pub meta: HashMap<String, String>,
}

const LORA: &str = "voice.lora.";

fn bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

impl Voice {
    pub fn load(p: &Path) -> Result<Self> {
        let buf = std::fs::read(p)?;
        let (_, md) = SafeTensors::read_metadata(&buf)?;
        let st = SafeTensors::deserialize(&buf)?;
        let get = |n: &str| -> Result<Vec<f32>> {
            let t = st.tensor(n).map_err(|e| anyhow!("{}: {n}: {e}", p.display()))?;
            Ok(t.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
        };
        let mut names: Vec<String> = st.names().into_iter().filter(|n| n.starts_with(LORA)).cloned().collect();
        names.sort();
        let lora = names.into_iter().map(|n| Ok((n.clone(), st.tensor(&n)?.shape().to_vec(), get(&n)?))).collect::<Result<_>>()?;
        Ok(Self { se: get(acoustic::VOICE_NAMES[0])?, sd: get(acoustic::VOICE_NAMES[1])?, lora, meta: md.metadata().clone().unwrap_or_default() })
    }
    pub fn save(&self, p: &Path) -> Result<()> {
        let mut views = vec![
            (acoustic::VOICE_NAMES[0].to_string(), TensorView::new(Dtype::F32, vec![self.se.len()], bytes(&self.se))?),
            (acoustic::VOICE_NAMES[1].to_string(), TensorView::new(Dtype::F32, vec![self.sd.len()], bytes(&self.sd))?),
        ];
        for (n, sh, x) in &self.lora {
            views.push((n.clone(), TensorView::new(Dtype::F32, sh.clone(), bytes(x))?));
        }
        safetensors::serialize_to_file(views, &Some(self.meta.clone()), p)?;
        Ok(())
    }
    /// The adapter on base weight `w`: `(a [r, c], b [o, r], r)`.
    fn adapter(&self, w: &str) -> Option<(&[f32], &[f32], usize)> {
        let (an, bn) = acoustic::adapter_names(w);
        let a = self.lora.iter().find(|l| l.0 == an)?;
        let b = self.lora.iter().find(|l| l.0 == bn)?;
        Some((&a.2, &b.2, a.1[0]))
    }
    /// Fold the adapter into the base weights (`w += b·a`), so inference
    /// runs the unchanged graphs at the unchanged cost.
    pub fn merge(&self, tc: &TConfig, params: &mut [Vec<f32>]) -> Result<()> {
        if self.lora.is_empty() {
            return Ok(());
        }
        let specs = tc.params();
        for w in acoustic::adapter_targets(tc) {
            let Some((a, b, r)) = self.adapter(&w) else { continue };
            let i = specs.iter().position(|s| s.name == w).ok_or_else(|| anyhow!("no weight {w}"))?;
            let (o, c) = (specs[i].shape[0], specs[i].shape[1]);
            if a.len() != r * c || b.len() != o * r {
                return Err(anyhow!("adapter {w}: shape mismatch"));
            }
            for oi in 0..o {
                for ri in 0..r {
                    let s = b[oi * r + ri];
                    for ci in 0..c {
                        params[i][oi * c + ci] += s * a[ri * c + ci];
                    }
                }
            }
        }
        Ok(())
    }
    pub fn name(&self) -> String {
        self.meta.get("name").cloned().unwrap_or_else(|| "voice".into())
    }
    pub fn num(&self, k: &str) -> Option<f64> {
        self.meta.get(k).and_then(|v| v.parse().ok())
    }
    /// Metadata from the CLI: --name, --pitch, --rate, --accent.
    pub fn meta_from(a: &Args, source: &str) -> HashMap<String, String> {
        let mut m = HashMap::from([("format".to_string(), "ribt-voice-1".to_string()), ("source".to_string(), source.to_string())]);
        for k in ["name", "pitch", "rate", "accent", "nfe", "temp"] {
            if let Some(v) = a.opt(&format!("--{k}")) {
                m.insert(k.into(), v);
            }
        }
        m
    }
}

pub fn voice_blend(a: &Args) -> Result<()> {
    let va = Voice::load(Path::new(&a.need("--a")?))?;
    let vb = Voice::load(Path::new(&a.need("--b")?))?;
    let t: f32 = a.get("--t", 0.5);
    let mix = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (1.0 - t) * p + t * q).collect::<Vec<f32>>();
    let mut meta = va.meta.clone();
    meta.insert("name".into(), a.opt("--name").unwrap_or_else(|| format!("{}~{}@{t}", va.name(), vb.name())));
    meta.insert("source".into(), format!("blend {} {} t={t}", va.name(), vb.name()));
    // adapters blend exactly by stacking ranks: (1−t)·b₁a₁ + t·b₂a₂ = [(1−t)b₁ | t·b₂]·[a₁; a₂]
    let mut lora = Vec::new();
    let mut ws: Vec<String> = va.lora.iter().chain(&vb.lora).filter_map(|l| l.0.strip_prefix(LORA)?.strip_suffix(".a").map(String::from)).collect();
    ws.sort();
    ws.dedup();
    for w in ws {
        let (pa, pb) = (va.adapter(&w), vb.adapter(&w));
        let (r1, r2) = (pa.map_or(0, |p| p.2), pb.map_or(0, |p| p.2));
        let one = pa.or(pb).unwrap();
        let (c, o) = (one.0.len() / one.2, one.1.len() / one.2);
        let mut a = Vec::with_capacity((r1 + r2) * c);
        a.extend(pa.map_or(&[][..], |p| p.0));
        a.extend(pb.map_or(&[][..], |p| p.0));
        let mut b = vec![0.0f32; o * (r1 + r2)];
        for oi in 0..o {
            for ri in 0..r1 {
                b[oi * (r1 + r2) + ri] = (1.0 - t) * pa.unwrap().1[oi * r1 + ri];
            }
            for ri in 0..r2 {
                b[oi * (r1 + r2) + r1 + ri] = t * pb.unwrap().1[oi * r2 + ri];
            }
        }
        let (an, bn) = acoustic::adapter_names(&w);
        lora.push((an, vec![r1 + r2, c], a));
        lora.push((bn, vec![o, r1 + r2], b));
    }
    let out = Voice { se: mix(&va.se, &vb.se), sd: mix(&va.sd, &vb.sd), lora, meta };
    out.save(Path::new(&a.need("--out")?))?;
    eprintln!("blend: {} (1−{t}) + {} ({t}) → {}", va.name(), vb.name(), out.name());
    Ok(())
}

/// DTW path between `a` `[m, ta]` and `b` `[m, tb]` (L1 / m): for each b
/// frame, the first a frame it is matched to.
fn dtw_map(a: &[f32], ta: usize, sa: usize, b: &[f32], tb: usize, sb: usize, m: usize) -> Vec<usize> {
    let cost = |i: usize, j: usize| (0..m).map(|c| (a[c * sa + i] - b[c * sb + j]).abs()).sum::<f32>() / m as f32;
    let w = tb + 1;
    let mut d = vec![f32::INFINITY; (ta + 1) * w];
    d[0] = 0.0;
    for i in 1..=ta {
        for j in 1..=tb {
            let best = d[(i - 1) * w + j - 1].min(d[(i - 1) * w + j]).min(d[i * w + j - 1]);
            d[i * w + j] = best + cost(i - 1, j - 1);
        }
    }
    let mut map = vec![usize::MAX; tb];
    let (mut i, mut j) = (ta, tb);
    while i > 0 && j > 0 {
        map[j - 1] = map[j - 1].min(i - 1);
        let (dg, up, lf) = (d[(i - 1) * w + j - 1], d[(i - 1) * w + j], d[i * w + j - 1]);
        if dg <= up && dg <= lf {
            i -= 1;
            j -= 1;
        } else if up <= lf {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    map
}

struct Clip {
    utt: Utt,
    mel: Vec<f32>,
}

/// Aligner output for the fit clips: `id TAB ids` from the phonemized corpus and
/// `id TAB … TAB durations` from `ribt-align durs`. Tokens come from the phon file,
/// so a clip's tokens and its durations always line up; the lilt phonemizer
/// disagrees with the aligner's on ~10% of clips, which is what breaks the pairing.
fn read_align(phon: Option<&str>, durs: Option<&str>) -> Result<HashMap<String, (Vec<u16>, Vec<u32>)>> {
    let mut m: HashMap<String, (Vec<u16>, Vec<u32>)> = HashMap::new();
    if let Some(p) = phon {
        for l in std::fs::read_to_string(p)?.lines().filter(|l| !l.trim().is_empty()) {
            let c: Vec<&str> = l.split('\t').collect();
            let ids: Vec<u16> = c.get(1).map(|s| s.split(' ').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
            m.insert(c[0].into(), (ids, Vec::new()));
        }
    }
    if let Some(d) = durs {
        for l in std::fs::read_to_string(d)?.lines() {
            if l.starts_with('#') {
                continue;
            }
            let c: Vec<&str> = l.split('\t').collect();
            if c.len() < 6 {
                continue;
            }
            let ds: Vec<u32> = c[5].split(' ').filter_map(|x| x.parse().ok()).collect();
            m.entry(c[0].into()).or_default().1 = ds;
        }
    }
    Ok(m)
}

/// Load `id<TAB>…<TAB>text` rows: tokens, audio, F0/energy, log-mel, and durations.
/// With `--phon`/`--durs` the aligner's durations are used as-is; otherwise they
/// are a DTW of the clip against `vparams`' own decoder mel, which misaligns.
fn load_clips(dev: &Device, tc: &TConfig, vparams: &[Vec<f32>], rows: &str, wav_dir: &Path, fe: &mut Frontend, tts: &mut HashMap<(usize, usize), Tts>, align: &HashMap<String, (Vec<u16>, Vec<u32>)>) -> Result<Vec<Clip>> {
    let mut out = Vec::new();
    let (mut n_al, mut n_dtw) = (0, 0);
    for l in std::fs::read_to_string(rows)?.lines().filter(|l| !l.trim().is_empty()) {
        let c: Vec<&str> = l.split('\t').collect();
        let (id, text) = (c[0], c[c.len() - 1]);
        let (x, sr) = ribt::data::read_wav(&wav_dir.join(format!("{id}.wav")))?;
        if sr as usize != SR {
            return Err(anyhow!("{id}: {sr} Hz"));
        }
        let frames = x.len() / HOP;
        if frames > fe.frames {
            eprintln!("{id}: {frames} frames > {}, skipped", fe.frames);
            continue;
        }
        let full = fe.run(dev, &[x[..frames * HOP].to_vec()]);
        let mel: Vec<f32> = (0..tc.n_mels).flat_map(|ch| full[ch * fe.frames..ch * fe.frames + frames].to_vec()).collect();
        // the aligner's own tokens and durations, when it covered this clip
        let al = align.get(id).filter(|(t, d)| {
            let s: usize = d.iter().map(|&v| v as usize).sum();
            t.len() >= 3 && t.len() == d.len() && s.abs_diff(frames) <= 2
        });
        if let Some((t, d)) = al {
            // the two frame counts are the same wav read twice; absorb a 1–2 frame difference
            let mut d = d.clone();
            let s: usize = d.iter().map(|&v| v as usize).sum();
            let last = d.len() - 2;
            d[last] = (d[last] as i64 + frames as i64 - s as i64).max(1) as u32;
            let (f0, en) = frame_feats(&x, frames);
            let audio: Vec<i16> = x[..frames * HOP].iter().map(|v| (v * 32768.0).clamp(-32768.0, 32767.0) as i16).collect();
            out.push(Clip { utt: Utt { id: id.into(), spk: String::new(), toks: t.clone(), durs: d.clone(), audio, f0, en }, mel });
            n_al += 1;
            continue;
        }
        n_dtw += 1;
        let (ps, _) = ribt::lilt::phonemize(text);
        let mut toks: Vec<u16> = vec![0];
        toks.extend(ribt::lilt::vocab::ids(&ps).iter().map(|&v| v as u16));
        toks.push(0);
        // predicted durations, scaled to the clip, then DTW of the clip onto that grid
        let n = toks.len().div_ceil(32) * 32;
        let zr = (vec![0.0f32; tc.n_mels * REF_T], vec![0.0f32; REF_T]);
        let (oh, tm) = onehot_rows(tc, &[&toks], n);
        let e = tts.entry((n, 0)).or_insert_with(|| Tts::with(dev, tc, vparams, Dims { batch: 1, n, t: 64, r: REF_T }, true));
        let [h, ld, pp, ep, spk] = e.encode(dev, &oh, &tm, &zr.0, &zr.1);
        let raw: f64 = (0..toks.len()).map(|i| dur_of(ld[i], 1.0) as f64).sum();
        let durs_p: Vec<u32> = (0..toks.len()).map(|i| dur_of(ld[i], frames as f64 / raw.max(1.0))).collect();
        let pf: usize = durs_p.iter().sum::<u32>() as usize;
        let t = pf.div_ceil(64) * 64;
        let e = tts.entry((n, t)).or_insert_with(|| Tts::with(dev, tc, vparams, Dims { batch: 1, n, t, r: REF_T }, true));
        let al = acoustic::align_matrix(&[&durs_p], n, t);
        let mut fm = vec![0.0f32; t];
        fm[..pf].iter_mut().for_each(|v| *v = 1.0);
        let (pmel, _, _) = e.decode(dev, &h, &pp, &ep, &tm, &al, &fm, &spk);
        let map = dtw_map(&pmel, pf, t, &mel, frames, frames, tc.n_mels);
        let mut tok_of = Vec::with_capacity(pf);
        for (k, &dk) in durs_p.iter().enumerate() {
            tok_of.extend(std::iter::repeat(k).take(dk as usize));
        }
        let mut durs = vec![0u32; toks.len()];
        for &i in &map {
            durs[tok_of[i.min(pf - 1)]] += 1;
        }
        let (f0, en) = frame_feats(&x, frames);
        let audio: Vec<i16> = x[..frames * HOP].iter().map(|v| (v * 32768.0).clamp(-32768.0, 32767.0) as i16).collect();
        out.push(Clip { utt: Utt { id: id.into(), spk: String::new(), toks, durs, audio, f0, en }, mel });
    }
    if !align.is_empty() {
        eprintln!("durations: {n_al} clips from the aligner, {n_dtw} by DTW (aligner missing or token count differs)");
    }
    Ok(out)
}

pub fn voice_fit(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let run = PathBuf::from(a.need("--run")?);
    let (params, nm, _) = load_run(&run, &tc, a.opt("--ckpt"))?;
    let init = Voice::load(Path::new(&a.need("--voice")?))?;
    let out = PathBuf::from(a.need("--out")?);
    let wav_dir = PathBuf::from(a.need("--wav-dir")?);
    let steps: usize = a.get("--steps", 300);
    let peak: f64 = a.get("--lr", 3e-3);
    let batch: usize = a.get("--batch", 8);
    let tmax: usize = a.get("--t", 1024);
    // --rank R > 0 also trains a rank-R decoder adapter (rung 3)
    let rank: usize = a.get("--rank", 0);
    if !init.lora.is_empty() {
        return Err(anyhow!("--voice already has an adapter; start from a vectors-only voice"));
    }
    let specs = acoustic::voice_specs_r(&tc, rank);
    let nb = tc.params().len();
    let mut all = params;
    all.push(init.se.clone());
    all.push(init.sd.clone());
    let dev = Device::new();
    let mut fe = Frontend::new(&dev, 1, tmax);
    let mut tts = HashMap::new();
    let t0 = Instant::now();
    // --phon/--durs: the aligner's tokens and durations, instead of DTW against the model
    let al = read_align(a.opt("--phon").as_deref(), a.opt("--durs").as_deref())?;
    let fit = load_clips(&dev, &tc, &all, &a.need("--lines")?, &wav_dir, &mut fe, &mut tts, &al)?;
    let held = match a.opt("--held") {
        Some(h) => load_clips(&dev, &tc, &all, &h, &wav_dir, &mut fe, &mut tts, &al)?,
        None => Vec::new(),
    };
    drop(tts);
    let mut irng = Rng::new(a.get("--seed", 0));
    all.extend(specs[nb + 2..].iter().map(|s| s.init_data(&mut irng)));
    let med_f0 = |cs: &[Clip]| {
        let mut v: Vec<f32> = cs.iter().flat_map(|c| c.utt.f0.iter().copied().filter(|&f| f as f64 >= ribt::dsp::F0_MIN)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    eprintln!("voice-fit: {} fit clips, {} held, target median F0 {:.0} Hz; aligned in {:.1} s", fit.len(), held.len(), med_f0(&fit), t0.elapsed().as_secs_f64());
    let nmax = fit.iter().chain(&held).map(|c| c.utt.toks.len()).max().unwrap_or(32).div_ceil(32) * 32;
    let tm_ = fit.iter().chain(&held).map(|c| c.utt.frames()).max().unwrap_or(64).div_ceil(64) * 64;
    let d = Dims { batch, n: nmax, t: tm_, r: REF_T };
    let vf = acoustic::voice_fit_graph(&tc, d, rank, !a.0.iter().any(|x| x == "--no-f0-loss"));
    let (am, av): (Vec<Vec<f32>>, Vec<Vec<f32>>) = specs.iter().map(|s| (vec![0.0; s.numel()], vec![0.0; s.numel()])).unzip();
    let zr = (vec![0.0f32; batch * tc.n_mels * REF_T], vec![0.0f32; batch * REF_T]);
    let inputs = acoustic::assemble(&vf.tg.b, &specs, &all, Some((&am, &av)), &[("refmel", &zr.0), ("rw", &zr.1)]);
    let tcomp = Instant::now();
    let mut prog = Program::new(&dev, &vf.tg.b.g, &vf.outputs, &refs(&inputs));
    drop(inputs);
    eprintln!("voice-fit program {:?}: {} kernels, compile {:.1} s", d, prog.kernels(), tcomp.elapsed().as_secs_f64());
    let b = &vf.tg.b;
    let dmap = input_map(b);
    let k_hyper = b.input_index(vf.hyper);
    let k_p: Vec<usize> = vf.tg.params[nb..].iter().map(|&t| b.input_index(t)).collect();
    let k_m: Vec<usize> = vf.m.iter().map(|&t| b.input_index(t)).collect();
    let k_v: Vec<usize> = vf.v.iter().map(|&t| b.input_index(t)).collect();
    let nv = k_p.len();
    let (o_p, o_m, o_v) = (1 + N_LOSS, 1 + N_LOSS + nv, 1 + N_LOSS + 2 * nv);
    let fill = |prog: &mut Program, cs: &[&Clip]| {
        let us: Vec<&Utt> = cs.iter().map(|c| &c.utt).collect();
        let hb = host_batch(&tc, d, &us, &nm, &vec![0; batch], &vec![0; batch]);
        let mut mel = vec![0.0f32; batch * tc.n_mels * d.t];
        for (bi, c) in cs.iter().enumerate() {
            let fr = c.utt.frames();
            for ch in 0..tc.n_mels {
                mel[(bi * tc.n_mels + ch) * d.t..(bi * tc.n_mels + ch) * d.t + fr].copy_from_slice(&c.mel[ch * fr..(ch + 1) * fr]);
            }
        }
        for (k, v) in hb.refs() {
            prog.write_input(&dev, dmap[k], v);
        }
        prog.write_input(&dev, dmap["mel"], &mel);
    };
    // mean losses over `cs` in batches (short last batch repeats rows), lr 0, no swap
    let evaluate = |prog: &mut Program, cs: &[Clip]| -> [f64; N_LOSS] {
        let mut acc = [0.0f64; N_LOSS];
        let mut k = 0.0f64;
        prog.write_input(&dev, k_hyper, &[0.0, 0.1, 0.02, 0.0, 1.0, 0.0]);
        for ch in cs.chunks(batch) {
            let sel: Vec<&Clip> = (0..batch).map(|j| &ch[j % ch.len()]).collect();
            fill(prog, &sel);
            prog.run(&dev);
            for (i, x) in acc.iter_mut().enumerate() {
                *x += prog.read(&dev, 1 + i)[0] as f64;
            }
            k += 1.0;
        }
        acc.map(|x| x / k.max(1.0))
    };
    let show = |tag: &str, l: &[f64; N_LOSS]| {
        eprintln!("{tag:<12} total {:.4}  mel {:.4}  dur {:.4}  pitch {:.4}  energy {:.4}  f0 {:.4}  vuv {:.4}", l[0], l[1], l[2], l[3], l[4], l[5], l[6]);
    };
    let (f0_, h0) = (evaluate(&mut prog, &fit), evaluate(&mut prog, &held));
    show("fit  before", &f0_);
    if !held.is_empty() {
        show("held before", &h0);
    }
    let mut log = std::fs::File::create(out.with_extension("log.tsv"))?;
    writeln!(log, "step\ttotal\tmel_l1\tdur\tpitch\tenergy\tf0_l1\tvuv_bce\tgrad_norm\tlr")?;
    let mut rng = Rng::new(7);
    let ts = Instant::now();
    // with --held: keep the params at the lowest held total (checked every
    // --check steps; step 0 and the last step count; --check 0: keep the last),
    // since a small clip set overfits fast
    let check: usize = a.get("--check", 25);
    let snap = |prog: &Program| -> Vec<Vec<f32>> { k_p.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect() };
    let mut best = (!held.is_empty() && check > 0).then(|| (h0[0], 0usize, snap(&prog)));
    for step in 0..steps {
        let sel: Vec<&Clip> = (0..batch).map(|_| &fit[rng.below(fit.len())]).collect();
        fill(&mut prog, &sel);
        let lr = lr_at(step, steps, peak, steps / 20);
        let t = (step + 1) as i32;
        prog.write_input(&dev, k_hyper, &[lr as f32, (1.0 - 0.9f64.powi(t)) as f32, (1.0 - 0.98f64.powi(t)) as f32, 0.0, 1.0, 0.0]);
        prog.run(&dev);
        let gn = prog.read(&dev, 0)[0];
        let ls: Vec<f32> = (0..N_LOSS).map(|i| prog.read(&dev, 1 + i)[0]).collect();
        if !(gn.is_finite() && ls[0].is_finite()) {
            eprintln!("step {}: non-finite, skipped", step + 1);
            continue;
        }
        for i in 0..nv {
            prog.swap_input_output(k_p[i], o_p + i);
            prog.swap_input_output(k_m[i], o_m + i);
            prog.swap_input_output(k_v[i], o_v + i);
        }
        writeln!(log, "{}\t{}\t{gn:.4}\t{lr:.3e}", step + 1, ls.iter().map(|x| format!("{x:.5}")).collect::<Vec<_>>().join("\t"))?;
        if (step + 1) % 50 == 0 {
            eprintln!("step {:>4}  total {:.4}  mel {:.4}  |g| {gn:.3}  lr {lr:.2e}  {:.0} ms/step", step + 1, ls[0], ls[1], ts.elapsed().as_secs_f64() * 1e3 / (step + 1) as f64);
        }
        if let (Some(b), true) = (&mut best, check > 0 && (step + 1) % check == 0) {
            let h = evaluate(&mut prog, &held);
            if h[0] < b.0 {
                *b = (h[0], step + 1, snap(&prog));
            }
        }
    }
    let (f1, h1) = (evaluate(&mut prog, &fit), evaluate(&mut prog, &held));
    show("fit  after", &f1);
    if !held.is_empty() {
        show("held after", &h1);
    }
    let fin = match best {
        Some((h, _, _)) if h1[0] < h => snap(&prog),
        Some((h, st, ps)) => {
            eprintln!("kept step {st}: held total {h:.4}");
            ps
        }
        None => snap(&prog),
    };
    let (se, sd) = (fin[0].clone(), fin[1].clone());
    let dist = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum::<f32>().sqrt();
    let norm = |x: &[f32]| x.iter().map(|p| p * p).sum::<f32>().sqrt();
    eprintln!("moved: |Δse| {:.3} (|se| {:.3}), |Δsd| {:.3} (|sd| {:.3})", dist(&se, &init.se), norm(&init.se), dist(&sd, &init.sd), norm(&init.sd));
    let mut meta = init.meta.clone();
    meta.insert("source".into(), format!("fit {steps} steps lr {peak} rank {rank} on {} clips from {}", fit.len(), init.name()));
    let lora = (2..nv).map(|i| (specs[nb + i].name.clone(), specs[nb + i].shape.clone(), fin[i].clone())).collect();
    if let Some(nm_) = a.opt("--name") {
        meta.insert("name".into(), nm_);
    }
    // flow defaults for synth: the af-only student needs more steps on other
    // voices (4 steps, temp 0.3 unless --nfe / --temp say otherwise)
    for (k, d) in [("nfe", "4"), ("temp", "0.3")] {
        match a.opt(&format!("--{k}")) {
            Some(v) => meta.insert(k.into(), v),
            None => meta.insert(k.into(), meta.get(k).cloned().unwrap_or_else(|| d.into())),
        };
    }
    Voice { se, sd, lora, meta }.save(&out)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

pub fn voice_resynth(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let voc_path = PathBuf::from(a.need("--voc")?);
    let wav_dir = PathBuf::from(a.need("--wav-dir")?);
    let out = PathBuf::from(a.need("--out-dir")?);
    std::fs::create_dir_all(&out)?;
    let n: usize = a.get("--n", 4);
    let dev = Device::new();
    let mut fe = Frontend::new(&dev, 1, a.get("--t", 1024));
    for l in std::fs::read_to_string(a.need("--lines")?)?.lines().filter(|l| !l.trim().is_empty()).take(n) {
        let id = l.split('\t').next().unwrap();
        let (x, sr) = ribt::data::read_wav(&wav_dir.join(format!("{id}.wav")))?;
        let frames = x.len() / HOP;
        if sr as usize != SR || frames > fe.frames {
            eprintln!("{id}: skipped");
            continue;
        }
        let full = fe.run(&dev, &[x[..frames * HOP].to_vec()]);
        let mel: Vec<f32> = (0..tc.n_mels).flat_map(|ch| full[ch * fe.frames..ch * fe.frames + frames].to_vec()).collect();
        let (f0, _) = frame_feats(&x, frames);
        let mut voc = Vocoder::new(&dev, &voc_path, frames)?;
        if voc.is_hid() {
            return Err(anyhow!("hidden-input vocoder: no mel input"));
        }
        let y = voc.run(&dev, &mel, &f0, frames);
        ribt::data::write_wav(&out.join(format!("{id}.wav")), &y, SR as u32)?;
        let v: Vec<f64> = f0.iter().filter(|&&f| f as f64 >= ribt::dsp::F0_MIN).map(|&f| 12.0 * (f as f64).log2()).collect();
        let mu = v.iter().sum::<f64>() / v.len().max(1) as f64;
        let sd = (v.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / v.len().max(1) as f64).sqrt();
        eprintln!("{id}: {frames} frames, voiced {:.2}, F0 sd {sd:.2} st", v.len() as f64 / frames as f64);
    }
    Ok(())
}

pub fn voice_probe(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let run = PathBuf::from(a.need("--run")?);
    let (mut params, nm, _) = load_run(&run, &tc, a.opt("--ckpt"))?;
    let v = Voice::load(Path::new(&a.need("--voice")?))?;
    v.merge(&tc, &mut params)?;
    params.extend([v.se.clone(), v.sd.clone()]);
    let voc_path = PathBuf::from(a.need("--voc")?);
    let wav_dir = PathBuf::from(a.need("--wav-dir")?);
    let out = PathBuf::from(a.need("--out-dir")?);
    std::fs::create_dir_all(&out)?;
    let fc = ribt::flow::FConfig::by_name("base").ok_or_else(|| anyhow!("fcfg"))?;
    let flow = match a.opt("--flow") {
        Some(f) => Some((ribt::ckpt::load_specs(Path::new(&f), &fc.params(), false)?.params, flow_cmd::FNorm::load(Path::new(&f))?)),
        None => None,
    };
    let (nfe, temp, seed): (usize, f64, u64) = (a.get("--nfe", 4), a.get("--temp", 0.3), a.get("--seed", 0));
    let shortcut = a.0.iter().any(|x| x == "--shortcut");
    let dev = Device::new();
    let mut fe = Frontend::new(&dev, 1, a.get("--t", 1024));
    let mut tts = HashMap::new();
    let clips = load_clips(&dev, &tc, &params, &a.need("--lines")?, &wav_dir, &mut fe, &mut tts, &HashMap::new())?;
    let zr = (vec![0.0f32; tc.n_mels * REF_T], vec![0.0f32; REF_T]);
    for c in clips.iter().take(a.get("--n", 4)) {
        let (toks, frames) = (&c.utt.toks, c.utt.frames());
        let n = toks.len().div_ceil(32) * 32;
        let t = frames.div_ceil(64) * 64;
        let (oh, tm) = onehot_rows(&tc, &[toks], n);
        let e = tts.entry((n, 0)).or_insert_with(|| Tts::with(&dev, &tc, &params, Dims { batch: 1, n, t: 64, r: REF_T }, true));
        let [h, _, pp, ep, spk] = e.encode(&dev, &oh, &tm, &zr.0, &zr.1);
        let e = tts.entry((n, t)).or_insert_with(|| Tts::with(&dev, &tc, &params, Dims { batch: 1, n, t, r: REF_T }, true));
        let al = acoustic::align_matrix(&[&c.utt.durs], n, t);
        let mut fm = vec![0.0f32; t];
        fm[..frames].iter_mut().for_each(|v| *v = 1.0);
        let (dmel, f0, hid) = e.decode(&dev, &h, &pp, &ep, &tm, &al, &fm, &spk);
        let trim = |m: &[f32]| -> Vec<f32> { (0..tc.n_mels).flat_map(|ch| m[ch * t..ch * t + frames].to_vec()).collect() };
        let pf0: Vec<f32> = (0..frames).map(|f| if f0[t + f] > 0.0 { ((f0[f] as f64) * nm.f0_sd + nm.f0_mu).exp() as f32 } else { 0.0 }).collect();
        let rf0: Vec<f32> = c.utt.f0.iter().map(|&f| if f as f64 >= ribt::dsp::F0_MIN { f } else { 0.0 }).collect();
        let mut voc = Vocoder::new(&dev, &voc_path, frames)?;
        let mut put = |tag: &str, mel: &[f32], f0: &[f32]| -> Result<()> {
            let y = voc.run(&dev, mel, f0, frames);
            ribt::data::write_wav(&out.join(format!("{}.{tag}.wav", c.utt.id)), &y, SR as u32)?;
            Ok(())
        };
        put("realmel_realf0", &c.mel, &rf0)?;
        put("realmel_predf0", &c.mel, &pf0)?;
        let dm = trim(&dmel);
        put("decmel_realf0", &dm, &rf0)?;
        put("decmel_predf0", &dm, &pf0)?;
        if let Some((fp, fnm)) = &flow {
            let mut vel = flow_cmd::Vel::new(&dev, &fc, fp, ribt::flow::FDims { batch: 1, t });
            let mun = fnm.norm(&dmel, &fm, 1, t);
            let ab = ribt::flow::attn_bias(&[frames], t, None);
            let x0 = flow_cmd::noise(&mut Rng::new(seed), &fm, 1, tc.n_mels, t, temp);
            vel.set_cond(&dev, &mun, &hid, &fm, &ab);
            let (x1, _) = vel.sample(&dev, &x0, nfe, shortcut);
            let fl = trim(&fnm.denorm(&x1));
            put("flowmel_realf0", &fl, &rf0)?;
            put("flowmel_predf0", &fl, &pf0)?;
        }
        // F0 agreement on frames both call voiced; voicing agreement
        let both: Vec<f64> = (0..frames).filter(|&f| pf0[f] > 0.0 && rf0[f] > 0.0).map(|f| (12.0 * (pf0[f] as f64 / rf0[f] as f64).log2()).abs()).collect();
        let vp = pf0.iter().filter(|&&f| f > 0.0).count() as f64 / frames as f64;
        let vr = rf0.iter().filter(|&&f| f > 0.0).count() as f64 / frames as f64;
        let agree = (0..frames).filter(|&f| (pf0[f] > 0.0) == (rf0[f] > 0.0)).count() as f64 / frames as f64;
        let mut srt = both.clone();
        srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("{}: voiced pred {vp:.2} real {vr:.2}, voicing agreement {agree:.2}, |ΔF0| median {:.2} st", c.utt.id, srt.get(srt.len() / 2).copied().unwrap_or(0.0));
    }
    Ok(())
}
