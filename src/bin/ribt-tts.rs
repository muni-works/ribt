//! `ribt-tts`: the V2 acoustic model, first rung (the ribt paper).
//! GPU work should run under `~/.cache/gpu-lock.sh` on shared machines.
//!
//!   ribt-tts train --phon A.phon.tsv[,B…] --durs D.durs.tsv[,…] --root LIBRITTS_R --run DIR
//!                  [--held test-clean.phon.tsv] [--steps N] [--total N] [--batch 16] [--n 192] [--t 1024]
//!                  [--lr 1e-3] [--warmup 1000] [--init CKPT]  (touch RUN/STOP to save + exit)
//!                  --init: params from CKPT, fresh Adam, step 0 (speaker fine-tune, §20); put the
//!                  base run's norm.txt in RUN first so pitch/energy normalization is unchanged
//!   ribt-tts eval  --run DIR --held test-clean.phon.tsv --durs D --root R [--n-utts 64] [--out-dir DIR]
//!                  held-out mel L1: teacher-forced (GT durations) and predicted durations (DTW)
//!   ribt-tts synth --run DIR --voc VOCODER.safetensors [--ref REF.wav (default: speaker 5163)] --text "…" --out OUT.wav
//!                  (--ref A.wav+B.wav+…: one mean speaker vector, the reference encoder's masked mean
//!                  pool over all the clips' crops laid end to end; §20)
//!                  [--pitch-range R] token pitch spread around the utterance mean × R (synth)
//!                  [--pitch SEMITONES] global pitch offset; [--rate R] speak R× faster (synth and stream)
//!                  (or --lines FILE with one `name<TAB>text` per line and --out-dir DIR)
//!                  [--flow FLOW.safetensors --nfe 16 [--shortcut] [--temp 1.0] [--seed 0]]  rung-2 decoder
//!                  [--ctx CTX.safetensors]  context-aware prosody (§25): --lines rows are
//!                  `name, text, prev, role[, emph]`, see src/bin/ribt-tts/ctx.rs
//!                  [--voice V.voice] a voice file instead of --ref; [--save-voice V.voice] write
//!                  the --ref vectors as one (src/bin/ribt-tts/voice.rs, also voice-blend, voice-fit)
//!   ribt-tts ctx-train …  context path training (§25), src/bin/ribt-tts/ctx.rs
//!   ribt-tts flow-train | flow-eval   rung 2, see src/bin/ribt-tts/flow_cmd.rs
//!   ribt-tts voc-pairs …  predicted-mel/real-audio pairs for the vocoder fine-tune (src/bin/ribt-tts/vocft.rs)
//!   ribt-tts diffro …    §29.4 differentiable reward optimization of the flow student (src/bin/ribt-tts/reward_cmd.rs)
//!   ribt-tts stream …    streaming chunked synthesis, TTFA/RTF/underruns (src/bin/ribt-tts/stream.rs)
//!   ribt-tts anatomy …   dump every synth tensor for scripts/anatomy_video.py (src/bin/ribt-tts/anatomy.rs)
//!   ribt-tts check [--cfg check]       one train step, GPU vs f64 interpreter
//!   ribt-tts bench [--batch 16] [--n 192] [--t 1024]   train-step ms on random data
//!
//! Training reads the misaki token rows (`*.phon.tsv`, Kokoro ids, both pads)
//! and the aligner's MAS durations (`ribt-align durs`); F0 (YIN) and frame
//! energy are computed from the audio at load.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use ribt::acoustic::{self, Dims, TConfig, N_LOSS};
use ribt::ckpt::{self, Checkpoint};
use ribt::model::{self, Config, Rng};
use ribt::runtime::{Device, Program};
use ribt::{dsp, lilt};

#[path = "ribt-tts/flow_cmd.rs"]
mod flow_cmd;
#[path = "ribt-tts/anatomy.rs"]
mod anatomy;
#[path = "ribt-tts/vocft.rs"]
mod vocft;
#[path = "ribt-tts/stream.rs"]
pub mod stream;
#[path = "ribt-tts/ctx.rs"]
mod ctx;
#[path = "ribt-tts/bodyv.rs"]
mod bodyv;
#[path = "ribt-tts/reward_cmd.rs"]
mod reward_cmd;
#[path = "ribt-tts/voice.rs"]
mod voice;

pub struct Args(pub Vec<String>);

impl Args {
    fn opt(&self, k: &str) -> Option<String> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1).cloned())
    }
    fn get<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        self.opt(k).and_then(|v| v.parse().ok()).unwrap_or(d)
    }
    fn need(&self, k: &str) -> Result<String> {
        self.opt(k).ok_or_else(|| anyhow!("{k} required"))
    }
    fn cfg(&self) -> Result<TConfig> {
        let n = self.opt("--cfg").unwrap_or_else(|| "base".into());
        TConfig::by_name(&n).ok_or_else(|| anyhow!("unknown cfg {n}"))
    }
}

fn main() -> Result<()> {
    let a = Args(std::env::args().skip(1).collect());
    // --accent midwest[,ng] (lilt §26.7); default neutral, or RIBT_LILT_ACCENT
    if let Some(ac) = a.opt("--accent") {
        ribt::lilt::accent::set(&ac).map_err(|e| anyhow!(e))?;
    }
    match a.0.first().map(String::as_str) {
        Some("train") => train(&a),
        Some("eval") => eval(&a),
        Some("synth") => synth(&a),
        Some("check") => check(&a),
        Some("bench") => bench(&a),
        Some("flow-train") => flow_cmd::flow_train(&a),
        Some("ctx-train") => ctx::ctx_train(&a),
        Some("flow-eval") => flow_cmd::flow_eval(&a),
        Some("anatomy") => anatomy::anatomy(&a),
        Some("voc-pairs") => vocft::voc_pairs(&a),
        Some("hid-init") => vocft::hid_init(&a),
        Some("flow-check") => flow_cmd::flow_check(&a),
        Some("flow-bench") => flow_cmd::flow_bench(&a),
        Some("stream") => stream::stream(&a),
        Some("diffro") => reward_cmd::diffro(&a),
        Some("diffro-check") => reward_cmd::diffro_check(&a),
        Some("barge") => bodyv::barge(&a),
        Some("body") => bodyv::body(&a),
        Some("story") => bodyv::story(&a),
        Some("surprise") => bodyv::surprise(&a),
        Some("voice-blend") => voice::voice_blend(&a),
        Some("voice-fit") => voice::voice_fit(&a),
        Some("voice-resynth") => voice::voice_resynth(&a),
        Some("voice-probe") => voice::voice_probe(&a),
        _ => {
            eprintln!("usage: ribt-tts train|eval|synth|check|bench (see src/bin/ribt-tts.rs)");
            Ok(())
        }
    }
}

fn refs(v: &[Vec<f32>]) -> Vec<&[f32]> {
    v.iter().map(|x| x.as_slice()).collect()
}

fn lr_at(step: usize, total: usize, peak: f64, warmup: usize) -> f64 {
    if step < warmup {
        return peak * (step + 1) as f64 / warmup as f64;
    }
    let p = ((step - warmup) as f64 / (total.saturating_sub(warmup)).max(1) as f64).min(1.0);
    peak * (0.02 + 0.98 * 0.5 * (1.0 + (std::f64::consts::PI * p).cos()))
}

// ---------------------------------------------------------------------------
// data

const HOP: usize = 256;
const SR: usize = 24000;
/// reference crop (frames, ~2.7 s)
const REF_T: usize = 256;
/// Default demo voice: LibriTTS-R speaker 5163 (Cam's pick from the rung-1
/// listening set, ~234 Hz).
const DEMO_REF: &str = "~/data/libritts_r/LibriTTS_R/train-clean-100/5163/18515/5163_18515_000042_000001.wav";
/// frontend context around a training crop (frames)
const CTX: usize = 4;

struct Utt {
    id: String,
    spk: String,
    toks: Vec<u16>,
    durs: Vec<u32>,
    audio: Vec<i16>,
    /// per frame: YIN F0 (Hz, 0 unvoiced), ln frame energy
    f0: Vec<f32>,
    en: Vec<f32>,
}

impl Utt {
    fn frames(&self) -> usize {
        self.audio.len() / HOP
    }
    fn audio_f32(&self, from: usize, frames: usize) -> Vec<f32> {
        self.audio[from * HOP..(from + frames) * HOP].iter().map(|&v| v as f32 / 32768.0).collect()
    }
}

fn wav_path(root: &Path, split: &str, id: &str) -> PathBuf {
    let mut it = id.split('_');
    let (s, c) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
    root.join(split).join(s).join(c).join(format!("{id}.wav"))
}

fn read_durs(files: &str) -> Result<HashMap<String, Vec<u32>>> {
    let mut m = HashMap::new();
    for f in files.split(',').filter(|s| !s.is_empty()) {
        for line in std::fs::read_to_string(f)?.lines() {
            if line.starts_with('#') {
                continue;
            }
            let c: Vec<&str> = line.split('\t').collect();
            if c.len() < 6 {
                continue;
            }
            m.insert(c[0].to_string(), c[5].split(' ').filter_map(|x| x.parse().ok()).collect());
        }
    }
    Ok(m)
}

/// Frame features of a clip: YIN F0 and ln mean-square energy over an
/// n_fft window centred like the mel frame.
fn frame_feats(x: &[f32], frames: usize) -> (Vec<f32>, Vec<f32>) {
    let vc = Config::v1();
    let xp = dsp::reflect_pad(&x[..frames * HOP], vc.mel_pad());
    let f0 = dsp::yin_f0(&xp, SR, vc.n_fft, HOP, frames);
    let en = (0..frames)
        .map(|t| {
            let w = &xp[t * HOP..t * HOP + vc.n_fft];
            ((w.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / w.len() as f64) + 1e-8).ln() as f32
        })
        .collect();
    (f0, en)
}

/// Utterances of the phon tsvs that have durations and fit `keep`.
fn load(phon: &str, durs: &HashMap<String, Vec<u32>>, root: &Path, keep: &(dyn Fn(usize, usize) -> bool + Sync)) -> Result<Vec<Utt>> {
    let mut rows: Vec<(String, PathBuf, Vec<u16>, Vec<u32>)> = Vec::new();
    let mut nodur = 0;
    for f in phon.split(',').filter(|s| !s.is_empty()) {
        let split = Path::new(f).file_name().unwrap().to_string_lossy().trim_end_matches(".phon.tsv").to_string();
        for line in std::fs::read_to_string(f)?.lines() {
            let mut c = line.split('\t');
            let (Some(id), Some(ids)) = (c.next(), c.next()) else { continue };
            let toks: Vec<u16> = ids.split(' ').filter_map(|x| x.parse().ok()).collect();
            let Some(d) = durs.get(id) else {
                nodur += 1;
                continue;
            };
            if d.len() != toks.len() || !keep(toks.len(), d.iter().sum::<u32>() as usize) {
                continue;
            }
            rows.push((id.to_string(), wav_path(root, &split, id), toks, d.clone()));
        }
    }
    if nodur > 0 {
        eprintln!("{nodur} rows without durations skipped");
    }
    let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(14);
    let chunk = rows.len().div_ceil(nt).max(1);
    let parts: Vec<Result<Vec<Utt>>> = std::thread::scope(|s| {
        let hs: Vec<_> = rows
            .chunks(chunk)
            .map(|c| {
                s.spawn(move || {
                    let mut out = Vec::new();
                    for (id, p, toks, d) in c {
                        let Ok(mut r) = hound::WavReader::open(p) else {
                            eprintln!("missing wav, skipped: {}", p.display());
                            continue;
                        };
                        let spec = r.spec();
                        if spec.sample_rate != SR as u32 || spec.channels != 1 {
                            return Err(anyhow!("{}: {} Hz × {} ch", p.display(), spec.sample_rate, spec.channels));
                        }
                        let audio: Vec<i16> = match (spec.sample_format, spec.bits_per_sample) {
                            (hound::SampleFormat::Int, 16) => r.samples::<i16>().collect::<Result<_, _>>()?,
                            _ => r.samples::<f32>().map(|x| x.map(|v| (v * 32767.0).clamp(-32768.0, 32767.0) as i16)).collect::<Result<_, _>>()?,
                        };
                        let frames = audio.len() / HOP;
                        if d.iter().sum::<u32>() as usize != frames {
                            continue;
                        }
                        let x: Vec<f32> = audio.iter().map(|&v| v as f32 / 32768.0).collect();
                        let (f0, en) = frame_feats(&x, frames);
                        let spk = id.split('_').next().unwrap_or("").to_string();
                        out.push(Utt { id: id.clone(), spk, toks: toks.clone(), durs: d.clone(), audio, f0, en });
                    }
                    Ok(out)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = Vec::new();
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}

/// Global normalization: ln F0 (voiced frames) and ln energy, mean / sd.
#[derive(Clone, Copy, Debug)]
struct Norm {
    f0_mu: f64,
    f0_sd: f64,
    en_mu: f64,
    en_sd: f64,
}

impl Norm {
    fn of(data: &[Utt]) -> Self {
        let (mut a, mut a2, mut na, mut e, mut e2, mut ne) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for u in data {
            for (&f, &en) in u.f0.iter().zip(&u.en) {
                if f as f64 >= dsp::F0_MIN {
                    let l = (f as f64).ln();
                    a += l;
                    a2 += l * l;
                    na += 1.0;
                }
                e += en as f64;
                e2 += (en as f64) * (en as f64);
                ne += 1.0;
            }
        }
        let (fm, em) = (a / na, e / ne);
        Self { f0_mu: fm, f0_sd: (a2 / na - fm * fm).sqrt(), en_mu: em, en_sd: (e2 / ne - em * em).sqrt() }
    }
    fn save(&self, p: &Path) -> Result<()> {
        std::fs::write(p, format!("{} {} {} {}\n", self.f0_mu, self.f0_sd, self.en_mu, self.en_sd))?;
        Ok(())
    }
    fn load(p: &Path) -> Result<Self> {
        let v: Vec<f64> = std::fs::read_to_string(p)?.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        Ok(Self { f0_mu: v[0], f0_sd: v[1], en_mu: v[2], en_sd: v[3] })
    }
}

/// Host-side batch tensors (everything but the mels).
struct HostBatch {
    data: Vec<(&'static str, Vec<f32>)>,
}

impl HostBatch {
    fn get(&self, k: &str) -> &[f32] {
        &self.data.iter().find(|(n, _)| *n == k).unwrap().1
    }
    fn refs(&self) -> Vec<(&str, &[f32])> {
        self.data.iter().map(|(k, v)| (*k, v.as_slice())).collect()
    }
}

/// Token-level pitch / energy targets (normalized) of one utterance.
fn token_targets(u: &Utt, nm: &Norm) -> (Vec<f32>, Vec<f32>) {
    let mut p = Vec::with_capacity(u.durs.len());
    let mut e = Vec::with_capacity(u.durs.len());
    let mut pos = 0usize;
    for &d in &u.durs {
        let span = pos..pos + d as usize;
        let (mut ps, mut pn, mut es) = (0.0, 0.0, 0.0);
        for f in span.clone() {
            let f0 = u.f0[f] as f64;
            if f0 >= dsp::F0_MIN {
                ps += (f0.ln() - nm.f0_mu) / nm.f0_sd;
                pn += 1.0;
            }
            es += (u.en[f] as f64 - nm.en_mu) / nm.en_sd;
        }
        p.push(if pn > 0.0 { (ps / pn) as f32 } else { 0.0 });
        e.push(if d > 0 { (es / d as f64) as f32 } else { 0.0 });
        pos += d as usize;
    }
    (p, e)
}

fn host_batch(tc: &TConfig, d: Dims, utts: &[&Utt], nm: &Norm, ref_lens: &[usize], starts: &[usize]) -> HostBatch {
    let (b, n, t, r) = (d.batch, d.n, d.t, d.r);
    let mut onehot = vec![0.0f32; b * tc.vocab * n];
    let mut tmask = vec![0.0f32; b * n];
    let mut fmask = vec![0.0f32; b * t];
    let mut rw = vec![0.0f32; b * r];
    let mut p_tok = vec![0.0f32; b * n];
    let mut e_tok = vec![0.0f32; b * n];
    let mut ld = vec![0.0f32; b * n];
    let mut lf0 = vec![0.0f32; b * t];
    let mut vuv = vec![0.0f32; b * t];
    for (bi, u) in utts.iter().enumerate() {
        for (i, &id) in u.toks.iter().enumerate() {
            onehot[(bi * tc.vocab + id as usize) * n + i] = 1.0;
            tmask[bi * n + i] = 1.0;
            ld[bi * n + i] = (u.durs[i] as f32 + 1.0).ln();
        }
        let (p, e) = token_targets(u, nm);
        p_tok[bi * n..bi * n + p.len()].copy_from_slice(&p);
        e_tok[bi * n..bi * n + e.len()].copy_from_slice(&e);
        for j in 0..t.min(u.frames().saturating_sub(starts[bi])) {
            let f = starts[bi] + j;
            fmask[bi * t + j] = 1.0;
            let f0 = u.f0[f] as f64;
            if f0 >= dsp::F0_MIN {
                vuv[bi * t + j] = 1.0;
                lf0[bi * t + j] = ((f0.ln() - nm.f0_mu) / nm.f0_sd) as f32;
            }
        }
        for k in 0..ref_lens[bi] {
            rw[bi * r + k] = 1.0 / ref_lens[bi] as f32;
        }
    }
    let durs: Vec<&[u32]> = utts.iter().map(|u| u.durs.as_slice()).collect();
    let align = acoustic::align_matrix_from(&durs, starts, n, t);
    let ntok: f32 = tmask.iter().sum::<f32>().max(1.0);
    let nfr: f32 = fmask.iter().sum::<f32>().max(1.0);
    let nv: f32 = vuv.iter().sum::<f32>().max(1.0);
    let tw = tmask.iter().map(|m| m / ntok).collect();
    let fw = fmask.iter().map(|m| m / nfr).collect();
    let vw = vuv.iter().map(|m| m / nv).collect();
    HostBatch {
        data: vec![
            ("onehot", onehot),
            ("tmask", tmask),
            ("rw", rw),
            ("align", align),
            ("fmask", fmask),
            ("p_tok", p_tok),
            ("e_tok", e_tok),
            ("ld_tok", ld),
            ("tw", tw),
            ("lf0", lf0),
            ("vuv", vuv),
            ("fw", fw),
            ("vw", vw),
        ],
    }
}

/// GPU log-mel for a batch of clips (each reflect-padded at its own length,
/// zeros after): `[B, n_mels, frames]`.
struct Frontend {
    prog: Program,
    k_audio: usize,
    batch: usize,
    frames: usize,
}

impl Frontend {
    fn new(dev: &Device, batch: usize, frames: usize) -> Self {
        let vc = Config::v1();
        let fe = model::frontend_graph(&vc, batch, frames);
        let fin = fe.b.assemble(&[], None, &[]);
        let prog = Program::new(dev, &fe.b.g, &[fe.mel], &refs(&fin));
        Self { k_audio: fe.b.input_index(fe.audio), prog, batch, frames }
    }
    fn run(&mut self, dev: &Device, clips: &[Vec<f32>]) -> Vec<f32> {
        let vc = Config::v1();
        let pad = vc.mel_pad();
        let row = self.frames * HOP + 2 * pad;
        let mut xp = vec![0.0f32; self.batch * row];
        for (b, x) in clips.iter().enumerate() {
            let fr = (x.len() / HOP).min(self.frames);
            let p = dsp::reflect_pad(&x[..fr * HOP], pad);
            xp[b * row..b * row + p.len()].copy_from_slice(&p);
        }
        self.prog.write_input(dev, self.k_audio, &xp);
        self.prog.run(dev);
        self.prog.read(dev, 0)
    }
}

/// Reference crop for utterance `u`: another utterance of the same speaker
/// (or `u` itself when the speaker has one), `REF_T` frames at a random
/// offset (shorter clips: whole, masked).
fn pick_ref<'a>(rng: &mut Rng, u: &Utt, data: &'a [Utt], by_spk: &HashMap<String, Vec<usize>>) -> (&'a Utt, usize, usize) {
    let pool = &by_spk[&u.spk];
    let mut j = pool[rng.below(pool.len())];
    if pool.len() > 1 {
        while data[j].id == u.id {
            j = pool[rng.below(pool.len())];
        }
    }
    let r = &data[j];
    let fr = r.frames();
    let len = fr.min(REF_T);
    let off = if fr > REF_T { rng.below(fr - REF_T + 1) } else { 0 };
    (r, off, len)
}

fn by_speaker(data: &[Utt]) -> HashMap<String, Vec<usize>> {
    let mut m: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, u) in data.iter().enumerate() {
        m.entry(u.spk.clone()).or_default().push(i);
    }
    m
}


/// Mel + reference mel + host tensors for `us`: a random crop of `d.t`
/// frames (`crop`), the frontend sees CTX frames of context each side (so
/// only true utterance edges get reflect padding).
#[allow(clippy::too_many_arguments)]
fn prep_crops(dev: &Device, tc: &TConfig, d: Dims, nm: &Norm, rng: &mut Rng, crop: bool, us: &[&Utt], pool: &[Utt], pool_spk: &HashMap<String, Vec<usize>>, fe: &mut Frontend, fe_ref: &mut Frontend) -> (HostBatch, Vec<f32>, Vec<f32>) {
    let starts: Vec<usize> = us.iter().map(|u| if crop && u.frames() > d.t { rng.below(u.frames() - d.t + 1) } else { 0 }).collect();
    let mut offs = Vec::new();
    let clips: Vec<Vec<f32>> = us
        .iter()
        .zip(&starts)
        .map(|(u, &s)| {
            let s0 = s.saturating_sub(CTX);
            let s1 = (s + d.t + CTX).min(u.frames());
            offs.push(s - s0);
            u.audio_f32(s0, s1 - s0)
        })
        .collect();
    let full = fe.run(dev, &clips);
    let cap = fe.frames;
    let mut mel = vec![0.0f32; d.batch * tc.n_mels * d.t];
    for (b, &o) in offs.iter().enumerate() {
        for c in 0..tc.n_mels {
            let src = &full[(b * tc.n_mels + c) * cap + o..(b * tc.n_mels + c) * cap + o + d.t];
            mel[(b * tc.n_mels + c) * d.t..(b * tc.n_mels + c + 1) * d.t].copy_from_slice(src);
        }
    }
    let mut rclips = Vec::new();
    let mut rlens = Vec::new();
    for u in us {
        let (r, off, len) = pick_ref(rng, u, pool, pool_spk);
        rclips.push(r.audio_f32(off, len));
        rlens.push(len);
    }
    let refmel = fe_ref.run(dev, &rclips);
    (host_batch(tc, d, us, nm, &rlens, &starts), mel, refmel)
}

// ---------------------------------------------------------------------------
// train

fn input_map(b: &model::Built) -> HashMap<&'static str, usize> {
    b.inputs
        .iter()
        .enumerate()
        .filter_map(|(k, s)| match s.role {
            model::Role::Data(n) => Some((n, k)),
            _ => None,
        })
        .collect()
}

fn train(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let specs = tc.params();
    let run = PathBuf::from(a.need("--run")?);
    std::fs::create_dir_all(&run)?;
    let root = PathBuf::from(a.need("--root")?);
    let steps: usize = a.get("--steps", 1000);
    let total: usize = a.get("--total", 50000);
    let batch: usize = a.get("--batch", 16);
    let (nmax, umax): (usize, usize) = (a.get("--n", 192), a.get("--max-frames", 1024));
    // decoder crop (frames); the encoder always sees every token
    let tmax: usize = a.get("--t", 256);
    let peak: f64 = a.get("--lr", 1e-3);
    let warmup: usize = a.get("--warmup", 1000);
    let clip: f64 = a.get("--clip", 1.0);
    let log_every: usize = a.get("--log-every", 100);
    let ckpt_every: usize = a.get("--ckpt-every", 1000);
    let eval_every: usize = a.get("--eval-every", 2000);
    let keep_every: usize = a.get("--keep-every", 10000);
    let keep = |n: usize, t: usize| n <= nmax && t <= umax && t >= 16;
    let d = Dims { batch, n: nmax, t: tmax, r: REF_T };

    let t0 = Instant::now();
    let durs = read_durs(&a.need("--durs")?)?;
    let data = load(&a.need("--phon")?, &durs, &root, &keep)?;
    if data.is_empty() {
        return Err(anyhow!("no utterances"));
    }
    let secs: f64 = data.iter().map(|u| u.audio.len()).sum::<usize>() as f64 / SR as f64;
    let spk = by_speaker(&data);
    eprintln!("train: {} utts, {} speakers ({:.2} h), loaded + F0 in {:.1} s", data.len(), spk.len(), secs / 3600.0, t0.elapsed().as_secs_f64());
    let norm_path = run.join("norm.txt");
    let nm = if norm_path.exists() { Norm::load(&norm_path)? } else {
        let nm = Norm::of(&data);
        nm.save(&norm_path)?;
        nm
    };
    eprintln!("norm {nm:?}");
    let held: Vec<Utt> = match a.opt("--held") {
        Some(h) => {
            let v = load(&h, &durs, &root, &keep)?;
            // --held-n: strided held-out subset size (default 2 x batch; rounded down to whole batches)
            let hn: usize = (a.get("--held-n", 2 * batch) / batch).max(1) * batch;
            let step = (v.len() / hn).max(1);
            v.into_iter().step_by(step).take(hn).collect()
        }
        None => Vec::new(),
    };
    let held_spk = by_speaker(&held);
    eprintln!("held-out: {} utts", held.len());

    let ck_path = run.join("last.safetensors");
    // --init: fine-tune start (params only, fresh Adam, step 0); ignored once RUN/last exists
    let ck = if ck_path.exists() {
        ckpt::load_specs(&ck_path, &specs, false)?
    } else if let Some(i) = a.opt("--init") {
        Checkpoint { params: ckpt::load_specs(Path::new(&i), &specs, false)?.params, adam: None, step: 0 }
    } else {
        Checkpoint { params: acoustic::init_params(&tc, 0), adam: None, step: 0 }
    };
    eprintln!("cfg {tc:?}\nparams {} ({:.2}M)  start step {}", tc.n_params(), tc.n_params() as f64 / 1e6, ck.step);

    let dev = Device::new();
    let mut fe = Frontend::new(&dev, batch, tmax + 2 * CTX);
    let mut fe_ref = Frontend::new(&dev, batch, REF_T);
    let tt = acoustic::train_graph(&tc, d);
    let adam = ck.adam.as_ref().map(|(m, v)| (m.as_slice(), v.as_slice()));
    let inputs = acoustic::assemble(&tt.tg.b, &specs, &ck.params, adam, &[]);
    let tcomp = Instant::now();
    let mut prog = Program::new(&dev, &tt.tg.b.g, &tt.outputs, &refs(&inputs));
    drop(inputs);
    eprintln!("train program: {} kernels, compile {:.1} s", prog.kernels(), tcomp.elapsed().as_secs_f64());
    // forward-only program for held-out losses (params copied in at eval)
    let fg = acoustic::tts_graph(&tc, d);
    let finp = acoustic::assemble(&fg.b, &specs, &ck.params, None, &[]);
    let mut fprog = Program::new(&dev, &fg.b.g, &fg.losses, &refs(&finp));
    drop(finp);
    let fk_p: Vec<usize> = fg.params.iter().map(|&p| fg.b.input_index(p)).collect();
    let fmap = input_map(&fg.b);
    let b = &tt.tg.b;
    let np = specs.len();
    let dmap = input_map(b);
    let k_hyper = b.input_index(tt.hyper);
    let k_p: Vec<usize> = tt.tg.params.iter().map(|&t| b.input_index(t)).collect();
    let k_m: Vec<usize> = tt.m.iter().map(|&t| b.input_index(t)).collect();
    let k_v: Vec<usize> = tt.v.iter().map(|&t| b.input_index(t)).collect();
    let (o_p, o_m, o_v) = (1 + N_LOSS, 1 + N_LOSS + np, 1 + N_LOSS + 2 * np);

    let open_log = |name: &str, header: &str| -> Result<std::fs::File> {
        let p = run.join(name);
        let new = !p.exists();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p)?;
        if new {
            writeln!(f, "{header}")?;
        }
        Ok(f)
    };
    let mut log = open_log("log.tsv", "step\ttotal\tmel_l1\tdur\tpitch\tenergy\tf0_l1\tvuv_bce\tgrad_norm\tlr\tms")?;
    let mut elog = open_log("eval.tsv", "step\ttotal\tmel_l1\tdur\tpitch\tenergy\tf0_l1\tvuv_bce\twall_s")?;
    let read_params = |prog: &Program| -> Vec<Vec<f32>> { k_p.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect() };
    let save = |prog: &Program, step: usize, path: &Path| -> Result<()> {
        let params = read_params(prog);
        let m: Vec<Vec<f32>> = k_m.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect();
        let v: Vec<Vec<f32>> = k_v.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect();
        ckpt::save_specs(path, &specs, &Checkpoint { params, adam: Some((m, v)), step })
    };
    let prep = |rng: &mut Rng, crop: bool, us: &[&Utt], pool: &[Utt], pool_spk: &HashMap<String, Vec<usize>>, fe: &mut Frontend, fe_ref: &mut Frontend| {
        prep_crops(&dev, &tc, d, &nm, rng, crop, us, pool, pool_spk, fe, fe_ref)
    };
    let do_eval = |prog: &Program, fprog: &mut Program, step: usize, elog: &mut std::fs::File, fe: &mut Frontend, fe_ref: &mut Frontend| -> Result<()> {
        if held.len() < batch {
            return Ok(());
        }
        let params = read_params(prog);
        for (k, p) in fk_p.iter().zip(&params) {
            fprog.write_input(&dev, *k, p);
        }
        let mut rng = Rng::new(77);
        let mut acc = [0.0f64; N_LOSS];
        let mut k = 0.0;
        for c in held.chunks(batch).filter(|c| c.len() == batch) {
            let us: Vec<&Utt> = c.iter().collect();
            let (hb, mel, refmel) = prep(&mut rng, false, &us, &held, &held_spk, fe, fe_ref);
            for (name, v) in hb.refs() {
                fprog.write_input(&dev, fmap[name], v);
            }
            fprog.write_input(&dev, fmap["mel"], &mel);
            fprog.write_input(&dev, fmap["refmel"], &refmel);
            fprog.run(&dev);
            for (i, x) in acc.iter_mut().enumerate() {
                *x += fprog.read(&dev, i)[0] as f64;
            }
            k += 1.0;
        }
        let s: Vec<String> = acc.iter().map(|x| format!("{:.5}", x / k)).collect();
        writeln!(elog, "{step}\t{}\t{:.0}", s.join("\t"), t0.elapsed().as_secs_f64())?;
        elog.flush()?;
        eprintln!("eval step {step:>6}  held-out total {}  mel L1 {}  dur {}  pitch {}  energy {}  f0 {}  vuv {}", s[0], s[1], s[2], s[3], s[4], s[5], s[6]);
        Ok(())
    };
    if ck.step == 0 {
        do_eval(&prog, &mut fprog, 0, &mut elog, &mut fe, &mut fe_ref)?;
    }
    let start = ck.step;
    let mut step = ck.step;
    drop(ck);
    let mut rng = Rng::new(1000 + step as u64);
    let stop = run.join("STOP");
    let mut acc = [0.0f64; N_LOSS + 2];
    let mut acc_n = 0usize;
    let mut skipped = 0;
    let mut times = Vec::new();
    while step < start + steps {
        let ts = Instant::now();
        let us: Vec<&Utt> = (0..batch).map(|_| &data[rng.below(data.len())]).collect();
        let (hb, mel, refmel) = prep(&mut rng, true, &us, &data, &spk, &mut fe, &mut fe_ref);
        let lr = lr_at(step, total, peak, warmup);
        let t = (step + 1) as i32;
        let hyper = [lr as f32, (1.0 - 0.9f64.powi(t)) as f32, (1.0 - 0.98f64.powi(t)) as f32, 0.0, clip as f32, 0.0];
        for (name, v) in hb.refs() {
            prog.write_input(&dev, dmap[name], v);
        }
        prog.write_input(&dev, dmap["mel"], &mel);
        prog.write_input(&dev, dmap["refmel"], &refmel);
        prog.write_input(&dev, k_hyper, &hyper);
        prog.run(&dev);
        let gn = prog.read(&dev, 0)[0];
        let ls: Vec<f32> = (0..N_LOSS).map(|i| prog.read(&dev, 1 + i)[0]).collect();
        if !(gn.is_finite() && ls[0].is_finite()) {
            skipped += 1;
            eprintln!("step {}: non-finite (loss {}, |g| {gn}), skipped", step + 1, ls[0]);
            if skipped > 20 {
                return Err(anyhow!("too many non-finite steps"));
            }
            continue;
        }
        for i in 0..np {
            prog.swap_input_output(k_p[i], o_p + i);
            prog.swap_input_output(k_m[i], o_m + i);
            prog.swap_input_output(k_v[i], o_v + i);
        }
        step += 1;
        let ms = ts.elapsed().as_secs_f64() * 1e3;
        times.push(ms);
        for (i, x) in ls.iter().enumerate() {
            acc[i] += *x as f64;
        }
        acc[N_LOSS] += gn as f64;
        acc[N_LOSS + 1] += ms;
        acc_n += 1;
        let lsj: Vec<String> = ls.iter().map(|x| format!("{x:.5}")).collect();
        writeln!(log, "{step}\t{}\t{gn:.4}\t{lr:.3e}\t{ms:.0}", lsj.join("\t"))?;
        if step % log_every == 0 {
            let k = acc_n as f64;
            eprintln!(
                "step {step:>6}  total {:.4}  mel {:.4}  dur {:.4}  pitch {:.4}  energy {:.4}  f0 {:.4}  vuv {:.4}  |g| {:.3}  lr {lr:.2e}  {:.0} ms/step",
                acc[0] / k,
                acc[1] / k,
                acc[2] / k,
                acc[3] / k,
                acc[4] / k,
                acc[5] / k,
                acc[6] / k,
                acc[7] / k,
                acc[8] / k
            );
            acc = [0.0; N_LOSS + 2];
            acc_n = 0;
            log.flush()?;
        }
        let stopping = stop.exists();
        if step % ckpt_every == 0 || stopping {
            save(&prog, step, &ck_path)?;
        }
        if keep_every > 0 && step % keep_every == 0 {
            save(&prog, step, &run.join(format!("step{step:07}.safetensors")))?;
        }
        if eval_every > 0 && step % eval_every == 0 {
            do_eval(&prog, &mut fprog, step, &mut elog, &mut fe, &mut fe_ref)?;
        }
        if stopping {
            std::fs::remove_file(&stop).ok();
            eprintln!("STOP: saved step {step}");
            return Ok(());
        }
    }
    save(&prog, step, &ck_path)?;
    if eval_every > 0 && step % eval_every != 0 {
        do_eval(&prog, &mut fprog, step, &mut elog, &mut fe, &mut fe_ref)?;
    }
    let mut s: Vec<f64> = times.iter().skip(2).copied().collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !s.is_empty() {
        eprintln!("median {:.0} ms/step (frontends + host batch + train step)", s[s.len() / 2]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// inference

/// Two-stage TTS inference at one shape: encoder (+ predictors), host
/// durations → alignment, decoder.
struct Tts {
    enc: Program,
    dec: Program,
    emap: HashMap<&'static str, usize>,
    dmap: HashMap<&'static str, usize>,
}

impl Tts {
    fn new(dev: &Device, tc: &TConfig, params: &[Vec<f32>], d: Dims) -> Self {
        Self::with(dev, tc, params, d, false)
    }

    /// `voice`: `params` end with a voice file's two vectors (`acoustic::voice_specs`).
    fn with(dev: &Device, tc: &TConfig, params: &[Vec<f32>], d: Dims, voice: bool) -> Self {
        let specs = if voice { acoustic::voice_specs(tc) } else { tc.params() };
        let eg = acoustic::enc_graph_with(tc, d, voice);
        let ei = acoustic::assemble(&eg.b, &specs, params, None, &[]);
        let enc = Program::new(dev, &eg.b.g, &[eg.enc.h, eg.enc.ld, eg.enc.pitch, eg.enc.energy, eg.enc.spk_dec, eg.enc.spk_enc], &refs(&ei));
        let dg = acoustic::dec_graph(tc, d);
        let di = acoustic::assemble(&dg.b, &specs, params, None, &[]);
        let dec = Program::new(dev, &dg.b.g, &[dg.mel, dg.f0, dg.hid], &refs(&di));
        Self { enc, dec, emap: input_map(&eg.b), dmap: input_map(&dg.b) }
    }

    /// Encoder pass: returns (h, ln(d+1), pitch, energy, spk_dec).
    fn encode(&mut self, dev: &Device, onehot: &[f32], tmask: &[f32], refmel: &[f32], rw: &[f32]) -> [Vec<f32>; 5] {
        self.enc.write_input(dev, self.emap["onehot"], onehot);
        self.enc.write_input(dev, self.emap["tmask"], tmask);
        self.enc.write_input(dev, self.emap["refmel"], refmel);
        self.enc.write_input(dev, self.emap["rw"], rw);
        self.enc.run(dev);
        [0, 1, 2, 3, 4].map(|i| self.enc.read(dev, i))
    }

    /// Encoder-side speaker vector `[d]` of the last `encode`.
    fn spk_enc(&self, dev: &Device) -> Vec<f32> {
        self.enc.read(dev, 5)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(&mut self, dev: &Device, h: &[f32], pitch: &[f32], energy: &[f32], tmask: &[f32], align: &[f32], fmask: &[f32], spk: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        for (k, v) in [("h", h), ("pitch", pitch), ("energy", energy), ("tmask", tmask), ("align", align), ("fmask", fmask), ("spk_dec", spk)] {
            self.dec.write_input(dev, self.dmap[k], v);
        }
        self.dec.run(dev);
        (self.dec.read(dev, 0), self.dec.read(dev, 1), self.dec.read(dev, 2))
    }
}

/// Predicted durations from ln(d+1) (rounded, ≥ 0; pads at the ends kept).
/// Add st semitones to normalized token pitch ((ln F0 − μ) / σ).
pub(crate) fn shift_pitch(pp: &mut [f32], st: f64, nm: &Norm) {
    if st != 0.0 {
        let d = (st * std::f64::consts::LN_2 / 12.0 / nm.f0_sd) as f32;
        pp.iter_mut().for_each(|v| *v += d);
    }
}

fn dur_of(ld: f32, scale: f64) -> u32 {
    (((ld as f64).exp() - 1.0) * scale).round().max(0.0) as u32
}

fn onehot_rows(tc: &TConfig, rows: &[&[u16]], n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut oh = vec![0.0f32; rows.len() * tc.vocab * n];
    let mut tm = vec![0.0f32; rows.len() * n];
    for (b, r) in rows.iter().enumerate() {
        for (i, &id) in r.iter().enumerate() {
            oh[(b * tc.vocab + id as usize) * n + i] = 1.0;
            tm[b * n + i] = 1.0;
        }
    }
    (oh, tm)
}

fn load_run(run: &Path, tc: &TConfig, ckpt_name: Option<String>) -> Result<(Vec<Vec<f32>>, Norm, usize)> {
    let p = run.join(ckpt_name.unwrap_or_else(|| "last.safetensors".into()));
    let ck = ckpt::load_specs(&p, &tc.params(), false)?;
    eprintln!("loaded {} (step {})", p.display(), ck.step);
    Ok((ck.params, Norm::load(&run.join("norm.txt"))?, ck.step))
}

/// O(T₁·T₂) DTW over frames with L1 cost / n_mels; returns the mean cost
/// along the path (normalized by path length).
fn dtw_l1(a: &[f32], ta: usize, b: &[f32], tb: usize, m: usize, sa: usize, sb: usize) -> f64 {
    // a[c*sa + t], b[c*sb + t]
    let cost = |i: usize, j: usize| (0..m).map(|c| (a[c * sa + i] - b[c * sb + j]).abs() as f64).sum::<f64>() / m as f64;
    let inf = f64::INFINITY;
    let mut d = vec![inf; (ta + 1) * (tb + 1)];
    let mut len = vec![0u32; (ta + 1) * (tb + 1)];
    d[0] = 0.0;
    let w = tb + 1;
    for i in 1..=ta {
        for j in 1..=tb {
            let c = cost(i - 1, j - 1);
            let (mut best, mut bl) = (d[(i - 1) * w + j - 1], len[(i - 1) * w + j - 1]);
            if d[(i - 1) * w + j] < best {
                best = d[(i - 1) * w + j];
                bl = len[(i - 1) * w + j];
            }
            if d[i * w + j - 1] < best {
                best = d[i * w + j - 1];
                bl = len[i * w + j - 1];
            }
            d[i * w + j] = best + c;
            len[i * w + j] = bl + 1;
        }
    }
    d[ta * w + tb] / len[ta * w + tb] as f64
}

fn write_npy(path: &Path, shape: &[usize], v: &[f32]) -> Result<()> {
    let sh = shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ");
    let mut h = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({sh}{}), }}", if shape.len() == 1 { "," } else { "" });
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend((h.len() as u16).to_le_bytes());
    b.extend(h.as_bytes());
    for x in v {
        b.extend(x.to_le_bytes());
    }
    std::fs::write(path, b)?;
    Ok(())
}

fn eval(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let run = PathBuf::from(a.need("--run")?);
    let (params, nm, step) = load_run(&run, &tc, a.opt("--ckpt"))?;
    let root = PathBuf::from(a.need("--root")?);
    let durs = read_durs(&a.need("--durs")?)?;
    let n_utts: usize = a.get("--n-utts", 64);
    let batch: usize = a.get("--batch", 16);
    let (nmax, tmax): (usize, usize) = (a.get("--n", 192), a.get("--t", 1024));
    let held = load(&a.need("--held")?, &durs, &root, &|n, t| n <= nmax && t <= tmax * 3 / 4 && t >= 16)?;
    let stride = (held.len() / n_utts).max(1);
    let held: Vec<Utt> = held.into_iter().step_by(stride).take(n_utts).collect();
    let hspk = by_speaker(&held);
    eprintln!("eval on {} held-out utts ({} speakers), step {step}", held.len(), hspk.len());
    let dev = Device::new();
    let d = Dims { batch, n: nmax, t: tmax, r: REF_T };
    let mut tts = Tts::new(&dev, &tc, &params, d);
    let mut fe = Frontend::new(&dev, batch, tmax);
    let mut fe_ref = Frontend::new(&dev, batch, REF_T);
    let mut rng = Rng::new(77);
    let out_dir = a.opt("--out-dir").map(PathBuf::from);
    if let Some(o) = &out_dir {
        std::fs::create_dir_all(o)?;
    }
    let m = tc.n_mels;
    let (mut tf_l1, mut tfp_l1, mut pd_l1, mut nfr, mut ld_mae, mut ntok, mut len_ratio) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Vec::new());
    let mut nutt = 0.0;
    for c in held.chunks(batch) {
        let mut us: Vec<&Utt> = c.iter().collect();
        let real = us.len();
        while us.len() < batch {
            us.push(us[0]);
        }
        let clips: Vec<Vec<f32>> = us.iter().map(|u| u.audio_f32(0, u.frames())).collect();
        let mel = fe.run(&dev, &clips);
        let mut rclips = Vec::new();
        let mut rlens = Vec::new();
        for u in &us {
            let (r, off, len) = pick_ref(&mut rng, u, &held, &hspk);
            rclips.push(r.audio_f32(off, len));
            rlens.push(len);
        }
        let refmel = fe_ref.run(&dev, &rclips);
        let hb = host_batch(&tc, d, &us, &nm, &rlens, &vec![0; us.len()]);
        let [h, ld, pp, ep, spk] = tts.encode(&dev, hb.get("onehot"), hb.get("tmask"), &refmel, hb.get("rw"));
        // (1) teacher-forced: GT durations, GT pitch/energy
        let (m_tf, _, _) = tts.decode(&dev, &h, hb.get("p_tok"), hb.get("e_tok"), hb.get("tmask"), hb.get("align"), hb.get("fmask"), &spk);
        // (2) GT durations, predicted pitch/energy
        let (m_tfp, _, _) = tts.decode(&dev, &h, &pp, &ep, hb.get("tmask"), hb.get("align"), hb.get("fmask"), &spk);
        // (3) predicted durations + pitch/energy
        let pdurs: Vec<Vec<u32>> = us.iter().enumerate().map(|(b, u)| (0..u.toks.len()).map(|i| dur_of(ld[b * nmax + i], 1.0)).collect()).collect();
        let pdr: Vec<Vec<u32>> = pdurs.iter().map(|v| {
            let mut v = v.clone();
            let mut tot: u32 = 0;
            for x in v.iter_mut() {
                let room = tmax as u32 - tot;
                *x = (*x).min(room);
                tot += *x;
            }
            v
        }).collect();
        let pr: Vec<&[u32]> = pdr.iter().map(|v| v.as_slice()).collect();
        let pal = acoustic::align_matrix(&pr, nmax, tmax);
        let mut pfm = vec![0.0f32; batch * tmax];
        for (b, v) in pdr.iter().enumerate() {
            let s: u32 = v.iter().sum();
            for f in 0..s as usize {
                pfm[b * tmax + f] = 1.0;
            }
        }
        let (m_pd, _, _) = tts.decode(&dev, &h, &pp, &ep, hb.get("tmask"), &pal, &pfm, &spk);
        for (b, u) in us.iter().enumerate().take(real) {
            let fr = u.frames();
            let off = b * m * tmax;
            for f in 0..fr {
                for cc in 0..m {
                    let i = off + cc * tmax + f;
                    tf_l1 += (m_tf[i] - mel[i]).abs() as f64 / m as f64;
                    tfp_l1 += (m_tfp[i] - mel[i]).abs() as f64 / m as f64;
                }
            }
            nfr += fr as f64;
            let pf: u32 = pdr[b].iter().sum();
            len_ratio.push(pf as f64 / fr as f64);
            pd_l1 += dtw_l1(&m_pd[off..], pf as usize, &mel[off..], fr, m, tmax, tmax);
            nutt += 1.0;
            for i in 0..u.toks.len() {
                ld_mae += ((ld[b * nmax + i] as f64) - ((u.durs[i] as f64) + 1.0).ln()).abs();
                ntok += 1.0;
            }
            if let Some(o) = &out_dir {
                let grab = |x: &[f32], t: usize| -> Vec<f32> { (0..m).flat_map(|cc| x[off + cc * tmax..off + cc * tmax + t].to_vec()).collect() };
                write_npy(&o.join(format!("{}.gt.npy", u.id)), &[m, fr], &grab(&mel, fr))?;
                write_npy(&o.join(format!("{}.tf.npy", u.id)), &[m, fr], &grab(&m_tf, fr))?;
                write_npy(&o.join(format!("{}.pd.npy", u.id)), &[m, pf as usize], &grab(&m_pd, pf as usize))?;
            }
        }
    }
    len_ratio.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = len_ratio[len_ratio.len() / 2];
    let mean_lr = len_ratio.iter().sum::<f64>() / len_ratio.len() as f64;
    println!("| step | utts | mel L1 teacher-forced (GT dur, GT p/e) | GT dur, pred p/e | pred dur (DTW L1) | ln-dur MAE | length ratio pred/GT (mean, median) |");
    println!("|---|---|---|---|---|---|---|");
    println!("| {step} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.3}, {:.3} |", nutt, tf_l1 / nfr, tfp_l1 / nfr, pd_l1 / nutt, ld_mae / ntok, mean_lr, med);
    Ok(())
}

// ---------------------------------------------------------------------------
// synth

/// The V1.x vocoder (log-mel → audio) with an F0 source track, batch 1.
struct Vocoder {
    cfg: Config,
    prog: Program,
    k_mel: usize,
    k_f0: Option<usize>,
    /// §29.5: harmonic-source phase at each frame centre of the last run
    pub phis: Vec<f64>,
}

impl Vocoder {
    fn new(dev: &Device, ckpt_path: &Path, frames: usize) -> Result<Self> {
        let hid = Config::by_name("v1f0hid").unwrap();
        let cfg = if ckpt::numel(ckpt_path, "embed.w") == Some(hid.dim * hid.in_ch() * hid.embed_k) {
            hid
        } else if ckpt::has(ckpt_path, "hsrc.w") {
            Config::by_name("v1f0h").unwrap()
        } else if ckpt::has(ckpt_path, "src.w") {
            Config::by_name("v1f0").unwrap()
        } else {
            Config::v1()
        };
        let params = ckpt::load(ckpt_path, &cfg)?.params;
        let inf = model::infer_graph(&cfg, 1, frames);
        let inputs = inf.b.assemble(&params, None, &[]);
        let prog = Program::new(dev, &inf.b.g, &[inf.audio], &refs(&inputs));
        Ok(Self { k_mel: inf.b.input_index(inf.mel), k_f0: inf.f0.map(|t| inf.b.input_index(t)), prog, cfg, phis: vec![] })
    }
    /// hidden-input G (`cfg.hid`)?
    fn is_hid(&self) -> bool {
        self.cfg.hid > 0
    }
    /// G's main input from the decoder's hidden `[hid, stride]`: the first
    /// `valid` of `frames` frames get (hidden, 1, ln gain 0), the rest zeros.
    fn hid_input(&self, hid: &[f32], stride: usize, valid: usize, frames: usize) -> Vec<f32> {
        let hd = self.cfg.hid;
        let mut x = vec![0.0f32; (hd + 2) * frames];
        for c in 0..hd {
            x[c * frames..c * frames + valid].copy_from_slice(&hid[c * stride..c * stride + valid]);
        }
        x[hd * frames..hd * frames + valid].iter_mut().for_each(|v| *v = 1.0);
        x
    }
    fn run(&mut self, dev: &Device, mel: &[f32], f0hz: &[f32], frames: usize) -> Vec<f32> {
        self.run_phase(dev, mel, f0hz, frames, 0.0)
    }
    /// `run` with the harmonic source starting at phase `phi0` (frame 0's centre)
    fn run_phase(&mut self, dev: &Device, mel: &[f32], f0hz: &[f32], frames: usize, phi0: f64) -> Vec<f32> {
        if let Some(k) = self.k_f0 {
            let c = &self.cfg;
            let (src, phis) = dsp::source_rows(f0hz, 1, frames, c.sr, c.n_fft, c.hop, c.n_mels, c.hsrc, &[phi0]);
            self.phis = phis;
            self.prog.write_input(dev, k, &src);
        }
        self.prog.write_input(dev, self.k_mel, mel);
        self.prog.run(dev);
        self.prog.read(dev, 0)
    }
}

fn synth(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let run = PathBuf::from(a.need("--run")?);
    let (mut params, nm, _) = load_run(&run, &tc, a.opt("--ckpt"))?;
    // §25: context path params appended after the base ones
    let ctx_on = match a.opt("--ctx") {
        Some(c) => {
            let cc = acoustic::CtxConfig::base();
            params.extend(ckpt::load_specs(Path::new(&c), &cc.params(&tc), false)?.params);
            true
        }
        None => false,
    };
    let ctx_m: usize = a.get("--m", 128);
    // per-line context (ctx mode lines: name, text, prev, role[, emph word indices])
    let mut ctx_jobs: HashMap<String, (String, String, String)> = HashMap::new();
    let voc_path = PathBuf::from(a.need("--voc")?);
    // --rate R speaks R× faster (durations / R, composes with --pace);
    // --pitch S shifts the predicted pitch by S semitones (token pitch, so the
    // decoder's mel and F0 and therefore the vocoder's source follow)
    // --voice: a voice file's vectors replace the reference encoder; its
    // pitch / rate / accent are defaults under the CLI flags
    let voice = a.opt("--voice").map(|p| voice::Voice::load(Path::new(&p))).transpose()?;
    if let Some(v) = &voice {
        if ctx_on {
            return Err(anyhow!("--voice with --ctx: not wired"));
        }
        v.merge(&tc, &mut params)?;
        params.extend([v.se.clone(), v.sd.clone()]);
        if let (None, Some(ac)) = (a.opt("--accent"), v.meta.get("accent")) {
            ribt::lilt::accent::set(ac).map_err(|e| anyhow!(e))?;
        }
    }
    let vnum = |k: &str| voice.as_ref().and_then(|v| v.num(k));
    let pace: f64 = a.get("--pace", 1.0) / a.opt("--rate").and_then(|v| v.parse().ok()).or(vnum("rate")).unwrap_or(1.0);
    let pitch_st: f64 = a.opt("--pitch").and_then(|v| v.parse().ok()).or(vnum("pitch")).unwrap_or(0.0);
    let pitch_range: f32 = a.get("--pitch-range", 1.0);
    let save_voice = a.opt("--save-voice").map(PathBuf::from);
    let mut saved = false;
    // jobs: (name, text, reference wavs); a lines file may give a per-line
    // reference as a third column, else every --ref is used
    let dflt_refs: Vec<String> = a.opt("--ref").unwrap_or_else(|| DEMO_REF.into()).split(',').map(String::from).collect();
    let jobs: Vec<(String, String, Vec<String>)> = match (a.opt("--lines"), a.opt("--text")) {
        (Some(f), _) => std::fs::read_to_string(f)?
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .filter_map(|l| {
                let c: Vec<&str> = l.split('\t').collect();
                if ctx_on {
                    let g = |i: usize| c.get(i).map_or("-", |v| v).to_string();
                    ctx_jobs.insert(c[0].to_string(), (g(2), g(3), g(4)));
                    return (c.len() >= 2).then(|| (c[0].to_string(), c[1].to_string(), dflt_refs.clone()));
                }
                (c.len() >= 2).then(|| (c[0].to_string(), c[1].to_string(), c.get(2).map_or_else(|| dflt_refs.clone(), |r| vec![r.to_string()])))
            })
            .collect(),
        (None, Some(t)) => vec![("out".into(), t, dflt_refs.clone())],
        _ => return Err(anyhow!("--text or --lines")),
    };
    let out_dir = PathBuf::from(a.opt("--out-dir").unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&out_dir)?;
    let dev = Device::new();
    // reference mels (REF_T frames after 0.3 s, or the whole clip). A `+`-joined
    // reference is one mean speaker vector: its clips' crops end to end in an
    // `rr = REF_T × parts` window, rw = 1/total over the valid frames (the two
    // k=3 ref convs see the clip seams; the pool is otherwise the frame mean).
    let parts_of = |r: &str| r.split('+').count();
    let kmax = jobs.iter().flat_map(|j| j.2.iter()).map(|r| parts_of(r)).max().unwrap_or(1);
    let rr = REF_T * kmax;
    let mut fe_ref = Frontend::new(&dev, 1, REF_T);
    let mut ref_cache: HashMap<String, (String, Vec<f32>, Vec<f32>)> = HashMap::new();
    let n_mels = tc.n_mels;
    let mut ref_of = |r: &str| -> Result<(String, Vec<f32>, Vec<f32>)> {
        if let Some(v) = ref_cache.get(r) {
            return Ok(v.clone());
        }
        let mut mel = vec![0.0f32; n_mels * rr];
        let mut pos = 0usize;
        for part in r.split('+') {
            let rp = match part.strip_prefix("~/") {
                Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
                None => PathBuf::from(part),
            };
            let (x, sr) = ribt::data::read_wav(&rp)?;
            if sr as usize != SR {
                return Err(anyhow!("{part}: {sr} Hz"));
            }
            let fr = x.len() / HOP;
            let off = if fr > REF_T + 30 { 30 } else { 0 };
            let len = (fr - off).min(REF_T);
            let clip: Vec<f32> = x[off * HOP..(off + len) * HOP].to_vec();
            let m = fe_ref.run(&dev, &[clip]);
            for c in 0..n_mels {
                mel[c * rr + pos..c * rr + pos + len].copy_from_slice(&m[c * REF_T..c * REF_T + len]);
            }
            pos += len;
        }
        let mut rw = vec![0.0f32; rr];
        for v in rw.iter_mut().take(pos) {
            *v = 1.0 / pos as f32;
        }
        let first = r.split('+').next().unwrap_or(r);
        let mut tag = Path::new(first).file_stem().unwrap().to_string_lossy().split('_').next().unwrap_or("spk").to_string();
        if parts_of(r) > 1 {
            tag.push_str("mean");
        }
        let v = (tag, mel, rw);
        ref_cache.insert(r.to_string(), v.clone());
        Ok(v)
    };
    let mut cache: HashMap<(usize, usize), Tts> = HashMap::new();
    let mut vcache: HashMap<usize, Vocoder> = HashMap::new();
    // rung 2: flow decoder on top of rung 1's μ / hidden
    let fc = ribt::flow::FConfig::by_name(&a.opt("--fcfg").unwrap_or_else(|| "base".into())).ok_or_else(|| anyhow!("fcfg"))?;
    let flow = match a.opt("--flow") {
        Some(f) => {
            let p = PathBuf::from(f);
            Some((ckpt::load_specs(&p, &fc.params(), false)?.params, flow_cmd::FNorm::load(&p)?))
        }
        None => None,
    };
    // a voice file's nfe / temp are defaults under the flags
    let nfe: usize = a.opt("--nfe").and_then(|v| v.parse().ok()).or(vnum("nfe").map(|v| v as usize)).unwrap_or(16);
    let shortcut = a.0.iter().any(|x| x == "--shortcut");
    let temp: f64 = a.opt("--temp").and_then(|v| v.parse().ok()).or(vnum("temp")).unwrap_or(1.0);
    let seed: u64 = a.get("--seed", 0);
    let chunk = a.opt("--chunk").and_then(|c| c.parse().ok()).map(|c: usize| (c, a.get("--look", 1usize)));
    let mut fcache: HashMap<usize, flow_cmd::Vel> = HashMap::new();
    // --check-voc MEL_G: with a hidden-input --voc, also vocode the decoder's mel with MEL_G and compare
    let check_voc = a.opt("--check-voc").map(PathBuf::from);
    if check_voc.is_some() && flow.is_some() {
        return Err(anyhow!("--check-voc compares against the decoder's mel: no --flow"));
    }
    let mut cvcache: HashMap<usize, Vocoder> = HashMap::new();
    let mut check_rows: Vec<(f64, f64)> = Vec::new();
    let mut meta = std::fs::File::create(out_dir.join("synth.tsv"))?;
    writeln!(meta, "name\tspeaker\ttext\tphonemes\ttokens\tframes\tseconds\tenc_ms\tdec_ms\tvoc_ms\tflow_ms\tnfe")?;
    let mut ccache: HashMap<usize, ctx::CtxEnc> = HashMap::new();
    for (name, text, rlist) in &jobs {
        let (clean, marked) = ctx::strip_emph(text);
        let text = &clean;
        let (ps, words) = lilt::phonemize(text);
        let oov: Vec<&str> = words.iter().filter(|w| w.prosody.oov).map(|w| w.text.as_str()).collect();
        if !oov.is_empty() {
            eprintln!("{name}: OOV (dropped by lilt): {oov:?}");
        }
        let mut toks: Vec<u16> = vec![0];
        toks.extend(lilt::vocab::ids(&ps).iter().map(|&v| v as u16));
        toks.push(0);
        let n = toks.len().div_ceil(32) * 32;
        for r in rlist {
            let (tag, refmel, rw) = match &voice {
                Some(v) => (v.name(), vec![0.0f32; n_mels * rr], vec![0.0f32; rr]),
                None => ref_of(r)?,
            };
            let (refmel, rw) = (&refmel, &rw);
            let von = voice.is_some();
            let tts = cache.entry((n, 0)).or_insert_with(|| Tts::with(&dev, &tc, &params, Dims { batch: 1, n, t: 64, r: rr }, von));
            let (oh, tm) = onehot_rows(&tc, &[&toks], n);
            let t_enc = Instant::now();
            let [h, ld, mut pp, ep, spk] = if ctx_on {
                let (prev, role, emph) = ctx_jobs.get(name).cloned().unwrap_or(("-".into(), "none".into(), "-".into()));
                let mut em: Vec<usize> = emph.split(',').filter_map(|x| x.parse().ok()).collect();
                em.extend(&marked);
                let pv = ctx::prev_tokens(&prev, ctx_m);
                let ri = ctx::role_index(&role);
                let gate = if ri == 0 && prev == "-" && em.is_empty() { 0.0 } else { 1.0 };
                let cdata = ctx::ctx_tensors(tc.vocab, n, ctx_m, &[(&toks, &pv, ri, &em, gate)]);
                let enc = ccache.entry(n).or_insert_with(|| ctx::CtxEnc::new(&dev, &tc, &params, n, ctx_m, rr));
                let mut d: Vec<(&str, &[f32])> = vec![("onehot", &oh), ("tmask", &tm), ("refmel", refmel), ("rw", rw)];
                d.extend(cdata.iter().map(|(k, v)| (*k, v.as_slice())));
                enc.encode(&dev, &d)
            } else {
                tts.encode(&dev, &oh, &tm, refmel, rw)
            };
            if let (Some(p), false) = (&save_voice, ctx_on || voice.is_some()) {
                if !saved {
                    saved = true;
                    let mut meta = voice::Voice::meta_from(a, &format!("zero-shot --ref {r}"));
                    meta.entry("name".into()).or_insert_with(|| tag.clone());
                    voice::Voice { se: cache[&(n, 0)].spk_enc(&dev), sd: spk.clone(), lora: Vec::new(), meta }.save(p)?;
                    eprintln!("wrote voice {}", p.display());
                }
            }
            shift_pitch(&mut pp, pitch_st, &nm);
            // --pitch-range R: token pitch spread around the utterance mean × R
            if pitch_range != 1.0 {
                let k = toks.len();
                let mu = pp[..k].iter().sum::<f32>() / k as f32;
                pp[..k].iter_mut().for_each(|v| *v = mu + (*v - mu) * pitch_range);
            }
            let enc_ms = t_enc.elapsed().as_secs_f64() * 1e3;
            let durs: Vec<u32> = (0..toks.len()).map(|i| dur_of(ld[i], pace)).collect();
            let frames: usize = durs.iter().sum::<u32>() as usize;
            let t = frames.div_ceil(64) * 64;
            let key = (n, t);
            if !cache.contains_key(&key) {
                cache.insert(key, Tts::with(&dev, &tc, &params, Dims { batch: 1, n, t, r: rr }, von));
            }
            let tts = cache.get_mut(&key).unwrap();
            let al = acoustic::align_matrix(&[&durs], n, t);
            let mut fm = vec![0.0f32; t];
            fm[..frames].iter_mut().for_each(|v| *v = 1.0);
            let t_dec = Instant::now();
            let (mut mel, f0, hid) = tts.decode(&dev, &h, &pp, &ep, &tm, &al, &fm, &spk);
            let dec_ms = t_dec.elapsed().as_secs_f64() * 1e3;
            let mut flow_ms = 0.0;
            if let Some((fp, fnm)) = &flow {
                let vel = fcache.entry(t).or_insert_with(|| flow_cmd::Vel::new(&dev, &fc, fp, ribt::flow::FDims { batch: 1, t }));
                let mun = fnm.norm(&mel, &fm, 1, t);
                let ab = ribt::flow::attn_bias(&[frames], t, chunk);
                let x0 = flow_cmd::noise(&mut Rng::new(seed), &fm, 1, tc.n_mels, t, temp);
                let t_fl = Instant::now();
                vel.set_cond(&dev, &mun, &hid, &fm, &ab);
                let (x1, _) = vel.sample(&dev, &x0, nfe, shortcut);
                flow_ms = t_fl.elapsed().as_secs_f64() * 1e3;
                mel = fnm.denorm(&x1);
            }
            // trim to the predicted length; F0 channel 0 = normalized ln F0, 1 = voicing logit
            let mel_c: Vec<f32> = (0..tc.n_mels).flat_map(|c| mel[c * t..c * t + frames].to_vec()).collect();
            let f0hz: Vec<f32> = (0..frames)
                .map(|f| if f0[t + f] > 0.0 { ((f0[f] as f64) * nm.f0_sd + nm.f0_mu).exp() as f32 } else { 0.0 })
                .collect();
            let voc = match vcache.entry(frames) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => e.insert(Vocoder::new(&dev, &voc_path, frames)?),
            };
            let t_voc = Instant::now();
            // hidden-decoder rung 1: a hidden-input G reads the decoder's last layer, not the mel
            let vin = if voc.is_hid() { voc.hid_input(&hid, t, frames, frames) } else { mel_c.clone() };
            let y = voc.run(&dev, &vin, &f0hz, frames);
            let voc_ms = t_voc.elapsed().as_secs_f64() * 1e3;
            if let (Some(cv), true) = (check_voc.as_ref(), voc.is_hid()) {
                // step-0 identity: the mel G on the decoder's own mel (no flow) must equal G'(hidden)
                let mv = cvcache.entry(frames).or_insert_with(|| Vocoder::new(&dev, cv, frames).unwrap());
                let ym = mv.run(&dev, &mel_c, &f0hz, frames);
                let (mut mx, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
                for (a, b) in y.iter().zip(&ym) {
                    let d = (*a as f64 - *b as f64).abs();
                    mx = mx.max(d);
                    num += d * d;
                    den += (*b as f64) * (*b as f64);
                }
                let peak = ym.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                eprintln!("hid-check {name}: max |G'(h) − G(mel)| {mx:.3e} (peak {peak:.3}), rel RMS {:.3e}", (num / den.max(1e-30)).sqrt());
                check_rows.push((mx, (num / den.max(1e-30)).sqrt()));
            }
            let stem = format!("{name}_{tag}");
            ribt::data::write_wav(&out_dir.join(format!("{stem}.wav")), &y, SR as u32)?;
            write_npy(&out_dir.join(format!("{stem}.mel.npy")), &[tc.n_mels, frames], &mel_c)?;
            write_npy(&out_dir.join(format!("{stem}.f0.npy")), &[frames], &f0hz)?;
            let js = |v: &[String]| v.join(" ");
            std::fs::write(
                out_dir.join(format!("{stem}.durs.txt")),
                format!("{}\n{}\n{}\n", js(&toks.iter().map(|x| x.to_string()).collect::<Vec<_>>()), js(&durs.iter().map(|x| x.to_string()).collect::<Vec<_>>()), js(&pp[..toks.len()].iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>())),
            )?;
            let secs = y.len() as f64 / SR as f64;
            writeln!(meta, "{name}\t{tag}\t{text}\t{ps}\t{}\t{frames}\t{secs:.2}\t{enc_ms:.1}\t{dec_ms:.1}\t{voc_ms:.1}\t{flow_ms:.1}\t{}", toks.len(), if flow.is_some() { nfe } else { 0 })?;
            eprintln!("{stem}: {} tokens → {frames} frames ({secs:.2} s); enc {enc_ms:.1} ms, dec {dec_ms:.1} ms, flow {flow_ms:.1} ms ({nfe} NFE), voc {voc_ms:.1} ms", toks.len());
        }
    }
    if !check_rows.is_empty() {
        let mx = check_rows.iter().fold(0.0f64, |m, r| m.max(r.0));
        let rr = check_rows.iter().fold(0.0f64, |m, r| m.max(r.1));
        println!("hid-check: {} utts, worst max abs diff {mx:.3e}, worst rel RMS {rr:.3e}", check_rows.len());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// checks

fn random_batch(tc: &TConfig, d: Dims, rng: &mut Rng) -> (HostBatch, Vec<f32>, Vec<f32>) {
    // synthetic utterances: random tokens/durations that fill 50–100% of N, T
    let mut utts = Vec::new();
    for _ in 0..d.batch {
        let n = (d.n / 2 + rng.below(d.n / 2)).max(3);
        let toks: Vec<u16> = (0..n).map(|_| rng.below(tc.vocab) as u16).collect();
        let mut durs: Vec<u32> = (0..n).map(|_| 1 + rng.below(3) as u32).collect();
        while durs.iter().sum::<u32>() as usize > d.t {
            let i = rng.below(n);
            if durs[i] > 0 {
                durs[i] -= 1;
            }
        }
        let fr = durs.iter().sum::<u32>() as usize;
        let f0: Vec<f32> = (0..fr).map(|_| if rng.uniform() < 0.6 { 100.0 + 100.0 * rng.uniform() as f32 } else { 0.0 }).collect();
        let en: Vec<f32> = (0..fr).map(|_| -6.0 + rng.normal() as f32).collect();
        utts.push(Utt { id: String::new(), spk: String::new(), toks, durs, audio: vec![0; fr * HOP], f0, en });
    }
    let nm = Norm { f0_mu: 5.0, f0_sd: 0.3, en_mu: -6.0, en_sd: 1.0 };
    let us: Vec<&Utt> = utts.iter().collect();
    let rl: Vec<usize> = (0..d.batch).map(|_| d.r / 2 + rng.below(d.r / 2 + 1)).collect();
    let hb = host_batch(tc, d, &us, &nm, &rl, &vec![0; us.len()]);
    let mel: Vec<f32> = (0..d.batch * tc.n_mels * d.t).map(|_| (-5.0 + 2.0 * rng.normal()) as f32).collect();
    let refmel: Vec<f32> = (0..d.batch * tc.n_mels * d.r).map(|_| (-5.0 + 2.0 * rng.normal()) as f32).collect();
    (hb, mel, refmel)
}

fn check(a: &Args) -> Result<()> {
    let name = a.opt("--cfg").unwrap_or_else(|| "check".into());
    let mut tc = TConfig::by_name(&name).ok_or_else(|| anyhow!("cfg"))?;
    tc.enc_layers = a.get("--enc-layers", tc.enc_layers);
    let specs = tc.params();
    let d = Dims { batch: a.get("--batch", 3), n: a.get("--n", 24), t: a.get("--t", 64), r: a.get("--r", 16) };
    let mut rng = Rng::new(4);
    let params: Vec<Vec<f32>> = acoustic::init_params(&tc, 1).into_iter().map(|p| p.iter().map(|v| v + (0.02 * rng.normal()) as f32).collect()).collect();
    let (hb, mel, refmel) = random_batch(&tc, d, &mut rng);
    let mut data = hb.refs();
    data.push(("mel", &mel));
    data.push(("refmel", &refmel));
    if a.0.iter().any(|x| x == "--fwd") {
        // forward-only loss graph (the held-out eval program): bisects train-graph faults
        let fg = acoustic::tts_graph(&tc, d);
        let inp = acoustic::assemble(&fg.b, &specs, &params, None, &data);
        let i64s: Vec<Vec<f64>> = inp.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
        let want = fg.b.g.eval(&i64s, &fg.losses);
        let dev = Device::new();
        let mut prog = Program::new(&dev, &fg.b.g, &fg.losses, &refs(&inp));
        prog.run(&dev);
        for i in 0..fg.losses.len() {
            eprintln!("  fwd loss {i}: gpu {} f64 {}", prog.read(&dev, i)[0], want[i][0]);
        }
        return Ok(());
    }
    if a.0.iter().any(|x| x == "--enc") {
        // encoder stage alone: spk_dec, h, ld, pitch, energy
        let eg = acoustic::enc_graph(&tc, d);
        let outs = [eg.enc.spk_dec, eg.enc.h, eg.enc.ld, eg.enc.pitch, eg.enc.energy];
        let inp = acoustic::assemble(&eg.b, &specs, &params, None, &data);
        let i64s: Vec<Vec<f64>> = inp.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
        let want = eg.b.g.eval(&i64s, &outs);
        let dev = Device::new();
        let mut prog = Program::new(&dev, &eg.b.g, &outs, &refs(&inp));
        prog.run(&dev);
        for (i, n) in ["spk_dec", "h", "ld", "pitch", "energy"].iter().enumerate() {
            let g = prog.read(&dev, i);
            let sc = want[i].iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
            let e = g.iter().zip(&want[i]).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0f64, f64::max) / sc;
            eprintln!("  enc {n}: rel {e:.2e} (n {})", g.len());
        }
        return Ok(());
    }
    let tt = acoustic::train_graph(&tc, d);
    let mut inp = acoustic::assemble(&tt.tg.b, &specs, &params, None, &data);
    inp[tt.tg.b.input_index(tt.hyper)] = vec![1e-3, 0.1, 0.02, 0.0, 1.0, 0.0];
    let i64s: Vec<Vec<f64>> = inp.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
    if a.0.iter().any(|x| x == "--grads") {
        // raw parameter gradients (before the clip / Adam): finds the NaN source
        let want = tt.tg.b.g.eval(&i64s, &tt.grads);
        let dev = Device::new();
        let mut prog = Program::new(&dev, &tt.tg.b.g, &tt.grads, &refs(&inp));
        prog.run(&dev);
        for (i, sp) in specs.iter().enumerate() {
            let g = prog.read(&dev, i);
            let nan = g.iter().filter(|v| !v.is_finite()).count();
            let sc = want[i].iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
            let e = g.iter().zip(&want[i]).filter(|(g, _)| g.is_finite()).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0f64, f64::max) / sc;
            if nan > 0 || e > 1e-3 {
                eprintln!("  grad {}: rel {e:.2e}, non-finite {nan}/{}", sp.name, g.len());
            }
        }
        return Ok(());
    }
    let t = Instant::now();
    let want = tt.tg.b.g.eval(&i64s, &tt.outputs);
    eprintln!("interpreter {:.1} s", t.elapsed().as_secs_f64());
    let dev = Device::new();
    let mut prog = Program::new(&dev, &tt.tg.b.g, &tt.outputs, &refs(&inp));
    prog.run(&dev);
    #[cfg(feature = "cuda")]
    if a.0.iter().any(|x| x == "--bisect") {
        prog.bisect(&dev, &i64s, a.get("--tol", 1e-4), Path::new("bisect_kernel.cu"));
        return Ok(());
    }
    let rel = |got: &[f32], want: &[f64]| -> f64 {
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
        got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((*g as f64 - w).abs())) / scale
    };
    let np = specs.len();
    let mut loss_err: f64 = 0.0;
    let verbose = a.0.iter().any(|x| x == "-v");
    for i in 0..=N_LOSS {
        let (g, e) = (prog.read(&dev, i), rel(&prog.read(&dev, i), &want[i]));
        if verbose {
            eprintln!("  out {i}: gpu {} f64 {} rel {e:.2e}", g[0], want[i][0]);
        }
        loss_err = loss_err.max(if e.is_nan() { f64::INFINITY } else { e });
    }
    if verbose {
        for (k, o) in (1 + N_LOSS + 3 * np..tt.outputs.len()).enumerate() {
            eprintln!("  grad-norm partial {k} (params {}..): gpu {} f64 {}", 8 * k, prog.read(&dev, o)[0], want[o][0]);
        }
    }
    let mut m_err: f64 = 0.0;
    let mut worst = String::new();
    for i in 0..np {
        let o = 1 + N_LOSS + np + i;
        let e = rel(&prog.read(&dev, o), &want[o]);
        let e = if e.is_nan() { f64::INFINITY } else { e };
        if verbose && e > 1e-3 {
            eprintln!("  grad {}: rel {e:.2e}", specs[i].name);
        }
        if e > m_err || (m_err == 0.0 && worst.is_empty() && e > 0.0) {
            m_err = e;
            worst = specs[i].name.clone();
        }
    }
    eprintln!(
        "{} vs f64 interpreter, cfg {name} ({d:?}, {} params): losses + |g| worst rel {loss_err:.2e} (total {} vs {}); m' (grads) worst rel {m_err:.2e} ({worst})",
        ribt::runtime::BACKEND,
        tc.n_params(),
        prog.read(&dev, 1)[0],
        want[1][0]
    );
    if loss_err > 1e-4 || m_err > 1e-3 {
        return Err(anyhow!("GPU/interpreter mismatch"));
    }
    Ok(())
}

fn bench(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let specs = tc.params();
    let d = Dims { batch: a.get("--batch", 16), n: a.get("--n", 192), t: a.get("--t", 1024), r: REF_T };
    let mut rng = Rng::new(4);
    let (hb, mel, refmel) = random_batch(&tc, d, &mut rng);
    let mut data = hb.refs();
    data.push(("mel", &mel));
    data.push(("refmel", &refmel));
    let tt = acoustic::train_graph(&tc, d);
    let mut inp = acoustic::assemble(&tt.tg.b, &specs, &acoustic::init_params(&tc, 0), None, &data);
    inp[tt.tg.b.input_index(tt.hyper)] = vec![1e-4, 0.1, 0.02, 0.0, 1.0, 0.0];
    let dev = Device::new();
    let t = Instant::now();
    let mut prog = Program::new(&dev, &tt.tg.b.g, &tt.outputs, &refs(&inp));
    eprintln!("{} params, {d:?}: {} kernels, compile {:.1} s", tc.n_params(), prog.kernels(), t.elapsed().as_secs_f64());
    let ms = prog.time(&dev, a.get("--iters", 10));
    prog.run(&dev);
    eprintln!("train step {ms:.1} ms (kernels only); losses {:?}", (0..=N_LOSS).map(|i| prog.read(&dev, i)[0]).collect::<Vec<_>>());
    #[cfg(feature = "metal")]
    if a.0.iter().any(|x| x == "--profile") {
        let ex = &prog.exe.0;
        let nk = ex.program.kernels.len();
        let mut rows: Vec<(f64, usize)> = (0..nk).map(|i| (ex.time_kernels(&dev.0, i..i + 1, 3), i)).collect();
        let sum: f64 = rows.iter().map(|r| r.0).sum();
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        eprintln!("per-kernel sum {sum:.1} ms; top 25:");
        for (ms, i) in rows.iter().take(25) {
            let k = &ex.program.kernels[*i];
            let sh: Vec<usize> = prog.graph.shape(k.node).iter().map(|d| d.n()).collect();
            let red = "";
            eprintln!("  {ms:8.2} ms  k{i:<5} node {:?} shape {sh:?} reduce {red} groups {:?} thr {} frag {} reads {}", k.node, k.groups, k.threads, k.fragment, k.reads.len());
        }
    }
    Ok(())
}
