//! `ribt` CLI. Every GPU subcommand should run under
//! `~/.cache/gpu-lock.sh` on shared machines.
//!
//!   ribt train  --data DIR[,DIR] --run DIR [--heldout DIR[,DIR]] [--cfg v1] [--steps N] [--total N]
//!                     [--ckpt-every 1000] [--eval-every 0] [--keep-every 10000]   (touch RUN/STOP to save+exit)
//!                     [--batch 8] [--frames 64] [--lr 5e-4]
//!   ribt gan    --data DIR[,DIR] --run DIR --init V1.safetensors [--cfg v1f0] [--heldout DIRS [--held-by-clip]]
//!                     [--lr 2e-4] [--lr-d 2e-4] [--w-mel 45] [--w-mrstft 1] [--w-adv 1] [--w-fm 1] [--w-mrd 0.1]
//!                     V1.1: MPD + MRD adversarial fine-tune, alternating G / D programs
//!   ribt eval   --run DIR (--data DIR | --heldout DIRS) [--out DIR]   held-out losses + resynthesis
//!   ribt infer  --ckpt F [--cfg v1] IN.wav|IN.npy OUT.wav
//!   ribt mel    IN.wav OUT.npy [--cfg v1]             log-mel [n_mels, T]
//!   ribt bench  [--ckpt F] [--cfg v1] [--seconds 1]   inference RTF, batch 1
//!   ribt gpucheck                                     tiny train step, GPU vs f64
//!   ribt lilt say "text"                          text -> Kokoro phonemes (lilt, CPU)
//!   ribt lilt parity [--golden data/lilt/golden_eval.tsv] [--split test,edge] [--show 12] [--stream 500]
//!   ribt lilt train-tagger [--golden data/lilt/golden.tsv] [--out data/lilt/tagger.bin] [--iters 8]
//!   ribt lilt bench [--golden F]                  words/s + streaming latency
//!   ribt stepbench [--cfg v1] [--batch 8] [--frames 64] [--iters 50]   train-step ms (no data)

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use ribt::ckpt::{self, Checkpoint};
use ribt::data::{self, Dataset};
use ribt::dsp;
use ribt::model::{self, Config, Rng};
use ribt::runtime::{Device, Program};

struct Args(Vec<String>);

impl Args {
    fn opt(&self, k: &str) -> Option<String> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1).cloned())
    }
    fn get<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        self.opt(k).and_then(|v| v.parse().ok()).unwrap_or(d)
    }
    fn flag(&self, k: &str) -> bool {
        self.0.iter().any(|a| a == k)
    }
    fn positional(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut i = 1;
        while i < self.0.len() {
            if self.0[i].starts_with("--") {
                i += if self.0.get(i + 1).is_some_and(|n| !n.starts_with("--")) && !matches!(self.0[i].as_str(), "--f16") { 2 } else { 1 };
            } else {
                out.push(self.0[i].clone());
                i += 1;
            }
        }
        out
    }
    fn cfg(&self) -> Result<Config> {
        // a checkpoint with F0-source weights implies the f0 config
        let ck_f0 = self.opt("--ckpt").is_some_and(|p| ckpt::has(Path::new(&p), "src.w"));
        let name = self.opt("--cfg").unwrap_or_else(|| if ck_f0 { "v1f0".into() } else { "v1".into() });
        Config::by_name(&name).ok_or_else(|| anyhow!("unknown cfg {name}"))
    }
}

fn main() -> Result<()> {
    let a = Args(std::env::args().skip(1).collect());
    match a.0.first().map(String::as_str) {
        Some("train") => train(&a),
        Some("gan") => train_gan(&a),
        Some("dbench") => dbench(&a),
        Some("gancheck") => gancheck(&a),
        Some("gdiag") => gdiag(&a),
        Some("eval") => eval(&a),
        Some("infer") => infer(&a),
        Some("rinfer") => rinfer(&a),
        Some("phys-labels") => phys_labels(&a),
        Some("mel") => mel(&a),
        Some("bench") => bench(&a),
        Some("gpucheck") => gpucheck(&a),
        Some("stepbench") => stepbench(&a),
        Some("lilt") => lilt_cmd(&a),
        _ => {
            eprintln!("usage: ribt train|eval|infer|mel|bench|gpucheck (see src/main.rs)");
            Ok(())
        }
    }
}

fn device() -> Device {
    let d = Device::new();
    eprintln!("backend {} ({})", ribt::runtime::BACKEND, d.name());
    d
}

/// F0 source input `[B, n_mels + 2, frames]` for padded crops `xp`
/// (YIN + harmonic comb per row, rows in parallel).
fn src_of(cfg: &Config, xp: &[f32], batch: usize, frames: usize) -> Vec<f32> {
    let row = xp.len() / batch;
    let parts: Vec<Vec<f32>> = std::thread::scope(|s| {
        let hs: Vec<_> = xp
            .chunks(row)
            .map(|r| {
                s.spawn(move || {
                    let f0 = dsp::yin_f0(r, cfg.sr, cfg.n_fft, cfg.hop, frames);
                    src_rows(cfg, &f0, 1, frames)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    parts.concat()
}

/// The vocoder's "src" rows from per-frame F0 (Hz): the frame-rate source,
/// plus the §29.5 harmonic source when `cfg.hsrc > 0` (phase 0 at frame 0).
fn src_rows(cfg: &Config, f0: &[f32], batch: usize, frames: usize) -> Vec<f32> {
    dsp::source_rows(f0, batch, frames, cfg.sr, cfg.n_fft, cfg.hop, cfg.n_mels, cfg.hsrc, &[]).0
}

fn refs(v: &[Vec<f32>]) -> Vec<&[f32]> {
    v.iter().map(|x| x.as_slice()).collect()
}

// ---------------------------------------------------------------------------
// train

fn lr_at(step: usize, total: usize, peak: f64, warmup: usize) -> f64 {
    if step < warmup {
        return peak * (step + 1) as f64 / warmup as f64;
    }
    let p = ((step - warmup) as f64 / (total.saturating_sub(warmup)).max(1) as f64).min(1.0);
    peak * (0.05 + 0.95 * 0.5 * (1.0 + (std::f64::consts::PI * p).cos()))
}

fn dirs(s: &str) -> Vec<PathBuf> {
    s.split(',').filter(|d| !d.is_empty()).map(PathBuf::from).collect()
}

/// Training audio: the smoke `Dataset` (f32) or a corpus held as i16.
enum TrainSet {
    Small(Dataset),
    Corpus(data::Corpus),
}

impl TrainSet {
    fn batch(&self, rng: &mut Rng, batch: usize, n: usize, pad: usize) -> (Vec<f32>, Vec<f32>) {
        match self {
            TrainSet::Small(d) => d.batch(rng, batch, n, pad),
            TrainSet::Corpus(c) => c.batch(rng, batch, n, pad),
        }
    }
}

/// Train / held-out split. With `--heldout DIRS` (LibriTTS-R test-clean):
/// a fixed, speaker-round-robin held-out set from those dirs, and every
/// wav under `--data DIRS` from other speakers for training. Without it:
/// the old smoke split (every 20th clip of `--data`).
fn load_split(a: &Args, cfg: &Config) -> Result<(TrainSet, Dataset)> {
    let data_dirs = dirs(&a.opt("--data").ok_or_else(|| anyhow!("--data"))?);
    let sr = cfg.sr as u32;
    match a.opt("--heldout") {
        Some(h) => {
            let held = Dataset::heldout(&dirs(&h), sr, a.get("--heldout-n", 48))?;
            // --held-by-clip (§20 speaker fine-tune): held-out clips are other
            // files of the training speaker, so exclude nothing by speaker
            let exclude = if a.flag("--held-by-clip") { Default::default() } else { held.clips.iter().map(|(p, _)| data::speaker_of(p)).collect() };
            let t = Instant::now();
            let c = data::Corpus::load(&data_dirs, sr, &exclude, cfg.hop * 32)?;
            eprintln!(
                "data: {} train clips, {} speakers ({:.2} h) loaded in {:.0} s; held-out {} clips from {} speakers ({:.1} min)",
                c.clips.len(),
                c.speakers,
                c.seconds(cfg.sr) / 3600.0,
                t.elapsed().as_secs_f64(),
                held.clips.len(),
                held.clips.iter().map(|(p, _)| data::speaker_of(p)).collect::<std::collections::HashSet<_>>().len(),
                held.seconds(cfg.sr) / 60.0
            );
            if c.clips.is_empty() {
                return Err(anyhow!("no training wavs"));
            }
            Ok((TrainSet::Corpus(c), held))
        }
        None => {
            let (ds, held) = Dataset::load(&data_dirs[0], sr, 20)?;
            eprintln!(
                "data: {} train clips ({:.1} min), {} held out ({:.1} min)",
                ds.clips.len(),
                ds.seconds(cfg.sr) / 60.0,
                held.clips.len(),
                held.seconds(cfg.sr) / 60.0
            );
            Ok((TrainSet::Small(ds), held))
        }
    }
}

/// Held-out losses (the training graph's loss terms) on fixed crops of
/// `frames` frames, batch 1, every whole crop of every held-out clip.
struct HeldoutEval {
    prog: Program,
    k_f0: Option<usize>,
    k_pad: usize,
    k_audio: usize,
    k_params: Vec<usize>,
    n: usize,
    n_res: usize,
}

struct HeldoutLosses {
    crops: usize,
    seconds: f64,
    /// total, MR-STFT, mel L1, then SC per resolution, then log-mag per resolution
    avg: Vec<f64>,
}

impl HeldoutEval {
    fn new(dev: &Device, cfg: &Config, params: &[Vec<f32>], frames: usize) -> Self {
        let lg = model::loss_graph(cfg, 1, frames);
        let inputs = lg.b.assemble(params, None, &[]);
        let l = &lg.losses;
        let mut outs = vec![l.total, l.mrstft, l.mel_l1];
        outs.extend(&l.sc);
        outs.extend(&l.logmag);
        let prog = Program::new(dev, &lg.b.g, &outs, &refs(&inputs));
        Self {
            k_f0: lg.f0.map(|t| lg.b.input_index(t)),
            k_pad: lg.b.input_index(lg.audio_pad),
            k_audio: lg.b.input_index(lg.audio),
            k_params: lg.params.iter().map(|&t| lg.b.input_index(t)).collect(),
            prog,
            n: frames * cfg.hop,
            n_res: cfg.loss_res.len(),
        }
    }

    fn run(&mut self, dev: &Device, cfg: &Config, params: &[Vec<f32>], held: &Dataset) -> HeldoutLosses {
        for (&k, p) in self.k_params.iter().zip(params) {
            self.prog.write_input(dev, k, p);
        }
        let n = self.n;
        let mut sums = vec![0.0f64; 3 + 2 * self.n_res];
        let mut crops = 0;
        for (_, clip) in &held.clips {
            let mut s = 0;
            while s + n <= clip.len() {
                let seg = &clip[s..s + n];
                let xp = dsp::reflect_pad(seg, cfg.mel_pad());
                if let Some(k) = self.k_f0 {
                    self.prog.write_input(dev, k, &src_of(cfg, &xp, 1, n / cfg.hop));
                }
                self.prog.write_input(dev, self.k_pad, &xp);
                self.prog.write_input(dev, self.k_audio, seg);
                self.prog.run(dev);
                for (i, v) in sums.iter_mut().enumerate() {
                    *v += self.prog.read(dev, i)[0] as f64;
                }
                crops += 1;
                s += n;
            }
        }
        HeldoutLosses { crops, seconds: n as f64 / cfg.sr as f64, avg: sums.iter().map(|v| v / crops.max(1) as f64).collect() }
    }
}

impl HeldoutLosses {
    fn line(&self, n_res: usize) -> String {
        let a = &self.avg;
        format!(
            "held-out ({} crops of {:.2} s): total {:.4}  MR-STFT {:.4}  mel L1 {:.4}  sc {:?}  logmag {:?}",
            self.crops,
            self.seconds,
            a[0],
            a[1],
            a[2],
            a[3..3 + n_res].iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(),
            a[3 + n_res..].iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>()
        )
    }
}

fn train(a: &Args) -> Result<()> {
    let cfg = a.cfg()?;
    let run = PathBuf::from(a.opt("--run").ok_or_else(|| anyhow!("--run"))?);
    std::fs::create_dir_all(&run)?;
    let steps: usize = a.get("--steps", 200);
    let total: usize = a.get("--total", 2000);
    let batch: usize = a.get("--batch", 8);
    let frames: usize = a.get("--frames", 64);
    let peak: f64 = a.get("--lr", 5e-4);
    let wd: f64 = a.get("--wd", 0.01);
    let clip: f64 = a.get("--clip", 1.0);
    let log_every: usize = a.get("--log-every", 10);
    let ckpt_every: usize = a.get("--ckpt-every", 1000);
    let eval_every: usize = a.get("--eval-every", 0);
    let keep_every: usize = a.get("--keep-every", 10000);

    let (ds, held) = load_split(a, &cfg)?;
    let ck_path = run.join("last.safetensors");
    let ck = if ck_path.exists() {
        ckpt::load(&ck_path, &cfg)?
    } else {
        Checkpoint { params: model::init_params(&cfg, 0), adam: None, step: 0 }
    };
    eprintln!("cfg {:?}\nparams {}  start step {}", cfg, cfg.n_params(), ck.step);

    let dev = device();
    let t0 = Instant::now();
    let tg = model::train_graph(&cfg, batch, frames);
    let built_ms = t0.elapsed().as_secs_f64() * 1e3;
    let adam = ck.adam.as_ref().map(|(m, v)| (m.as_slice(), v.as_slice()));
    let inputs = tg.lg.b.assemble(&ck.params, adam, &[]);
    let mut prog = Program::new(&dev, &tg.lg.b.g, &tg.outputs, &refs(&inputs));
    drop(inputs);
    eprintln!("train program: {} kernels, graph build {built_ms:.0} ms, optimize+codegen+upload {:.0} ms", prog.kernels(), prog.compile_ms);
    let mut heval = if eval_every > 0 && !held.clips.is_empty() { Some(HeldoutEval::new(&dev, &cfg, &ck.params, a.get("--eval-frames", 64))) } else { None };
    let b = &tg.lg.b;
    let np = cfg.params().len();
    let r = cfg.loss_res.len();
    let k_pad = b.input_index(tg.lg.audio_pad);
    let k_audio = b.input_index(tg.lg.audio);
    let k_hyper = b.input_index(tg.hyper);
    let k_p: Vec<usize> = tg.lg.params.iter().map(|&t| b.input_index(t)).collect();
    let k_m: Vec<usize> = tg.m.iter().map(|&t| b.input_index(t)).collect();
    let k_v: Vec<usize> = tg.v.iter().map(|&t| b.input_index(t)).collect();
    let o_p = 4 + 2 * r;
    let (o_m, o_v) = (o_p + np, o_p + 2 * np);

    let mut rng = Rng::new(1000 + ck.step as u64);
    let n = frames * cfg.hop;
    use std::io::Write;
    let open_log = |name: &str, header: &str| -> Result<std::fs::File> {
        let p = run.join(name);
        let new = !p.exists();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p)?;
        if new {
            writeln!(f, "{header}")?;
        }
        Ok(f)
    };
    let mut log = open_log("log.tsv", "step\ttotal\tmrstft\tmel_l1\tgrad_norm\tlr\tms")?;
    let mut elog = open_log("eval.tsv", "step\ttotal\tmrstft\tmel_l1\tsc_res\tlogmag_res\twall_s")?;
    let save = |prog: &Program, step: usize, keep: bool| -> Result<Vec<Vec<f32>>> {
        let params: Vec<Vec<f32>> = k_p.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect();
        let m: Vec<Vec<f32>> = k_m.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect();
        let v: Vec<Vec<f32>> = k_v.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect();
        let ck = Checkpoint { params, adam: Some((m, v)), step };
        ckpt::save(&ck_path, &cfg, &ck)?;
        if keep {
            ckpt::save(&run.join(format!("step{step:07}.safetensors")), &cfg, &Checkpoint { adam: None, params: ck.params.clone(), step })?;
        }
        Ok(ck.params)
    };
    let do_eval = |heval: &mut Option<HeldoutEval>, elog: &mut std::fs::File, params: &[Vec<f32>], step: usize| -> Result<()> {
        let Some(h) = heval.as_mut() else { return Ok(()) };
        let t = Instant::now();
        let l = h.run(&dev, &cfg, params, &held);
        let a = &l.avg;
        let join = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
        writeln!(elog, "{step}\t{}\t{}\t{}\t{}\t{}\t{:.0}", a[0], a[1], a[2], join(&a[3..3 + r]), join(&a[3 + r..]), t0.elapsed().as_secs_f64())?;
        eprintln!("step {step:>7}  {}  ({:.1} s)", l.line(r), t.elapsed().as_secs_f64());
        Ok(())
    };
    if ck.step == 0 {
        do_eval(&mut heval, &mut elog, &ck.params, 0)?;
    }
    drop(ck.adam);
    let start = ck.step;
    let mut step = ck.step;
    let mut times = Vec::new();
    let mut acc = [0.0f64; 4];
    let mut acc_n = 0;
    let mut skipped = 0;
    let stop = run.join("STOP");
    while step < start + steps {
        let (xp, x) = ds.batch(&mut rng, batch, n, cfg.mel_pad());
        let lr = lr_at(step, total, peak, 100.min(total / 10).max(1));
        let t = (step + 1) as i32;
        let hyper = [lr as f32, (1.0 - model::BETA1.powi(t)) as f32, (1.0 - model::BETA2.powi(t)) as f32, wd as f32, clip as f32, 0.0];
        let ts = Instant::now();
        prog.write_input(&dev, k_pad, &xp);
        prog.write_input(&dev, k_audio, &x);
        prog.write_input(&dev, k_hyper, &hyper);
        prog.run(&dev);
        let l: Vec<f32> = (0..4).map(|i| prog.read(&dev, i)[0]).collect();
        if !(l[0].is_finite() && l[3].is_finite()) {
            // ping-pong makes rollback free: not swapping keeps p, m, v
            skipped += 1;
            eprintln!("step {}: non-finite loss/grad norm ({}, {}), update skipped", step + 1, l[0], l[3]);
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
        let ms = ts.elapsed().as_secs_f64() * 1e3;
        times.push(ms);
        step += 1;
        for i in 0..4 {
            acc[i] += l[i] as f64;
        }
        acc_n += 1;
        writeln!(log, "{step}\t{}\t{}\t{}\t{}\t{lr:.3e}\t{ms:.2}", l[0], l[1], l[2], l[3])?;
        if step % log_every == 0 {
            let k = acc_n as f64;
            eprintln!(
                "step {step:>7}  total {:.4}  mrstft {:.4}  mel_l1 {:.4}  |g| {:.3}  lr {lr:.2e}  {ms:.1} ms/step",
                acc[0] / k,
                acc[1] / k,
                acc[2] / k,
                acc[3] / k
            );
            acc = [0.0; 4];
            acc_n = 0;
        }
        let at_eval = eval_every > 0 && step % eval_every == 0;
        let stopping = stop.exists();
        if step % ckpt_every == 0 || at_eval || stopping {
            let params = save(&prog, step, keep_every > 0 && step % keep_every == 0)?;
            log.flush()?;
            if at_eval {
                do_eval(&mut heval, &mut elog, &params, step)?;
            }
        }
        if stopping {
            eprintln!("STOP file found: saved step {step}, exiting");
            std::fs::remove_file(&stop).ok();
            return Ok(());
        }
    }
    // the first steps carry pipeline warm-up; report the rest
    let steady: Vec<f64> = times.iter().skip(2).copied().collect();
    if !steady.is_empty() {
        let mean = steady.iter().sum::<f64>() / steady.len() as f64;
        let mut s = steady.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "timing (fwd+bwd+AdamW, batch {batch} × {frames} frames = {:.2} s audio): mean {mean:.1} ms/step, median {:.1}, {:.2} steps/s",
            (batch * n) as f64 / cfg.sr as f64,
            s[s.len() / 2],
            1e3 / mean
        );
    }
    let params = save(&prog, step, true)?;
    if eval_every > 0 && step % eval_every != 0 {
        do_eval(&mut heval, &mut elog, &params, step)?;
    }
    eprintln!("saved {}", ck_path.display());
    Ok(())
}
// ---------------------------------------------------------------------------
// frontend / inference

/// Log-mel of a whole clip, `[n_mels, T]` with `T = len / hop` (clip
/// truncated to a multiple of hop), computed in the IR on the GPU.
fn wav_to_mel(dev: &Device, cfg: &Config, x: &[f32]) -> (Vec<f32>, usize) {
    let frames = x.len() / cfg.hop;
    let xp = dsp::reflect_pad(&x[..frames * cfg.hop], cfg.mel_pad());
    let fe = model::frontend_graph(cfg, 1, frames);
    let inputs = fe.b.assemble(&[], None, &[("audio_pad", &xp)]);
    let mut prog = Program::new(dev, &fe.b.g, &[fe.mel], &refs(&inputs));
    prog.run(dev);
    (prog.read(&dev, 0), frames)
}

struct Vocoder {
    prog: Program,
    k_mel: usize,
    k_f0: Option<usize>,
}

impl Vocoder {
    fn new(dev: &Device, cfg: &Config, params: &[Vec<f32>], frames: usize) -> Self {
        let inf = model::infer_graph(cfg, 1, frames);
        let inputs = inf.b.assemble(params, None, &[]);
        let prog = Program::new(dev, &inf.b.g, &[inf.audio], &refs(&inputs));
        Self { k_mel: inf.b.input_index(inf.mel), k_f0: inf.f0.map(|t| inf.b.input_index(t)), prog }
    }
    fn run(&mut self, dev: &Device, mel: &[f32]) -> Vec<f32> {
        self.run_f0(dev, mel, None)
    }
    fn run_f0(&mut self, dev: &Device, mel: &[f32], f0: Option<&[f32]>) -> Vec<f32> {
        if let (Some(k), Some(f)) = (self.k_f0, f0) {
            self.prog.write_input(dev, k, f);
        }
        self.prog.write_input(dev, self.k_mel, mel);
        self.prog.run(dev);
        self.prog.read(dev, 0)
    }
}

fn load_params(a: &Args, cfg: &Config) -> Result<Vec<Vec<f32>>> {
    Ok(match a.opt("--ckpt") {
        Some(p) => ckpt::load(Path::new(&p), cfg)?.params,
        None => {
            eprintln!("(no --ckpt: random init)");
            model::init_params(cfg, 0)
        }
    })
}

fn infer(a: &Args) -> Result<()> {
    let cfg = a.cfg()?;
    let pos = a.positional();
    let [inp, out] = &pos[..] else { return Err(anyhow!("infer IN OUT")) };
    let dev = device();
    let params = load_params(a, &cfg)?;
    let mut f0 = None;
    let (mel, frames) = if inp.ends_with(".npy") {
        if cfg.f0 {
            return Err(anyhow!("an f0 config needs a wav input (f0 is estimated from it)"));
        }
        let (shape, v) = data::read_npy(Path::new(inp))?;
        let [m, t] = shape[..] else { return Err(anyhow!("mel npy must be [n_mels, T]")) };
        if m != cfg.n_mels {
            return Err(anyhow!("mel has {m} bins, cfg wants {}", cfg.n_mels));
        }
        (v, t)
    } else {
        let (x, sr) = data::read_wav(Path::new(inp))?;
        if sr as usize != cfg.sr {
            return Err(anyhow!("{sr} Hz input; resample to {} first", cfg.sr));
        }
        let (mut m, mut frames) = wav_to_mel(&dev, &cfg, &x);
        if cfg.f0 {
            let xp = dsp::reflect_pad(&x[..frames * cfg.hop], cfg.mel_pad());
            f0 = Some(src_of(&cfg, &xp, 1, frames));
        }
        // §16: vocode a given mel `[n_mels, T]` (and F0 Hz `[T]`) instead, e.g. a
        // teacher-forced prediction aligned with this wav
        if let Some(mp) = a.opt("--mel-npy") {
            let (shape, v) = data::read_npy(Path::new(&mp))?;
            let [mm, t] = shape[..] else { return Err(anyhow!("mel npy must be [n_mels, T]")) };
            if mm != cfg.n_mels {
                return Err(anyhow!("mel has {mm} bins"));
            }
            (m, frames) = (v, t);
            if cfg.f0 {
                f0 = Some(match a.opt("--f0-npy") {
                    Some(fp) => src_rows(&cfg, &data::read_npy(Path::new(&fp))?.1, 1, t),
                    None => {
                        let xp = dsp::reflect_pad(&x[..t * cfg.hop], cfg.mel_pad());
                        src_of(&cfg, &xp, 1, t)
                    }
                });
            }
        }
        (m, frames)
    };
    let mut voc = Vocoder::new(&dev, &cfg, &params, frames);
    let t = Instant::now();
    let y = voc.run_f0(&dev, &mel, f0.as_deref());
    let ms = t.elapsed().as_secs_f64() * 1e3;
    data::write_wav(Path::new(out), &y, cfg.sr as u32)?;
    eprintln!("{} frames -> {} samples in {ms:.1} ms (RTF {:.4})", frames, y.len(), ms / 1e3 / (y.len() as f64 / cfg.sr as f64));
    Ok(())
}

fn mel(a: &Args) -> Result<()> {
    let cfg = a.cfg()?;
    let pos = a.positional();
    let [inp, out] = &pos[..] else { return Err(anyhow!("mel IN.wav OUT.npy")) };
    let (x, _) = data::read_wav(Path::new(inp))?;
    let dev = device();
    let (m, frames) = wav_to_mel(&dev, &cfg, &x);
    write_npy(Path::new(out), &[cfg.n_mels, frames], &m)
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

fn bench(a: &Args) -> Result<()> {
    let cfg = a.cfg()?;
    let secs: f64 = a.get("--seconds", 1.0);
    let frames = (secs * cfg.sr as f64 / cfg.hop as f64).ceil() as usize;
    let dev = device();
    let params = load_params(a, &cfg)?;
    let mut voc = Vocoder::new(&dev, &cfg, &params, frames);
    eprintln!("inference program: {} kernels, compile {:.0} ms", voc.prog.kernels(), voc.prog.compile_ms);
    let mut rng = Rng::new(9);
    let mel: Vec<f32> = (0..cfg.n_mels * frames).map(|_| (rng.normal() - 4.0) as f32).collect();
    for _ in 0..3 {
        voc.run(&dev, &mel);
    }
    let iters = a.get("--iters", 20usize);
    let mut ts = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        voc.prog.write_input(&dev, voc.k_mel, &mel);
        voc.prog.run(&dev);
        let _ = voc.prog.read(&dev, 0);
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ts[ts.len() / 2];
    let audio = frames as f64 * cfg.hop as f64 / cfg.sr as f64;
    let kernel_ms = voc.prog.time(&dev, iters);
    #[cfg(feature = "metal")]
    if a.flag("--tune") {
        // precision search (f16 storage / f16 fragments on the matrix
        // units) against the f32 GPU output, under a 1e-2 relative budget
        use tang_compile::tensor::ErrorBudget;
        let want: Vec<Vec<f64>> = vec![voc.run(&dev, &mel).iter().map(|&v| v as f64).collect()];
        let inf = model::infer_graph(&cfg, 1, frames);
        let mut inputs = inf.b.assemble(&params, None, &[]);
        inputs[voc.k_mel] = mel.clone();
        let (og, outs) = ribt::runtime::optimize(&inf.b.g, &[inf.audio]);
        let budget = ErrorBudget::rel(a.get("--budget", 1e-2));
        let r = ribt::tang_msl::tune_precision(&dev.0, og, &outs, &refs(&inputs), &want, budget, iters);
        let kept: Vec<&str> = r.trials.iter().filter(|t| t.kept).map(|t| t.what.as_str()).collect();
        println!(
            "tune_precision @ {:.0e} (vs f32 GPU output): {:.3} -> {:.3} ms kernel-only, err {:.1e}, {} buffers f16, {} nodes on f16 fragments; {} trials, kept {:?}",
            budget.max_rel, r.base_ms, r.ms, r.error.max_rel, r.assignment.len(), r.fragments.len(), r.trials.len(), kept
        );
        let over = r.trials.iter().filter(|t| t.ms.is_none()).count();
        println!("  trials over budget: {over}; fragment trials: {:?}", r.trials.iter().filter(|t| t.what.starts_with("fragment")).map(|t| (t.what.clone(), t.error.max_rel, t.ms)).collect::<Vec<_>>());
    }
    println!(
        "tang IR ({}, f32) batch 1, {audio:.3} s audio ({frames} frames): median {med:.2} ms end-to-end (write+run+read), kernel-only {kernel_ms:.2} ms, RTF {:.4}",
        ribt::runtime::BACKEND,
        med / 1e3 / audio
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// eval: held-out losses (the training graph's loss terms) + resynthesis

fn eval(a: &Args) -> Result<()> {
    let cfg = a.cfg()?;
    let run = PathBuf::from(a.opt("--run").ok_or_else(|| anyhow!("--run"))?);
    let out = PathBuf::from(a.opt("--out").unwrap_or_else(|| run.join("eval").display().to_string()));
    std::fs::create_dir_all(&out)?;
    let ck_path = a.opt("--ckpt").map(PathBuf::from).unwrap_or_else(|| run.join("last.safetensors"));
    let params = if a.flag("--init") { model::init_params(&cfg, 0) } else { ckpt::load(&ck_path, &cfg)?.params };
    let held = match a.opt("--heldout") {
        Some(h) => Dataset::heldout(&dirs(&h), cfg.sr as u32, a.get("--heldout-n", 48))?,
        None => Dataset::load(Path::new(&a.opt("--data").ok_or_else(|| anyhow!("--data or --heldout"))?), cfg.sr as u32, 20)?.1,
    };
    let dev = device();
    let mut h = HeldoutEval::new(&dev, &cfg, &params, a.get("--frames", 64));
    println!("{}", h.run(&dev, &cfg, &params, &held).line(cfg.loss_res.len()));
    // resynthesize the first few held-out clips whole
    let n_out: usize = a.get("--n", 6);
    for (p, clip) in held.clips.iter().take(n_out) {
        let (mel, frames) = wav_to_mel(&dev, &cfg, clip);
        let f0 = cfg.f0.then(|| src_of(&cfg, &dsp::reflect_pad(&clip[..frames * cfg.hop], cfg.mel_pad()), 1, frames));
        let mut voc = Vocoder::new(&dev, &cfg, &params, frames);
        let y = voc.run_f0(&dev, &mel, f0.as_deref());
        let stem = p.file_stem().unwrap().to_string_lossy();
        data::write_wav(&out.join(format!("{stem}_ref.wav")), &clip[..frames * cfg.hop], cfg.sr as u32)?;
        data::write_wav(&out.join(format!("{stem}_tang.wav")), &y, cfg.sr as u32)?;
    }
    eprintln!("wrote {} resyntheses to {}", n_out.min(held.clips.len()), out.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// gpucheck: one tiny training step on Metal vs the f64 interpreter

fn gpucheck(a: &Args) -> Result<()> {
    let cfg = Config::by_name(&a.opt("--cfg").unwrap_or_else(|| "tiny".into())).ok_or_else(|| anyhow!("cfg"))?;
    let (batch, frames) = (2, a.get("--frames", 16usize));
    let tg = model::train_graph(&cfg, batch, frames);
    let mut rng = Rng::new(4);
    let mut params = model::init_params(&cfg, 1);
    for p in params.iter_mut() {
        for v in p.iter_mut() {
            *v += (0.1 * rng.normal()) as f32;
        }
    }
    let n = frames * cfg.hop;
    let ds = Dataset {
        clips: vec![(PathBuf::new(), (0..4 * n).map(|i| ((i as f64 * 0.37).sin() * 0.5 + 0.05 * rng.normal()) as f32).collect())],
    };
    let (xp, x) = ds.batch(&mut rng, batch, n, cfg.mel_pad());
    let mut inputs = tg.lg.b.assemble(&params, None, &[("audio_pad", &xp), ("audio", &x)]);
    let kh = tg.lg.b.input_index(tg.hyper);
    inputs[kh] = vec![1e-3, 1.0 - model::BETA1 as f32, 1.0 - model::BETA2 as f32, 0.01, 1.0, 0.0];
    let want = tg.lg.b.g.eval(&inputs.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect::<Vec<_>>(), &tg.outputs);
    let dev = device();
    let mut prog = Program::new(&dev, &tg.lg.b.g, &tg.outputs, &refs(&inputs));
    prog.run(&dev);
    let mut worst = 0.0f64;
    // p' = p − lr·m̂/√v̂ is ≈ p − lr·sign(g) on the first step, so a
    // near-zero gradient's f32 sign flip moves it by 2·lr: compare the
    // gradients through m' = (1−β₁)·g instead and skip p'.
    let np = cfg.params().len();
    let o_p = 4 + 2 * cfg.loss_res.len();
    for (i, w) in want.iter().enumerate() {
        if (o_p..o_p + np).contains(&i) {
            continue;
        }
        let got = prog.read(&dev, i);
        let scale = w.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-6);
        let e = got.iter().zip(w).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max) / scale;
        if e > 1e-3 {
            eprintln!("  output {i}: rel err {e:.2e} (scale {scale:.3e})");
        }
        worst = worst.max(e);
    }
    println!(
        "gpucheck: train step ({} kernels): loss gpu {:.6} vs f64 {:.6}; worst rel err over {} outputs {worst:.2e}",
        prog.kernels(),
        prog.read(&dev, 0)[0],
        want[0][0],
        want.len()
    );
    if worst > 1e-3 {
        return Err(anyhow!("GPU disagrees with the interpreter"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// stepbench: kernel-only ms per training step on synthetic audio (no data,
// no host-side batching), for comparing backends

fn stepbench(a: &Args) -> Result<()> {
    let cfg = a.cfg()?;
    let (batch, frames, iters) = (a.get("--batch", 8usize), a.get("--frames", 64usize), a.get("--iters", 50usize));
    let tg = model::train_graph(&cfg, batch, frames);
    let mut rng = Rng::new(4);
    let params = model::init_params(&cfg, 1);
    let n = frames * cfg.hop;
    let ds = Dataset { clips: vec![(PathBuf::new(), (0..4 * n).map(|i| ((i as f64 * 0.37).sin() * 0.5 + 0.05 * rng.normal()) as f32).collect())] };
    let (xp, x) = ds.batch(&mut rng, batch, n, cfg.mel_pad());
    let mut inputs = tg.lg.b.assemble(&params, None, &[("audio_pad", &xp), ("audio", &x)]);
    let kh = tg.lg.b.input_index(tg.hyper);
    inputs[kh] = vec![0.0, 1.0 - model::BETA1 as f32, 1.0 - model::BETA2 as f32, 0.0, 1.0, 0.0];
    let dev = device();
    let mut prog = Program::new(&dev, &tg.lg.b.g, &tg.outputs, &refs(&inputs));
    let ms = prog.time(&dev, iters);
    println!(
        "stepbench {}: cfg {} batch {batch} x {frames} frames, {} kernels, compile {:.0} ms, {ms:.2} ms/step ({:.1} steps/s), loss {:.4}",
        ribt::runtime::BACKEND,
        a.opt("--cfg").unwrap_or_else(|| "v1".into()),
        prog.kernels(),
        prog.compile_ms,
        1e3 / ms,
        prog.read(&dev, 0)[0]
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// gan: V1.1 adversarial fine-tune (MPD + MRD), alternating G and D programs

/// `[lr, 1−β1^t, 1−β2^t, wd, clip, adv_scale]` for Adam step `step` (0-based).
fn hyper(step: usize, lr: f64, wd: f64, clip: f64, w: &ribt::gan::Weights, adv: f64) -> [f32; 6] {
    let t = (step + 1) as i32;
    [lr as f32, (1.0 - w.beta1.powi(t)) as f32, (1.0 - w.beta2.powi(t)) as f32, wd as f32, clip as f32, adv as f32]
}

/// Predicted-mel pair stores for the vocoder fine-tune (§16), written by
/// `ribt-tts voc-pairs`: `index.tsv` (id, frames, frame offset), `mel.i16`
/// (per utt `[frames, n_mels]`, log-mel × 1000) and `f0.f32` (Hz, 0 unvoiced).
struct PairStore {
    by_id: std::collections::HashMap<String, (usize, usize)>,
    mel: Vec<i16>,
    f0: Vec<f32>,
    /// hidden-decoder rung 1: `hid.f32` (per utt `[frames, hid]`, `--hid` of voc-pairs)
    hid: Vec<f32>,
}

impl PairStore {
    fn load(dir: &Path, n_mels: usize, hid: usize) -> Result<Self> {
        let mut by_id = std::collections::HashMap::new();
        for l in std::fs::read_to_string(dir.join("index.tsv"))?.lines().skip(1) {
            let c: Vec<&str> = l.split('\t').collect();
            if c.len() >= 3 {
                by_id.insert(c[0].to_string(), (c[1].parse()?, c[2].parse()?));
            }
        }
        let mb = std::fs::read(dir.join("mel.i16"))?;
        let mel: Vec<i16> = mb.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
        let fb = std::fs::read(dir.join("f0.f32"))?;
        let f0: Vec<f32> = fb.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        if mel.len() != f0.len() * n_mels {
            return Err(anyhow!("{}: mel/f0 sizes disagree", dir.display()));
        }
        let hid = if hid > 0 {
            let hb = std::fs::read(dir.join("hid.f32"))?;
            let h: Vec<f32> = hb.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            if h.len() != f0.len() * hid {
                return Err(anyhow!("{}: hid/f0 sizes disagree", dir.display()));
            }
            h
        } else {
            vec![]
        };
        Ok(Self { by_id, mel, f0, hid })
    }
}

/// Per crop: the predicted mel (+ ln gain, floored) and F0 of a pair store,
/// or None (frontend mel). Fills `mel_ext`/`mel_sel` `[B, n_mels, T]` and the
/// F0 source rows.
#[allow(clippy::too_many_arguments)]
fn pair_inputs(cfg: &Config, stores: &[PairStore], ids: &[String], meta: &[(usize, usize, f32)], xp: &[f32], frames: usize, rng: &mut Rng, p_pred: f64, pred_f0: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let (b, m) = (meta.len(), cfg.in_ch());
    let floor = if cfg.hid > 0 { 0.0 } else { 1e-5f32.ln() };
    let mut ext = vec![floor; b * m * frames];
    let mut sel = vec![if cfg.hid > 0 { 1.0 } else { 0.0 }; b * m * frames];
    let mut f0 = vec![0.0f32; b * frames];
    let row = xp.len() / b;
    let mut n_pred = 0;
    for (j, &(ci, s0, gain)) in meta.iter().enumerate() {
        let id = &ids[ci];
        let have: Vec<usize> = (0..stores.len()).filter(|&k| stores[k].by_id.contains_key(id)).collect();
        let pick = (!have.is_empty() && (cfg.hid > 0 || rng.uniform() < p_pred)).then(|| have[rng.below(have.len())]);
        assert!(pick.is_some() || cfg.hid == 0, "hidden-input G: clip {id} has no pair (use --pairs-only)");
        let yin = || dsp::yin_f0(&xp[j * row..(j + 1) * row], cfg.sr, cfg.n_fft, cfg.hop, frames);
        match pick {
            Some(k) => {
                n_pred += 1;
                let st = &stores[k];
                let (nf, off) = st.by_id[id];
                let (j0, lg) = (s0 / cfg.hop, gain.ln());
                for t in 0..frames {
                    let src = j0 + t;
                    if cfg.hid > 0 {
                        // rows: hidden, ones, ln gain (zero past the utterance)
                        let hd = cfg.hid;
                        if src < nf {
                            for c in 0..hd {
                                ext[(j * m + c) * frames + t] = st.hid[(off + src) * hd + c];
                            }
                            ext[(j * m + hd) * frames + t] = 1.0;
                            ext[(j * m + hd + 1) * frames + t] = lg;
                        }
                        continue;
                    }
                    for c in 0..m {
                        let i = (j * m + c) * frames + t;
                        sel[i] = 1.0;
                        if src < nf {
                            ext[i] = (st.mel[(off + src) * m + c] as f32 * 1e-3 + lg).max(floor);
                        }
                    }
                }
                if pred_f0 {
                    for t in 0..frames {
                        f0[j * frames + t] = if j0 + t < nf { st.f0[off + j0 + t] } else { 0.0 };
                    }
                } else {
                    f0[j * frames..(j + 1) * frames].copy_from_slice(&yin());
                }
            }
            None => f0[j * frames..(j + 1) * frames].copy_from_slice(&yin()),
        }
    }
    let src = src_rows(cfg, &f0, b, frames);
    (ext, sel, src, n_pred)
}

fn train_gan(a: &Args) -> Result<()> {
    use ribt::gan::{self, DConfig, Weights};
    use std::io::Write;
    let cfg = Config::by_name(&a.opt("--cfg").unwrap_or_else(|| "v1f0".into())).ok_or_else(|| anyhow!("cfg"))?;
    let dcfg = DConfig::by_name(&a.opt("--dcfg").unwrap_or_else(|| "v12".into())).ok_or_else(|| anyhow!("dcfg"))?;
    let run = PathBuf::from(a.opt("--run").ok_or_else(|| anyhow!("--run"))?);
    std::fs::create_dir_all(&run)?;
    let steps: usize = a.get("--steps", 200);
    let total: usize = a.get("--total", 200000);
    let batch: usize = a.get("--batch", 16);
    let frames: usize = a.get("--frames", 64);
    let lr_g: f64 = a.get("--lr", 2e-4);
    let lr_d: f64 = a.get("--lr-d", 2e-4);
    // clipping per network; Vocos and HiFi-GAN clip neither (1e9 = off)
    let (wd, clip, clip_d): (f64, f64, f64) = (a.get("--wd", 0.01), a.get("--clip", 1e9), a.get("--clip-d", 1e9));
    // D-only warm-up (G frozen, no adversarial gradient), then the adversarial
    // + FM weight ramps 0 → 1 over `--adv-ramp` G steps
    let d_warmup: usize = a.get("--d-warmup", 0);
    let adv_ramp: usize = a.get("--adv-ramp", 0);
    let w = Weights {
        mel: a.get("--w-mel", 45.0),
        mrstft: a.get("--w-mrstft", 1.0),
        adv: a.get("--w-adv", 1.0),
        fm: a.get("--w-fm", 1.0),
        mrd: a.get("--w-mrd", 0.1),
        beta1: a.get("--beta1", 0.8),
        beta2: a.get("--beta2", 0.9),
        phase: a.get("--w-phase", 0.0),
        phase_terms: [a.get("--w-phase-ip", 1.0), a.get("--w-phase-gd", 1.0), a.get("--w-phase-ptd", 1.0)],
        phase_mag: a.flag("--phase-mag"),
    };
    let (log_every, ckpt_every, eval_every, keep_every) = (a.get("--log-every", 10usize), a.get("--ckpt-every", 1000usize), a.get("--eval-every", 0usize), a.get("--keep-every", 10000usize));

    let (mut ds, held) = load_split(a, &cfg)?;
    // §16: fine-tune on predicted mels (pair stores) mixed with frontend mels
    let stores: Vec<PairStore> = match a.opt("--pairs") {
        Some(p) => dirs(&p).iter().map(|d| PairStore::load(d, cfg.n_mels, cfg.hid)).collect::<Result<_>>()?,
        None => vec![],
    };
    // hidden-decoder rung 1: train only on clips a pair store covers
    if a.flag("--pairs-only") || cfg.hid > 0 {
        if let TrainSet::Corpus(c) = &mut ds {
            let n0 = c.clips.len();
            c.clips.retain(|(p, _)| {
                let id = p.file_stem().unwrap().to_string_lossy().into_owned();
                stores.iter().any(|s| s.by_id.contains_key(&id))
            });
            eprintln!("pairs-only: {}/{n0} training clips kept", c.clips.len());
        }
    }
    let labels = if cfg.range == 2 { Some(load_labels(Path::new(&a.opt("--labels").ok_or_else(|| anyhow!("--labels FILE for the physics range"))?))?) } else { None };
    let paths: Vec<String> = match &ds { TrainSet::Corpus(c) => c.clips.iter().map(|(p, _)| p.display().to_string()).collect(), _ => vec![] };
    if let Some(l) = &labels {
        let miss = paths.iter().filter(|p| !l.by_path.contains_key(*p)).count();
        if miss > 0 {
            return Err(anyhow!("{miss} training clips have no labels"));
        }
    }
    let (g_path, d_path) = (run.join("last.safetensors"), run.join("d_last.safetensors"));
    let dspecs = dcfg.params();
    let gck = if g_path.exists() {
        ckpt::load_specs(&g_path, &cfg.params(), false)?
    } else {
        let init = a.opt("--init").ok_or_else(|| anyhow!("--init V1 checkpoint (no {} yet)", g_path.display()))?;
        let mut c = ckpt::load_specs(Path::new(&init), &cfg.params(), true)?;
        // fresh Adam for the new objective; steps restart at 0
        c.adam = None;
        c.step = 0;
        if a.flag("--reset-mag-head") {
            // §11.5: the head's log-mag now scales the range candidate; start at exp(0) = 1
            for (s, p) in cfg.params().iter().zip(c.params.iter_mut()) {
                if s.name.starts_with("head.mag.") {
                    p.iter_mut().for_each(|v| *v = 0.0);
                }
            }
        }
        c
    };
    let dck = if d_path.exists() { ckpt::load_specs(&d_path, &dspecs, false)? } else { Checkpoint { params: gan::init_dparams(&dcfg, 5), adam: None, step: 0 } };
    eprintln!("cfg {:?}\ndcfg {:?}\nweights {:?}  clip G {clip:e} D {clip_d:e}  d-warmup {d_warmup}  adv-ramp {adv_ramp}\nG params {}  D params {}  start step {}", cfg, dcfg, w, cfg.n_params(), dspecs.iter().map(|s| s.numel()).sum::<usize>(), gck.step);

    let dev = device();
    let t0 = Instant::now();
    let (p_pred, pred_f0): (f64, bool) = (a.get("--p-pred", 0.5), a.opt("--pair-f0").as_deref() != Some("yin"));
    let ids: Vec<String> = paths.iter().map(|p| Path::new(p).file_stem().unwrap().to_string_lossy().into_owned()).collect();
    if !stores.is_empty() {
        let cov = ids.iter().filter(|i| stores.iter().any(|s| s.by_id.contains_key(*i))).count();
        eprintln!("pairs: {} stores, {cov}/{} training clips covered, p_pred {p_pred}, F0 {}", stores.len(), ids.len(), if pred_f0 { "predicted" } else { "YIN" });
    }
    let gg = gan::gen_graph_ext(&cfg, &dcfg, &w, batch, frames, !stores.is_empty());
    let dg = gan::disc_graph(&cfg, &dcfg, &w, batch, frames);
    let gadam = gck.adam.as_ref().map(|(m, v)| (m.as_slice(), v.as_slice()));
    let dadam = dck.adam.as_ref().map(|(m, v)| (m.as_slice(), v.as_slice()));
    let mut gprog = Program::new(&dev, &gg.lg.b.g, &gg.outputs, &refs(&gg.lg.b.assemble_gd(&gck.params, gadam, &dck.params, None, &[])));
    let mut dprog = Program::new(&dev, &dg.b.g, &dg.outputs, &refs(&dg.b.assemble_gd(&[], None, &dck.params, dadam, &[])));
    eprintln!("G program {} kernels ({:.0} ms), D program {} kernels ({:.0} ms), total {:.0} ms", gprog.kernels(), gprog.compile_ms, dprog.kernels(), dprog.compile_ms, t0.elapsed().as_secs_f64() * 1e3);
    // the held-out eval vocodes frontend mels: not defined for a hidden-input G
    let mut heval = if eval_every > 0 && !held.clips.is_empty() && cfg.hid == 0 { Some(HeldoutEval::new(&dev, &cfg, &gck.params, a.get("--eval-frames", 64))) } else { None };
    let (gb, db) = (&gg.lg.b, &dg.b);
    let (np, nd) = (cfg.params().len(), dspecs.len());
    let gk_p: Vec<usize> = gg.lg.params.iter().map(|&t| gb.input_index(t)).collect();
    let gk_m: Vec<usize> = gg.m.iter().map(|&t| gb.input_index(t)).collect();
    let gk_v: Vec<usize> = gg.v.iter().map(|&t| gb.input_index(t)).collect();
    let gk_d: Vec<usize> = gg.dparams.iter().map(|&t| gb.input_index(t)).collect();
    let (gk_pad, gk_audio, gk_hyper) = (gb.input_index(gg.lg.audio_pad), gb.input_index(gg.lg.audio), gb.input_index(gg.hyper));
    let gk_f0 = gg.lg.f0.map(|t| gb.input_index(t));
    let gk_ext = gg.lg.mel_ext.map(|(e, s)| (gb.input_index(e), gb.input_index(s)));
    let gk_phys = gb.find(|r| matches!(r, model::Role::Data("phys"))).map(|t| gb.input_index(t));
    let gk_f0hz = gb.find(|r| matches!(r, model::Role::Data("f0hz"))).map(|t| gb.input_index(t));
    let go_p = gan::GEN_HEAD;
    let go_y = go_p + 3 * np;
    let dk_p: Vec<usize> = dg.dparams.iter().map(|&t| db.input_index(t)).collect();
    let dk_m: Vec<usize> = dg.dm.iter().map(|&t| db.input_index(t)).collect();
    let dk_v: Vec<usize> = dg.dv.iter().map(|&t| db.input_index(t)).collect();
    let (dk_audio, dk_fake, dk_hyper) = (db.input_index(dg.audio), db.input_index(dg.fake), db.input_index(dg.hyper));
    let do_p = gan::DISC_HEAD;

    let open_log = |name: &str, header: &str| -> Result<std::fs::File> {
        let p = run.join(name);
        let new = !p.exists();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p)?;
        if new {
            writeln!(f, "{header}")?;
        }
        Ok(f)
    };
    let mut log = open_log("log.tsv", "step\tg_total\tmrstft\tmel_l1\tg_norm\tadv_mpd\tadv_mrd\tfm_mpd\tfm_mrd\td_total\td_mpd\td_mrd\td_norm\tlr\tms\tadv_scale")?;
    let mut plog = open_log("phase.tsv", "step\tip\tgd\tptd")?;
    let mut elog = open_log("eval.tsv", "step\ttotal\tmrstft\tmel_l1\tsc_res\tlogmag_res\twall_s")?;
    let r = cfg.loss_res.len();
    let save = |gprog: &Program, dprog: &Program, step: usize, keep: bool| -> Result<Vec<Vec<f32>>> {
        let rd = |p: &Program, ks: &[usize]| -> Vec<Vec<f32>> { ks.iter().map(|&k| p.read_input(&dev, k).unwrap()).collect() };
        let gc = Checkpoint { params: rd(gprog, &gk_p), adam: Some((rd(gprog, &gk_m), rd(gprog, &gk_v))), step };
        ckpt::save_specs(&g_path, &cfg.params(), &gc)?;
        let dc = Checkpoint { params: rd(dprog, &dk_p), adam: Some((rd(dprog, &dk_m), rd(dprog, &dk_v))), step };
        ckpt::save_specs(&d_path, &dspecs, &dc)?;
        if keep {
            ckpt::save_specs(&run.join(format!("step{step:07}.safetensors")), &cfg.params(), &Checkpoint { adam: None, params: gc.params.clone(), step })?;
        }
        Ok(gc.params)
    };
    let do_eval = |heval: &mut Option<HeldoutEval>, elog: &mut std::fs::File, params: &[Vec<f32>], step: usize| -> Result<()> {
        let Some(h) = heval.as_mut() else { return Ok(()) };
        let t = Instant::now();
        let l = h.run(&dev, &cfg, params, &held);
        let av = &l.avg;
        let join = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
        writeln!(elog, "{step}\t{}\t{}\t{}\t{}\t{}\t{:.0}", av[0], av[1], av[2], join(&av[3..3 + r]), join(&av[3 + r..]), t0.elapsed().as_secs_f64())?;
        eprintln!("step {step:>7}  {}  ({:.1} s)", l.line(r), t.elapsed().as_secs_f64());
        Ok(())
    };
    if gck.step == 0 {
        do_eval(&mut heval, &mut elog, &gck.params, 0)?;
    }
    let start = gck.step;
    let mut step = start;
    // --seed 0 is the old stream exactly
    let mut rng = Rng::new(7000 + step as u64 + 1_000_003 * a.get("--seed", 0u64));
    let n = frames * cfg.hop;
    let mut acc = [0.0f64; 12];
    let mut acc_n = 0;
    let mut acc_ph = [0.0f64; 3];
    let mut skipped = 0;
    let stop = run.join("STOP");
    drop(gck);
    drop(dck);
    while step < start + steps {
        let ts = Instant::now();
        let (xp, x) = match (&ds, &labels) {
            (TrainSet::Corpus(c), Some(l)) => {
                let (xp, x, meta) = c.batch_meta(&mut rng, batch, n, cfg.mel_pad(), cfg.hop);
                let (th, f0) = crop_labels(l, &paths, &meta, cfg.hop, frames);
                gprog.write_input(&dev, gk_phys.unwrap(), &th);
                gprog.write_input(&dev, gk_f0hz.unwrap(), &f0);
                (xp, x)
            }
            (TrainSet::Corpus(c), None) if gk_ext.is_some() => {
                let (xp, x, meta) = c.batch_meta(&mut rng, batch, n, cfg.mel_pad(), cfg.hop);
                let (ext, sel, src, _) = pair_inputs(&cfg, &stores, &ids, &meta, &xp, frames, &mut rng, p_pred, pred_f0);
                let (ke, ks) = gk_ext.unwrap();
                gprog.write_input(&dev, ke, &ext);
                gprog.write_input(&dev, ks, &sel);
                if let Some(k) = gk_f0 {
                    gprog.write_input(&dev, k, &src);
                }
                (xp, x)
            }
            _ => ds.batch(&mut rng, batch, n, cfg.mel_pad()),
        };
        if let (Some(k), None) = (gk_f0, gk_ext) {
            gprog.write_input(&dev, k, &src_of(&cfg, &xp, batch, frames));
        }
        // G's own step count (the D warm-up does not advance it)
        let in_warmup = step < d_warmup;
        let gstep = step.saturating_sub(d_warmup);
        let lr_eff = if in_warmup { 0.0 } else { lr_at(gstep, total, lr_g, 100.min(total / 10).max(1)) };
        let adv_scale = if in_warmup { 0.0 } else if adv_ramp == 0 { 1.0 } else { ((gstep + 1) as f64 / adv_ramp as f64).min(1.0) };
        gprog.write_input(&dev, gk_pad, &xp);
        gprog.write_input(&dev, gk_audio, &x);
        gprog.write_input(&dev, gk_hyper, &hyper(gstep, lr_eff, wd, clip, &w, adv_scale));
        gprog.run(&dev);
        let gl: Vec<f32> = (0..gan::GEN_HEAD).map(|i| gprog.read(&dev, i)[0]).collect();
        if !gl.iter().all(|v| v.is_finite()) {
            skipped += 1;
            eprintln!("step {}: non-finite G step {gl:?}, skipped", step + 1);
            if skipped > 20 {
                return Err(anyhow!("too many non-finite steps"));
            }
            continue;
        }
        let y = gprog.read(&dev, go_y);
        let ph: Vec<f32> = gg.phase_out.map(|k| (0..3).map(|i| gprog.read(&dev, k + i)[0]).collect()).unwrap_or_default();
        if !in_warmup {
            for i in 0..np {
                gprog.swap_input_output(gk_p[i], go_p + i);
                gprog.swap_input_output(gk_m[i], go_p + np + i);
                gprog.swap_input_output(gk_v[i], go_p + 2 * np + i);
            }
        }
        // D step on this step's real and generated audio
        dprog.write_input(&dev, dk_audio, &x);
        dprog.write_input(&dev, dk_fake, &y);
        dprog.write_input(&dev, dk_hyper, &hyper(step, lr_at(step, total, lr_d, 100.min(total / 10).max(1)), wd, clip_d, &w, 1.0));
        dprog.run(&dev);
        let dl: Vec<f32> = (0..gan::DISC_HEAD).map(|i| dprog.read(&dev, i)[0]).collect();
        if dl.iter().all(|v| v.is_finite()) {
            for i in 0..nd {
                dprog.swap_input_output(dk_p[i], do_p + i);
                dprog.swap_input_output(dk_m[i], do_p + nd + i);
                dprog.swap_input_output(dk_v[i], do_p + 2 * nd + i);
            }
            // the G program reads D's new params
            for i in 0..nd {
                gprog.write_input(&dev, gk_d[i], &dprog.read_input(&dev, dk_p[i]).unwrap());
            }
        } else {
            eprintln!("step {}: non-finite D step {dl:?}, D update skipped", step + 1);
        }
        step += 1;
        let ms = ts.elapsed().as_secs_f64() * 1e3;
        let row: Vec<f64> = gl.iter().chain(&dl).map(|&v| v as f64).collect();
        for (a, v) in acc.iter_mut().zip(&row) {
            *a += v;
        }
        acc_n += 1;
        writeln!(log, "{step}\t{}\t{lr_eff:.3e}\t{ms:.1}\t{adv_scale:.3}", row.iter().map(|v| format!("{v:.5}")).collect::<Vec<_>>().join("\t"))?;
        if !ph.is_empty() {
            for (a, &v) in acc_ph.iter_mut().zip(&ph) {
                *a += v as f64;
            }
            writeln!(plog, "{step}\t{:.5}\t{:.5}\t{:.5}", ph[0], ph[1], ph[2])?;
        }
        if step % log_every == 0 {
            let k = acc_n as f64;
            let m: Vec<f64> = acc.iter().map(|v| v / k).collect();
            eprintln!(
                "step {step:>7}  G {:.3} (mrstft {:.4} mel {:.4} adv {:.3}/{:.3} fm {:.3}/{:.3})  D {:.3} ({:.3}/{:.3})  |g| {:.2}/{:.2}  lr {lr_eff:.1e}  {ms:.0} ms",
                m[0], m[1], m[2], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[3], m[11]
            );
            if gg.phase_out.is_some() {
                eprintln!("step {step:>7}  phase ip {:.4} gd {:.4} ptd {:.4}", acc_ph[0] / k, acc_ph[1] / k, acc_ph[2] / k);
            }
            acc = [0.0; 12];
            acc_ph = [0.0; 3];
            acc_n = 0;
        }
        let at_eval = eval_every > 0 && step % eval_every == 0;
        let stopping = stop.exists();
        if step % ckpt_every == 0 || at_eval || stopping {
            let params = save(&gprog, &dprog, step, keep_every > 0 && step % keep_every == 0)?;
            log.flush()?;
            if at_eval {
                do_eval(&mut heval, &mut elog, &params, step)?;
            }
        }
        if stopping {
            eprintln!("STOP file found: saved step {step}, exiting");
            std::fs::remove_file(&stop).ok();
            return Ok(());
        }
    }
    let params = save(&gprog, &dprog, step, true)?;
    if eval_every > 0 && step % eval_every != 0 {
        do_eval(&mut heval, &mut elog, &params, step)?;
    }
    eprintln!("saved {}", g_path.display());
    Ok(())
}

/// Kernel-only ms of the G and D step programs (synthetic data).
fn dbench(a: &Args) -> Result<()> {
    use ribt::gan::{self, DConfig, Weights};
    let name = a.opt("--cfg").unwrap_or_else(|| "v1f0".into());
    let cfg = Config::by_name(&name).ok_or_else(|| anyhow!("cfg"))?;
    let dcfg = DConfig::by_name(&a.opt("--dcfg").unwrap_or(name.clone())).ok_or_else(|| anyhow!("dcfg"))?;
    let (batch, frames, iters) = (a.get("--batch", 16usize), a.get("--frames", 64usize), a.get("--iters", 10usize));
    let dev = device();
    let w = Weights::default();
    let gg = gan::gen_graph(&cfg, &dcfg, &w, batch, frames);
    let dp = gan::init_dparams(&dcfg, 5);
    let mut gp = Program::new(&dev, &gg.lg.b.g, &gg.outputs, &refs(&gg.lg.b.assemble_gd(&model::init_params(&cfg, 1), None, &dp, None, &[])));
    let gms = gp.time(&dev, iters);
    println!("G step: {} kernels, compile {:.0} ms, {gms:.1} ms", gp.kernels(), gp.compile_ms);
    drop(gp);
    let dg = gan::disc_graph(&cfg, &dcfg, &w, batch, frames);
    let mut dpr = Program::new(&dev, &dg.b.g, &dg.outputs, &refs(&dg.b.assemble_gd(&[], None, &dp, None, &[])));
    let dms = dpr.time(&dev, iters);
    println!("D step: {} kernels, compile {:.0} ms, {dms:.1} ms  (D {:.2}M params, {:?})", dpr.kernels(), dpr.compile_ms, dcfg.n_params() as f64 / 1e6, dcfg);
    #[cfg(feature = "metal")]
    if a.flag("--profile") {
        let exe = &dpr.exe.0;
        let mut ks: Vec<(f64, usize)> = (0..exe.program.kernels.len()).map(|i| (exe.time_kernels(&dev.0, i..i + 1, 2), i)).collect();
        ks.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
        if let Ok(dir) = std::env::var("RIBT_DUMP") {
            for (_, i) in ks.iter().take(4) {
                std::fs::write(format!("{dir}/k{i}.metal"), &exe.program.kernels[*i].source).ok();
            }
        }
        for (ms, i) in ks.iter().take(a.get("--top", 12usize)) {
            let k = &exe.program.kernels[*i];
            let ni: usize = format!("{:?}", k.node).chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap();
            let shape: Vec<usize> = dpr.graph.shape(k.node).iter().map(|d| d.n()).collect();
            let red: Option<Vec<usize>> = dpr.graph.comp_at(ni).map(|c| c.red.iter().map(|d| d.n()).collect());
            println!("  {ms:8.2} ms  kernel {i} node {ni} shape {shape:?} red {red:?} points {} groups {:?} threads {}", k.points, k.groups, k.threads);
        }
    }
    Ok(())
}

/// One G step and one D step on the GPU vs the f64 interpreter (check-size
/// cfg): every output except p′ (see `gpucheck`), worst relative error.
fn gancheck(a: &Args) -> Result<()> {
    use ribt::gan::{self, DConfig, Weights};
    let name = a.opt("--cfg").unwrap_or_else(|| "checkf0".into());
    let cfg = Config::by_name(&name).ok_or_else(|| anyhow!("cfg"))?;
    let dcfg = DConfig::by_name(&a.opt("--dcfg").unwrap_or(name.clone())).ok_or_else(|| anyhow!("dcfg"))?;
    let (batch, frames) = (2, a.get("--frames", 16usize));
    let n = frames * cfg.hop;
    let mut rng = Rng::new(4);
    let mut params = model::init_params(&cfg, 1);
    for p in params.iter_mut() {
        for v in p.iter_mut() {
            *v += (0.05 * rng.normal()) as f32;
        }
    }
    let dparams = gan::init_dparams(&dcfg, 2);
    let ds = Dataset { clips: vec![(PathBuf::new(), (0..4 * n).map(|i| ((i as f64 * 0.037).sin() * 0.5 + 0.05 * rng.normal()) as f32).collect())] };
    let (xp, x) = ds.batch(&mut rng, batch, n, cfg.mel_pad());
    let f0 = src_of(&cfg, &xp, batch, frames);
    let fm: f32 = a.get("--fake-mix", 0.0f32);
    let fake: Vec<f32> = x.iter().enumerate().map(|(i, v)| fm * v + (1.0 - fm) * 0.5 * ((i as f32 * 0.011).sin()) + 0.01 * rng.normal() as f32).collect();
    let dev = device();
    let w = Weights::default();
    let compare = |name: &str, g: &tang_compile::tensor::TensorGraph, outs: &[tang_compile::tensor::TensorId], inputs: &[Vec<f32>], skip: std::ops::Range<usize>, lim: usize| -> f64 {
        let want = g.eval(&inputs.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect::<Vec<_>>(), outs);
        let mut prog = Program::new(&dev, g, outs, &refs(inputs));
        prog.run(&dev);
        let mut worst = 0.0f64;
        let mut rest = 0.0f64;
        for (i, wv) in want.iter().enumerate() {
            if skip.contains(&i) {
                continue;
            }
            let got = prog.read(&dev, i);
            let scale = wv.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-6);
            let e = got.iter().zip(wv).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max) / scale;
            if e > 1e-3 && i < lim {
                eprintln!("  {name} output {i}: rel err {e:.2e}");
            }
            if i < lim {
                worst = worst.max(e);
            } else {
                rest = rest.max(e);
            }
        }
        println!("{name}: {} kernels, loss gpu {:.6} vs f64 {:.6}, worst rel err over {} outputs {worst:.2e} (losses, p', m', v'; materialized intermediates' grads, not used by the update: {rest:.2e})", prog.kernels(), prog.read(&dev, 0)[0], want[0][0], want.len());
        worst
    };
    // --ext: the §16 fine-tune graph (item 0 vocodes an external mel, item 1 its frontend mel)
    let ext = a.flag("--ext") || cfg.hid > 0;
    let gg = gan::gen_graph_ext(&cfg, &dcfg, &w, batch, frames, ext);
    let mm = cfg.in_ch() * frames;
    let mel_ext: Vec<f32> = (0..batch * mm).map(|i| (-4.0 + 2.0 * ((i as f64 * 0.013).sin()) + 0.3 * rng.normal()) as f32).collect();
    let mel_sel: Vec<f32> = (0..batch * mm).map(|i| if i < mm { 1.0 } else { 0.0 }).collect();
    let f0hz = dsp::yin_f0_batch(&xp, batch, cfg.sr, cfg.n_fft, cfg.hop, frames);
    let np_ = ribt::glottis::n_params(&cfg);
    let phys: Vec<f32> = (0..batch * np_ * frames).map(|i| {
        let c = (i / frames) % np_;
        (match c { 1 => 7.5, 2 => -2.0, 3 => -1.0, _ => 0.0 } + 0.2 * rng.normal()) as f32
    }).collect();
    let mut gin = gg.lg.b.assemble_gd(&params, None, &dparams, None, &[("audio_pad", &xp), ("audio", &x), ("src", &f0), ("phys", &phys), ("f0hz", &f0hz), ("mel_ext", &mel_ext), ("mel_sel", &mel_sel)]);
    gin[gg.lg.b.input_index(gg.hyper)] = vec![1e-3, 1.0 - model::BETA1 as f32, 1.0 - model::BETA2 as f32, 0.01, 1.0, 1.0];
    let np = cfg.params().len();
    let e1 = if a.flag("--d-only") { 0.0 } else { compare("G step", &gg.lg.b.g, &gg.outputs, &gin, gan::GEN_HEAD..gan::GEN_HEAD + np, gan::GEN_HEAD + 3 * np) };
    let dg = gan::disc_graph(&cfg, &dcfg, &w, batch, frames);
    let mut din = dg.b.assemble_gd(&[], None, &dparams, None, &[("audio", &x), ("fake", &fake)]);
    din[dg.b.input_index(dg.hyper)] = vec![1e-3, 1.0 - model::BETA1 as f32, 1.0 - model::BETA2 as f32, 0.01, a.get("--clip", 1.0f32), 0.0];
    let nd = dcfg.params().len();
    // as for G: losses, p', m', v' decide; grad-norm partials and kept per-channel grads are reported apart
    let e2 = compare("D step", &dg.b.g, &dg.outputs, &din, gan::DISC_HEAD..gan::DISC_HEAD + nd, gan::DISC_HEAD + 3 * nd);
    if e1.max(e2) > 1e-3 {
        return Err(anyhow!("GPU disagrees with the interpreter"));
    }
    Ok(())
}

/// Which G loss term carries the gradient spikes: per-term, per-param
/// squared grad norms on real batches (`--init` checkpoint, fresh D unless --dckpt).
fn gdiag(a: &Args) -> Result<()> {
    use ribt::gan::{self, DConfig, Weights};
    let cfg = Config::by_name(&a.opt("--cfg").unwrap_or_else(|| "v1f0".into())).unwrap();
    let dcfg = DConfig::by_name(&a.opt("--dcfg").unwrap_or_else(|| "v12".into())).unwrap();
    let (batch, frames) = (a.get("--batch", 8usize), 64usize);
    let (ds, _) = load_split(a, &cfg)?;
    let params = ckpt::load_specs(Path::new(&a.opt("--init").ok_or_else(|| anyhow!("--init"))?), &cfg.params(), true)?.params;
    let dparams = match a.opt("--dckpt") { Some(p) => ckpt::load_specs(Path::new(&p), &dcfg.params(), false)?.params, None => gan::init_dparams(&dcfg, 5) };
    let dev = device();
    let dg = gan::diag_graph(&cfg, &dcfg, &Weights::default(), batch, frames);
    let b = &dg.lg.b;
    let mut prog = Program::new(&dev, &b.g, &dg.outputs, &refs(&b.assemble_gd(&params, None, &dparams, None, &[])));
    let (kp, ka, kf) = (b.input_index(dg.lg.audio_pad), b.input_index(dg.lg.audio), dg.lg.f0.map(|t| b.input_index(t)));
    let gg = gan::gen_graph(&cfg, &dcfg, &Weights::default(), batch, frames);
    let gb = &gg.lg.b;
    let mut gin = gb.assemble_gd(&params, None, &dparams, None, &[]);
    gin[gb.input_index(gg.hyper)] = vec![0.0, 1.0 - model::BETA1 as f32, 1.0 - model::BETA2 as f32, 0.01, 1.0, 1.0];
    let mut gprog = Program::new(&dev, &gb.g, &gg.outputs, &refs(&gin));
    let (gkp, gka, gkf) = (gb.input_index(gg.lg.audio_pad), gb.input_index(gg.lg.audio), gg.lg.f0.map(|t| gb.input_index(t)));
    let specs = cfg.params();
    let np = specs.len();
    let mut rng = Rng::new(a.get("--seed", 1u64));
    for it in 0..a.get("--iters", 20usize) {
        let (xp, x) = ds.batch(&mut rng, batch, frames * cfg.hop, cfg.mel_pad());
        prog.write_input(&dev, kp, &xp);
        prog.write_input(&dev, ka, &x);
        if let Some(kf) = kf {
            prog.write_input(&dev, kf, &src_of(&cfg, &xp, batch, frames));
        }
        prog.run(&dev);
        let f0 = if cfg.f0 { src_of(&cfg, &xp, batch, frames) } else { vec![] };
        gprog.write_input(&dev, gkp, &xp);
        gprog.write_input(&dev, gka, &x);
        if let Some(k) = gkf {
            gprog.write_input(&dev, k, &f0);
        }
        gprog.run(&dev);
        let head: Vec<f32> = (0..gan::GEN_HEAD).map(|i| gprog.read(&dev, i)[0]).collect();
        let npar = cfg.params().len();
        let base = gan::GEN_HEAD + 3 * npar + 1;
        let parts: Vec<f32> = (0..npar.div_ceil(8)).map(|i| gprog.read(&dev, base + i)[0]).collect();
        let mut line = format!("batch {it:>2}: gen head {head:?}\n  partials(8 params each) {parts:?}\n         ");
        for (ti, name) in dg.names.iter().enumerate() {
            let sq: Vec<f64> = (0..np).map(|i| prog.read(&dev, ti * np + i)[0] as f64).collect();
            let tot = sq.iter().sum::<f64>().sqrt();
            let (imax, _) = sq.iter().enumerate().fold((0, -1.0), |m, (i, &v)| if v > m.1 { (i, v) } else { m });
            line += &format!("  {name} {tot:.3e} ({})", specs[imax].name);
        }
        println!("{line}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// §11.5 physics range: analysis-by-synthesis labels, knobs, inference

/// One compiled fit program (analysis-by-synthesis), reused across batches.
struct Fitter {
    prog: Program,
    k: [usize; 6], // th, m, v, mel, f0, hyper
    batch: usize,
    frames: usize,
}

impl Fitter {
    fn new(dev: &Device, cfg: &Config, batch: usize, frames: usize) -> Self {
        let fg = ribt::glottis::fit_graph(cfg, batch, frames);
        let inputs = fg.b.assemble(&[], None, &[]);
        let prog = Program::new(dev, &fg.b.g, &fg.outputs, &refs(&inputs));
        let b = &fg.b;
        Self { k: [b.input_index(fg.th), b.input_index(fg.m), b.input_index(fg.v), b.input_index(fg.mel), b.input_index(fg.f0), b.input_index(fg.hyper)], prog, batch, frames }
    }
    /// Fits θ `[B, P, T]` to `mel` `[B, M, T]` given `f0` `[B, T]`; returns (θ, mel RMSE).
    fn fit(&mut self, dev: &Device, cfg: &Config, mel: &[f32], f0: &[f32], iters: usize, lr: f64) -> (Vec<f32>, f32) {
        let th = ribt::glottis::init_params(cfg, mel, self.batch, self.frames);
        let z = vec![0.0f32; th.len()];
        self.prog.write_input(dev, self.k[0], &th);
        self.prog.write_input(dev, self.k[1], &z);
        self.prog.write_input(dev, self.k[2], &z);
        self.prog.write_input(dev, self.k[3], mel);
        self.prog.write_input(dev, self.k[4], f0);
        let mut rmse = 0.0;
        for it in 0..iters {
            let t = (it + 1) as i32;
            // cosine decay
            let l = lr * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * it as f64 / iters as f64).cos()));
            self.prog.write_input(dev, self.k[5], &[l as f32, (1.0 - 0.9f64.powi(t)) as f32, (1.0 - 0.999f64.powi(t)) as f32, 0.0, 0.0, 0.0]);
            self.prog.run(dev);
            if it + 1 == iters {
                rmse = self.prog.read(dev, 1)[0];
            }
            self.prog.swap_input_output(self.k[0], 2);
            self.prog.swap_input_output(self.k[1], 3);
            self.prog.swap_input_output(self.k[2], 4);
        }
        (self.prog.read_input(dev, self.k[0]).unwrap(), rmse)
    }
}

/// Per-clip labels: θ `[P, T]` channel-major and F0 `[T]` (Hz).
struct Labels {
    np: usize,
    by_path: std::collections::HashMap<String, (usize, Vec<f32>, Vec<f32>)>,
}

fn save_labels(path: &Path, np: usize, rows: &[(String, usize, Vec<f32>, Vec<f32>)]) -> Result<()> {
    let mut b = Vec::new();
    b.extend(b"PHYS0001");
    b.extend((rows.len() as u32).to_le_bytes());
    b.extend((np as u32).to_le_bytes());
    for (p, t, th, f0) in rows {
        b.extend((p.len() as u32).to_le_bytes());
        b.extend(p.as_bytes());
        b.extend((*t as u32).to_le_bytes());
        for v in th.iter().chain(f0) {
            b.extend(v.to_le_bytes());
        }
    }
    std::fs::write(path, b)?;
    Ok(())
}

fn load_labels(path: &Path) -> Result<Labels> {
    let b = std::fs::read(path)?;
    if &b[..8] != b"PHYS0001" {
        return Err(anyhow!("bad labels file"));
    }
    let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
    let (n, np) = (u(8), u(12));
    let mut o = 16;
    let mut by_path = std::collections::HashMap::new();
    let f = |o: usize, n: usize| -> Vec<f32> { b[o..o + 4 * n].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
    for _ in 0..n {
        let pl = u(o);
        let p = String::from_utf8_lossy(&b[o + 4..o + 4 + pl]).into_owned();
        o += 4 + pl;
        let t = u(o);
        o += 4;
        let th = f(o, np * t);
        o += 4 * np * t;
        let f0 = f(o, t);
        o += 4 * t;
        by_path.insert(p, (t, th, f0));
    }
    Ok(Labels { np, by_path })
}

/// Log-mel of `[nb]` chunks of `frames` frames from padded clips (GPU).
struct ChunkMel {
    prog: Program,
    k: usize,
    nb: usize,
    frames: usize,
}

impl ChunkMel {
    fn new(dev: &Device, cfg: &Config, nb: usize, frames: usize) -> Self {
        let fe = model::frontend_graph(cfg, nb, frames);
        let inputs = fe.b.assemble(&[], None, &[]);
        let prog = Program::new(dev, &fe.b.g, &[fe.mel], &refs(&inputs));
        Self { k: fe.b.input_index(fe.audio), prog, nb, frames }
    }
}

/// `ribt phys-labels --data DIRS --heldout DIRS --out FILE [--iters 300] [--max-clips N]`
fn phys_labels(a: &Args) -> Result<()> {
    let cfg = Config::by_name("v1f0phys").unwrap();
    let out = PathBuf::from(a.opt("--out").ok_or_else(|| anyhow!("--out"))?);
    let iters: usize = a.get("--iters", 300);
    let lr: f64 = a.get("--fit-lr", 0.05);
    let (ts, _) = load_split(a, &cfg)?;
    let TrainSet::Corpus(c) = ts else { return Err(anyhow!("needs --heldout (corpus mode)")) };
    let (nb, ch) = (a.get("--nb", 128usize), 256usize);
    let dev = device();
    let mut cm = ChunkMel::new(&dev, &cfg, nb, ch);
    let mut fit = Fitter::new(&dev, &cfg, nb, ch);
    let np = ribt::glottis::n_params(&cfg);
    let (hop, pad, m) = (cfg.hop, cfg.mel_pad(), cfg.n_mels);
    let max_clips: usize = a.get("--max-clips", usize::MAX);
    // work items: (clip, chunk)
    let clips: Vec<&(PathBuf, Vec<i16>)> = c.clips.iter().take(max_clips).collect();
    let mut rows: Vec<(String, usize, Vec<f32>, Vec<f32>)> = Vec::new();
    let f0s: Vec<Vec<f32>>;
    let mut xps: Vec<Vec<f32>> = Vec::new();
    for (p, v) in &clips {
        let t = v.len() / hop;
        let x: Vec<f32> = v[..t * hop].iter().map(|&s| s as f32 / 32768.0).collect();
        rows.push((p.display().to_string(), t, vec![0.0; np * t], vec![]));
        xps.push(dsp::reflect_pad(&x, pad));
    }
    // F0 on the host, all clips in parallel
    {
        let nt = 20;
        let parts: Vec<Vec<Vec<f32>>> = std::thread::scope(|s| {
            let hs: Vec<_> = xps.chunks(xps.len().div_ceil(nt).max(1)).map(|cc| s.spawn(move || cc.iter().map(|xp| dsp::yin_f0(xp, cfg.sr, cfg.n_fft, cfg.hop, (xp.len() - 2 * pad) / hop)).collect::<Vec<_>>())).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        f0s = parts.concat();
    }
    let items: Vec<(usize, usize)> = rows.iter().enumerate().flat_map(|(i, r)| (0..r.1.div_ceil(ch)).map(move |k| (i, k))).collect();
    eprintln!("{} clips, {} chunks of {ch} frames, {iters} fit iterations per batch of {nb}", rows.len(), items.len());
    let t0 = Instant::now();
    let mut rm = Vec::new();
    for (bi, batch) in items.chunks(nb).enumerate() {
        let w = ch * hop + 2 * pad;
        let mut xin = vec![0.0f32; nb * w];
        let mut f0in = vec![0.0f32; nb * ch];
        for (j, &(ci, k)) in batch.iter().enumerate() {
            let xp = &xps[ci];
            let s0 = k * ch * hop;
            let n = (xp.len() - s0).min(w);
            xin[j * w..j * w + n].copy_from_slice(&xp[s0..s0 + n]);
            let t = rows[ci].1;
            for tt in 0..ch.min(t - k * ch) {
                f0in[j * ch + tt] = f0s[ci][k * ch + tt];
            }
        }
        cm.prog.write_input(&dev, cm.k, &xin);
        cm.prog.run(&dev);
        let mel = cm.prog.read(&dev, 0);
        let (th, r) = fit.fit(&dev, &cfg, &mel, &f0in, iters, lr);
        rm.push(r);
        for (j, &(ci, k)) in batch.iter().enumerate() {
            let t = rows[ci].1;
            for c in 0..np {
                for tt in 0..ch.min(t - k * ch) {
                    rows[ci].2[c * t + k * ch + tt] = th[(j * np + c) * ch + tt];
                }
            }
        }
        if bi % 10 == 0 {
            eprintln!("batch {bi}/{}: fit mel RMSE {r:.4} ({:.0} s)", items.len().div_ceil(nb), t0.elapsed().as_secs_f64());
        }
    }
    let _ = (cm.nb, cm.frames, m);
    for (r, f) in rows.iter_mut().zip(f0s) {
        r.3 = f;
    }
    save_labels(&out, np, &rows)?;
    eprintln!("wrote {} ({} clips), mean fit mel RMSE {:.4}, {:.0} s", out.display(), rows.len(), rm.iter().sum::<f32>() / rm.len().max(1) as f32, t0.elapsed().as_secs_f64());
    Ok(())
}

/// θ `[B, P, frames]` and F0 `[B, 1, frames]` for hop-aligned crops.
fn crop_labels(lab: &Labels, paths: &[String], meta: &[(usize, usize, f32)], hop: usize, frames: usize) -> (Vec<f32>, Vec<f32>) {
    use ribt::glottis::{P_ASP, P_CEP, P_TILT, PRIOR_ASP, PRIOR_TILT};
    let np = lab.np;
    let b = meta.len();
    let mut th = vec![0.0f32; b * np * frames];
    let mut f0 = vec![0.0f32; b * frames];
    for (j, &(ci, s0, gain)) in meta.iter().enumerate() {
        let (t, l, f) = &lab.by_path[&paths[ci]];
        let j0 = s0 / hop;
        for tt in 0..frames {
            let src = j0 + tt;
            for c in 0..np {
                th[(j * np + c) * frames + tt] = if src < *t {
                    l[c * t + src] + if c == P_CEP { gain.ln() } else { 0.0 }
                } else {
                    match c { P_TILT => PRIOR_TILT as f32, P_ASP => PRIOR_ASP as f32, P_CEP => -12.0, _ => 0.0 }
                };
            }
            f0[j * frames + tt] = if src < *t { f[src] } else { 0.0 };
        }
    }
    (th, f0)
}

/// `ribt rinfer --ckpt F --cfg v1f0{pinv,phys} IN.wav OUT.wav [--oq D] [--asp D] [--tilt D] [--iters 300]`
/// Range-operator inference: fits the physical params to the clip's mel,
/// applies knob deltas on voiced frames (logit OQ, ln aspiration, ln Fa),
/// and prints the exactness `max |mel(s) − target|`.
fn rinfer(a: &Args) -> Result<()> {
    use ribt::glottis::{P_ASP, P_OQ, P_TILT};
    let cfg = a.cfg()?;
    let pos = a.positional();
    let [inp, out] = &pos[..] else { return Err(anyhow!("rinfer IN OUT")) };
    let dev = device();
    let params = load_params(a, &cfg)?;
    let (x, _) = data::read_wav(Path::new(inp))?;
    let (mel, frames) = wav_to_mel(&dev, &cfg, &x);
    let xp = dsp::reflect_pad(&x[..frames * cfg.hop], cfg.mel_pad());
    let src = src_of(&cfg, &xp, 1, frames);
    let f0 = dsp::yin_f0(&xp, cfg.sr, cfg.n_fft, cfg.hop, frames);
    let np = ribt::glottis::n_params(&cfg);
    let (th0, rmse) = if cfg.range == 2 {
        let mut fit = Fitter::new(&dev, &cfg, 1, frames);
        fit.fit(&dev, &cfg, &mel, &f0, a.get("--iters", 300), a.get("--fit-lr", 0.05))
    } else {
        (vec![0.0; np * frames], 0.0)
    };
    let mut th = th0.clone();
    let (doq, dasp, dtilt): (f32, f32, f32) = (a.get("--oq", 0.0), a.get("--asp", 0.0), a.get("--tilt", 0.0));
    for t in 0..frames {
        if f0[t] as f64 >= dsp::F0_MIN {
            th[P_OQ * frames + t] += doq;
            th[P_ASP * frames + t] += dasp;
            th[P_TILT * frames + t] += dtilt;
        }
    }
    let inf = model::infer_graph(&cfg, 1, frames);
    let inputs = inf.b.assemble(&params, None, &[("mel", &mel), ("src", &src), ("phys", &th), ("phys0", &th0), ("f0hz", &f0)]);
    let mut outs = vec![inf.audio];
    outs.extend(inf.mel_err);
    let mut prog = Program::new(&dev, &inf.b.g, &outs, &refs(&inputs));
    prog.run(&dev);
    let y = prog.read(&dev, 0);
    let err = if outs.len() > 1 { prog.read(&dev, 1)[0] } else { f32::NAN };
    data::write_wav(Path::new(out), &y, cfg.sr as u32)?;
    if let Some(p) = a.opt("--dump-phys") {
        write_npy(Path::new(&p), &[np, frames], &th0)?;
    }
    println!("rinfer {inp}: frames {frames} fit_rmse {rmse:.4} mel_exact_err {err:.3e}");
    Ok(())
}

/// `ribt lilt ...`: the misaki port (the ribt paper). CPU only.
fn lilt_cmd(a: &Args) -> Result<()> {
    use ribt::lilt::{self, parity};
    // golden_eval.tsv (test+edge, committed) for parity/bench; golden.tsv
    // (all splits, regenerated by scripts/lilt_golden.py) for training
    let dflt = if a.0.get(1).is_some_and(|c| c == "train-tagger") { "golden.tsv" } else { "golden_eval.tsv" };
    let golden = a.opt("--golden").unwrap_or_else(|| format!("{}/data/lilt/{dflt}", env!("CARGO_MANIFEST_DIR")));
    match a.0.get(1).map(String::as_str) {
        Some("say") => {
            let text = a.0[2..].join(" ");
            let (ps, words) = lilt::phonemize(&text);
            println!("{ps}");
            for w in words {
                println!("  {:16} {:6} {:20} {:?}", w.text, w.prosody.pos, w.phonemes, (w.prosody.stressed, w.prosody.boundary, w.prosody.oov));
            }
        }
        Some("parity") => {
            let sents = parity::load(&golden)?;
            let split = a.opt("--split").unwrap_or_else(|| "test,edge".into());
            let splits: Vec<&str> = split.split(',').collect();
            let r = parity::evaluate(&sents, &splits);
            parity::print_report(&r, a.get("--show", 12));
            let sel: Vec<_> = sents.into_iter().filter(|s| splits.contains(&s.split.as_str())).collect();
            let (n, bad) = parity::stream_check(&sel, a.get("--stream", 500));
            println!("stream == batch      {}/{} sentences (char-by-char pushes)", n - bad, n);
        }
        Some("train-tagger") => {
            let sents = parity::load(&golden)?;
            let m = parity::train_tagger(&sents, a.get("--iters", 8));
            let out = a.opt("--out").unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/data/lilt/tagger.bin").into());
            let b = m.to_bytes();
            std::fs::write(&out, &b)?;
            println!("wrote {out}: {} features, {} tags, {} bytes", m.w.len(), m.tags.len(), b.len());
        }
        Some("bench") => {
            let sents = parity::load(&golden)?;
            let sel: Vec<_> = sents.into_iter().filter(|s| s.split != "train").collect();
            let t0 = Instant::now();
            lilt::phonemize("cold start.");
            println!("cold start (lexicon+tagger init) {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
            let s = parity::speed(&sel);
            println!("batch      {:.0} words/s (single thread)", s.words_per_s);
            println!("stream     {:.1} us per pushed word", s.us_per_push_word);
            println!("stream lag {:.3} words behind input on average; {:.1}% of words emitted on their own push", s.mean_lag_words, 100.0 * s.immediate_frac);
        }
        Some("fit-g2p") => {
            // docs §27: the OOV letter-context tree. Fitting wants a big GPU box.
            let (train, test) = lilt::oov::split();
            let min = a.get("--min", 1u32);
            let iters = a.get("--iters", 6usize);
            let half = a.get("--half", 4usize);
            let t0 = Instant::now();
            let m = lilt::oov::fit(&train, min, iters, half);
            println!("fit on {} words (min {min}, ±{half} letters, {iters} EM iters): {} nodes in {:.1} s", train.len(), m.len(), t0.elapsed().as_secs_f64());
            let s = lilt::oov::score(&m, &test);
            println!("held-out {} words: exact {:.2}%  exact-no-stress {:.2}%  PER {:.2}%  {:.2} us/word", s.words, 100.0 * s.exact as f64 / s.words as f64, 100.0 * s.exact_nostress as f64 / s.words as f64, 100.0 * s.per, s.us_per_word);
            if let Some(out) = a.opt("--out") {
                // the shipped model is refitted on every word; held-out numbers above are from the split fit
                let all: Vec<_> = train.into_iter().chain(test).collect();
                let m = lilt::oov::fit(&all, min, iters, half);
                let b = m.to_bytes();
                std::fs::write(&out, &b)?;
                println!("wrote {out}: {} nodes, {} bytes (fit on all {} words)", m.len(), b.len(), all.len());
            }
        }
        Some("g2p-eval") => {
            let (_, test) = lilt::oov::split();
            let m = lilt::oov::model().ok_or_else(|| anyhow::anyhow!("no embedded g2p model"))?;
            let s = lilt::oov::score(m, &test);
            println!("embedded model ({} nodes; fitted on all words, so this is a train score) on the held-out list: exact {:.2}%  PER {:.2}%  {:.2} us/word", m.len(), 100.0 * s.exact as f64 / s.words as f64, 100.0 * s.per, s.us_per_word);
        }
        Some("oov") => {
            // a TSV whose first column is a word or phrase: per-entry phonemes and the deletion count
            let path = a.0.get(2).ok_or_else(|| anyhow::anyhow!("ribt lilt oov FILE.tsv"))?;
            let col = a.get("--col", 1usize).max(1) - 1;
            let (mut n, mut dropped) = (0, 0);
            for l in std::fs::read_to_string(path)?.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
                let w = l.split('\t').nth(col).unwrap_or("");
                let (_, words) = lilt::phonemize(w);
                let lost: Vec<&str> = words.iter().filter(|x| x.prosody.oov || (x.phonemes.is_empty() && x.text.chars().any(char::is_alphanumeric))).map(|x| x.text.as_str()).collect();
                n += 1;
                dropped += !lost.is_empty() as usize;
                println!("{w}\t{}\t{}", lilt::join(&words), if lost.is_empty() { "ok".into() } else { format!("DROPPED {lost:?}") });
            }
            println!("# {dropped}/{n} entries lose a word");
        }
        _ => eprintln!("usage: ribt lilt say|parity|train-tagger|bench|fit-g2p|g2p-eval|oov"),
    }
    Ok(())
}
