//! `ribt-align`: the V2 aligner (the ribt paper). GPU work should
//! run under `~/.cache/gpu-lock.sh` on shared machines.
//!
//!   ribt-align train --phon A.phon.tsv[,B…] --root LIBRITTS_R --run DIR
//!                    [--held test-clean.phon.tsv] [--cfg base] [--steps N] [--total N]
//!                    [--batch 12] [--n 192] [--t 1024] [--lr 1e-3] [--bin-start 0.4] [--w-bin 1]
//!                    (touch RUN/STOP to save + exit)
//!   ribt-align durs  --ckpt F --phon A.phon.tsv[,…] --root LIBRITTS_R --out durs.tsv
//!                    [--attn-dir DIR --attn-n 8]   (soft attention + path npys for plots)
//!   ribt-align check [--cfg check]                  one train step, GPU vs f64 interpreter
//!   ribt-align scancheck [--cfg base] [--batch 12] [--n 192] [--t 1024] [--blank none] [--w-bin 0.5] [--iters 20]
//!                    in-graph scan losses (GPU, f32) vs the host (f64), then the
//!                    one-program step vs the two-program step: gradients and time
//!
//! `*.phon.tsv` come from `scripts/phonemize_libritts.py` (Kokoro ids); the
//! split name is the file name before `.phon.tsv`, and utterance
//! `S_C_…` lives at `ROOT/split/S/C/S_C_….wav`.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use ribt::align::{self, AConfig};
use ribt::ckpt::{self, Checkpoint};
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
    /// `--blank none` = blank-free forward-sum; default RAD-TTS's blank logit −1
    fn blank(&self) -> Option<f64> {
        match self.opt("--blank").as_deref() {
            Some("none") => None,
            Some(v) => v.parse().ok(),
            None => Some(-1.0),
        }
    }
    fn cfg(&self) -> Result<AConfig> {
        let n = self.opt("--cfg").unwrap_or_else(|| "base".into());
        let mut c = AConfig::by_name(&n).ok_or_else(|| anyhow!("unknown cfg {n}"))?;
        c.temp = self.get("--temp", c.temp);
        Ok(c)
    }
}

fn main() -> Result<()> {
    let a = Args(std::env::args().skip(1).collect());
    match a.0.first().map(String::as_str) {
        Some("train") => train(&a),
        Some("durs") => durs(&a),
        Some("check") => check(&a),
        Some("scancheck") => scancheck(&a),
        Some("melcheck") => melcheck(&a),
        _ => {
            eprintln!("usage: ribt-align train|durs|check|scancheck (see src/bin/ribt-align.rs)");
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
    peak * (0.05 + 0.95 * 0.5 * (1.0 + (std::f64::consts::PI * p).cos()))
}

// ---------------------------------------------------------------------------
// data

struct Utt {
    id: String,
    toks: Vec<u16>,
    audio: Vec<i16>,
}

impl Utt {
    fn frames(&self, hop: usize) -> usize {
        self.audio.len() / hop
    }
}

fn wav_path(root: &Path, split: &str, id: &str) -> PathBuf {
    let mut it = id.split('_');
    let (s, c) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
    root.join(split).join(s).join(c).join(format!("{id}.wav"))
}

/// Rows of the given phon tsvs with their audio (parallel readers), in file
/// order. `keep(n_tokens, frames)` filters.
fn load(phon: &str, root: &Path, hop: usize, keep: &(dyn Fn(usize, usize) -> bool + Sync)) -> Result<Vec<Utt>> {
    let mut rows: Vec<(String, PathBuf, Vec<u16>)> = Vec::new();
    for f in phon.split(',').filter(|s| !s.is_empty()) {
        let split = Path::new(f).file_name().unwrap().to_string_lossy().trim_end_matches(".phon.tsv").to_string();
        for line in std::fs::read_to_string(f)?.lines() {
            let mut c = line.split('\t');
            let (Some(id), Some(ids)) = (c.next(), c.next()) else { continue };
            let toks: Vec<u16> = ids.split(' ').filter_map(|x| x.parse().ok()).collect();
            rows.push((id.to_string(), wav_path(root, &split, id), toks));
        }
    }
    let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(12);
    let chunk = rows.len().div_ceil(nt).max(1);
    let parts: Vec<Result<Vec<Utt>>> = std::thread::scope(|s| {
        let hs: Vec<_> = rows
            .chunks(chunk)
            .map(|c| {
                s.spawn(move || {
                    let mut out = Vec::new();
                    for (id, p, toks) in c {
                        let Ok(mut r) = hound::WavReader::open(p) else {
                            eprintln!("missing/unreadable wav, skipped: {}", p.display());
                            continue;
                        };
                        let spec = r.spec();
                        if spec.sample_rate != 24000 || spec.channels != 1 {
                            return Err(anyhow!("{}: {} Hz × {} ch", p.display(), spec.sample_rate, spec.channels));
                        }
                        let frames = r.duration() as usize / hop;
                        if !keep(toks.len(), frames) {
                            continue;
                        }
                        let audio: Vec<i16> = match (spec.sample_format, spec.bits_per_sample) {
                            (hound::SampleFormat::Int, 16) => r.samples::<i16>().collect::<Result<_, _>>()?,
                            _ => r.samples::<f32>().map(|x| x.map(|v| (v * 32767.0).clamp(-32768.0, 32767.0) as i16)).collect::<Result<_, _>>()?,
                        };
                        out.push(Utt { id: id.clone(), toks: toks.clone(), audio });
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

/// Programs for one padded shape: frontend (audio → mel), fwd (→ z).
struct Shape {
    batch: usize,
    n: usize,
    t: usize,
    fe: Program,
    fwd: Program,
    k_audio: usize,
    k_mel: usize,
    k_oh: usize,
    k_p: Vec<usize>,
}

impl Shape {
    fn new(dev: &Device, ac: &AConfig, params: &[Vec<f32>], batch: usize, n: usize, t: usize) -> Self {
        let vc = Config::v1();
        assert_eq!(vc.n_mels, ac.n_mels);
        let fe = model::frontend_graph(&vc, batch, t);
        let fin = fe.b.assemble(&[], None, &[]);
        let fe_prog = Program::new(dev, &fe.b.g, &[fe.mel], &refs(&fin));
        let ag = align::align_graph(ac, batch, n, t);
        let inp = align::assemble(&ag.b, &ac.params(), params, None, &[]);
        let fwd = Program::new(dev, &ag.b.g, &[ag.z], &refs(&inp));
        Shape {
            batch,
            n,
            t,
            k_audio: fe.b.input_index(fe.audio),
            k_mel: ag.b.input_index(ag.mel),
            k_oh: ag.b.input_index(ag.onehot),
            k_p: ag.params.iter().map(|&p| ag.b.input_index(p)).collect(),
            fe: fe_prog,
            fwd,
        }
    }

    /// Padded audio `[B, t·hop + 2·pad]` (each clip reflect-padded at its own
    /// length, then zeros) → log-mel `[B, n_mels, t]`; frame `i < frames`
    /// equals the clip's own frontend output.
    fn mel(&mut self, dev: &Device, utts: &[&Utt]) -> Vec<f32> {
        let vc = Config::v1();
        let (hop, pad) = (vc.hop, vc.mel_pad());
        let row = self.t * hop + 2 * pad;
        let mut xp = vec![0.0f32; self.batch * row];
        for (b, u) in utts.iter().enumerate() {
            let fr = u.frames(hop).min(self.t);
            let x: Vec<f32> = u.audio[..fr * hop].iter().map(|&v| v as f32 / 32768.0).collect();
            let p = dsp::reflect_pad(&x, pad);
            xp[b * row..b * row + p.len()].copy_from_slice(&p);
        }
        self.fe.write_input(dev, self.k_audio, &xp);
        self.fe.run(dev);
        self.fe.read(dev, 0)
    }

    fn z(&mut self, dev: &Device, ac: &AConfig, mel: &[f32], utts: &[&Utt]) -> Vec<f32> {
        let rows: Vec<&[u16]> = utts.iter().map(|u| u.toks.as_slice()).collect();
        let oh = align::onehot(&rows, ac.vocab, self.n);
        self.fwd.write_input(dev, self.k_mel, mel);
        self.fwd.write_input(dev, self.k_oh, &oh);
        self.fwd.run(dev);
        self.fwd.read(dev, 0)
    }

    fn set_params(&mut self, dev: &Device, params: &[Vec<f32>]) {
        for (k, p) in self.k_p.iter().zip(params) {
            self.fwd.write_input(dev, *k, p);
        }
    }
}

fn lens(utts: &[&Utt], hop: usize) -> Vec<(usize, usize)> {
    utts.iter().map(|u| (u.toks.len(), u.frames(hop))).collect()
}

// ---------------------------------------------------------------------------
// train

fn train(a: &Args) -> Result<()> {
    let ac = a.cfg()?;
    let specs = ac.params();
    let run = PathBuf::from(a.opt("--run").ok_or_else(|| anyhow!("--run"))?);
    std::fs::create_dir_all(&run)?;
    let root = PathBuf::from(a.opt("--root").ok_or_else(|| anyhow!("--root"))?);
    let phon = a.opt("--phon").ok_or_else(|| anyhow!("--phon"))?;
    let steps: usize = a.get("--steps", 1000);
    let total: usize = a.get("--total", 10000);
    let batch: usize = a.get("--batch", 12);
    let (nmax, tmax): (usize, usize) = (a.get("--n", 192), a.get("--t", 1024));
    let peak: f64 = a.get("--lr", 1e-3);
    let clip: f64 = a.get("--clip", 1.0);
    let bin_start = (a.get("--bin-start", 0.4) * total as f64) as usize;
    let w_bin_max: f64 = a.get("--w-bin", 1.0);
    let scaling: f64 = a.get("--prior-scaling", 1.0);
    let blank = a.blank();
    let log_every: usize = a.get("--log-every", 50);
    let ckpt_every: usize = a.get("--ckpt-every", 500);
    let eval_every: usize = a.get("--eval-every", 500);
    let hop = Config::v1().hop;
    let keep = |n: usize, t: usize| n <= nmax && t <= tmax && t >= n;

    let t0 = Instant::now();
    let data = load(&phon, &root, hop, &keep)?;
    if data.is_empty() {
        return Err(anyhow!("no utterances fit n ≤ {nmax}, t ≤ {tmax}"));
    }
    let secs: f64 = data.iter().map(|u| u.audio.len()).sum::<usize>() as f64 / 24000.0;
    eprintln!("train: {} utts ({:.2} h) with n ≤ {nmax}, t ≤ {tmax}, loaded in {:.1} s", data.len(), secs / 3600.0, t0.elapsed().as_secs_f64());
    let held: Vec<Utt> = match a.opt("--held") {
        Some(h) => {
            let mut v = load(&h, &root, hop, &keep)?;
            let step = (v.len() / (4 * batch)).max(1);
            v = v.into_iter().step_by(step).take(4 * batch).collect();
            v
        }
        None => Vec::new(),
    };
    eprintln!("held-out: {} utts", held.len());

    let ck_path = run.join("last.safetensors");
    let ck = if ck_path.exists() { ckpt::load_specs(&ck_path, &specs, false)? } else { Checkpoint { params: align::init_params(&ac, 0), adam: None, step: 0 } };
    eprintln!("cfg {ac:?}\nparams {}  start step {}  bin from step {bin_start}  blank {blank:?}", ac.n_params(), ck.step);

    let dev = Device::new();
    eprintln!("backend {} ({})", ribt::runtime::BACKEND, dev.name());
    let mut sh = Shape::new(&dev, &ac, &ck.params, batch, nmax, tmax);
    let tg = align::train_graph(&ac, batch, nmax, tmax);
    let adam = ck.adam.as_ref().map(|(m, v)| (m.as_slice(), v.as_slice()));
    let inputs = align::assemble(&tg.ag.b, &specs, &ck.params, adam, &[]);
    let mut prog = Program::new(&dev, &tg.ag.b.g, &tg.outputs, &refs(&inputs));
    drop(inputs);
    eprintln!("programs: frontend {} + fwd {} + train {} kernels, train compile {:.0} ms", sh.fe.kernels(), sh.fwd.kernels(), prog.kernels(), prog.compile_ms);
    let b = &tg.ag.b;
    let np = specs.len();
    let k_mel = b.input_index(tg.ag.mel);
    let k_oh = b.input_index(tg.ag.onehot);
    let k_gz = b.input_index(tg.gz);
    let k_hyper = b.input_index(tg.hyper);
    let k_p: Vec<usize> = tg.ag.params.iter().map(|&t| b.input_index(t)).collect();
    let k_m: Vec<usize> = tg.m.iter().map(|&t| b.input_index(t)).collect();
    let k_v: Vec<usize> = tg.v.iter().map(|&t| b.input_index(t)).collect();
    let (o_p, o_m, o_v) = (1, 1 + np, 1 + 2 * np);

    let open_log = |name: &str, header: &str| -> Result<std::fs::File> {
        let p = run.join(name);
        let new = !p.exists();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p)?;
        if new {
            writeln!(f, "{header}")?;
        }
        Ok(f)
    };
    let mut log = open_log("log.tsv", "step\tloss_fs\tloss_bin\tw_bin\tsharp\tgrad_norm\tlr\tms")?;
    let mut elog = open_log("eval.tsv", "step\tloss_fs\tloss_bin\tsharp\twall_s")?;
    let read_params = |prog: &Program| -> Vec<Vec<f32>> { k_p.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect() };
    let save = |prog: &Program, step: usize| -> Result<()> {
        let params = read_params(prog);
        let m: Vec<Vec<f32>> = k_m.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect();
        let v: Vec<Vec<f32>> = k_v.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect();
        ckpt::save_specs(&ck_path, &specs, &Checkpoint { params, adam: Some((m, v)), step })
    };
    let do_eval = |sh: &mut Shape, step: usize, elog: &mut std::fs::File| -> Result<()> {
        if held.is_empty() {
            return Ok(());
        }
        let (mut lf, mut lb, mut s, mut k) = (0.0, 0.0, 0.0, 0.0);
        for c in held.chunks(batch).filter(|c| c.len() == batch) {
            let u: Vec<&Utt> = c.iter().collect();
            let mel = sh.mel(&dev, &u);
            let z = sh.z(&dev, &ac, &mel, &u);
            let (_, f, bl, sp, _) = align::batch_host(&z, nmax, tmax, &lens(&u, hop), 0.0, scaling, blank);
            lf += f;
            lb += bl;
            s += sp;
            k += 1.0;
        }
        writeln!(elog, "{step}\t{}\t{}\t{}\t{:.0}", lf / k, lb / k, s / k, t0.elapsed().as_secs_f64())?;
        eprintln!("eval step {step:>6}  held-out L_fs {:.4}  L_bin {:.4}  sharp {:.3}", lf / k, lb / k, s / k);
        Ok(())
    };
    if ck.step == 0 {
        do_eval(&mut sh, 0, &mut elog)?;
    }
    let start = ck.step;
    let mut step = ck.step;
    drop(ck);
    let mut rng = Rng::new(1000 + step as u64);
    let stop = run.join("STOP");
    let mut acc = [0.0f64; 5];
    let mut acc_n = 0usize;
    let mut skipped = 0;
    let mut times = Vec::new();
    while step < start + steps {
        let ts = Instant::now();
        let u: Vec<&Utt> = (0..batch).map(|_| &data[rng.below(data.len())]).collect();
        let ls = lens(&u, hop);
        let mel = sh.mel(&dev, &u);
        let z = sh.z(&dev, &ac, &mel, &u);
        let w_bin = if step >= bin_start { w_bin_max * ((step - bin_start) as f64 / 500.0).min(1.0) } else { 0.0 };
        let (gz, lf, lb, sp, _) = align::batch_host(&z, nmax, tmax, &ls, w_bin, scaling, blank);
        let lr = lr_at(step, total, peak, 200.min(total / 10).max(1));
        let t = (step + 1) as i32;
        let hyper = [lr as f32, (1.0 - 0.9f64.powi(t)) as f32, (1.0 - 0.999f64.powi(t)) as f32, 0.0, clip as f32, 0.0];
        let rows: Vec<&[u16]> = u.iter().map(|x| x.toks.as_slice()).collect();
        prog.write_input(&dev, k_mel, &mel);
        prog.write_input(&dev, k_oh, &align::onehot(&rows, ac.vocab, nmax));
        prog.write_input(&dev, k_gz, &gz);
        prog.write_input(&dev, k_hyper, &hyper);
        prog.run(&dev);
        let gn = prog.read(&dev, 0)[0];
        if !(gn.is_finite() && lf.is_finite()) {
            skipped += 1;
            eprintln!("step {}: non-finite (L_fs {lf}, |g| {gn}), skipped", step + 1);
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
        sh.set_params(&dev, &read_params(&prog));
        step += 1;
        let ms = ts.elapsed().as_secs_f64() * 1e3;
        times.push(ms);
        for (i, x) in [lf, lb, sp, gn as f64, ms].iter().enumerate() {
            acc[i] += x;
        }
        acc_n += 1;
        writeln!(log, "{step}\t{lf}\t{lb}\t{w_bin}\t{sp}\t{gn}\t{lr:.3e}\t{ms:.1}")?;
        if step % log_every == 0 {
            let k = acc_n as f64;
            eprintln!(
                "step {step:>6}  L_fs {:.4}  L_bin {:.4}  w_bin {w_bin:.2}  sharp {:.3}  |g| {:.3}  lr {lr:.2e}  {:.0} ms/step",
                acc[0] / k,
                acc[1] / k,
                acc[2] / k,
                acc[3] / k,
                acc[4] / k
            );
            acc = [0.0; 5];
            acc_n = 0;
        }
        let stopping = stop.exists();
        if step % ckpt_every == 0 || stopping {
            save(&prog, step)?;
            log.flush()?;
        }
        if eval_every > 0 && step % eval_every == 0 {
            do_eval(&mut sh, step, &mut elog)?;
        }
        if stopping {
            std::fs::remove_file(&stop).ok();
            eprintln!("STOP: saved step {step}");
            return Ok(());
        }
    }
    save(&prog, step)?;
    if eval_every > 0 && step % eval_every != 0 {
        do_eval(&mut sh, step, &mut elog)?;
    }
    let mut s: Vec<f64> = times.iter().skip(2).copied().collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !s.is_empty() {
        eprintln!("median {:.0} ms/step (frontend + fwd + host + train + param copy)", s[s.len() / 2]);
    }
    eprintln!("saved {}", ck_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// durations export

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

fn durs(a: &Args) -> Result<()> {
    let ac = a.cfg()?;
    let specs = ac.params();
    let ckp = PathBuf::from(a.opt("--ckpt").ok_or_else(|| anyhow!("--ckpt"))?);
    let params = ckpt::load_specs(&ckp, &specs, false)?.params;
    let root = PathBuf::from(a.opt("--root").ok_or_else(|| anyhow!("--root"))?);
    let phon = a.opt("--phon").ok_or_else(|| anyhow!("--phon"))?;
    let out = PathBuf::from(a.opt("--out").ok_or_else(|| anyhow!("--out"))?);
    let batch: usize = a.get("--batch", 12);
    let (nmax, tmax): (usize, usize) = (a.get("--n", 192), a.get("--t", 1024));
    let scaling: f64 = a.get("--prior-scaling", 1.0);
    let blank = a.blank();
    let attn_dir = a.opt("--attn-dir").map(PathBuf::from);
    let attn_n: usize = a.get("--attn-n", 8);
    let hop = Config::v1().hop;
    let t0 = Instant::now();
    let mut skipped = 0usize;
    let data = load(&phon, &root, hop, &|n, t| t >= n)?;
    eprintln!("{} utts loaded in {:.1} s", data.len(), t0.elapsed().as_secs_f64());
    let dev = Device::new();
    // bucket: the training shape at full batch, else a batch-1 shape rounded up
    let key = |u: &Utt| -> (usize, usize, usize) {
        let (n, t) = (u.toks.len(), u.frames(hop));
        if n <= nmax && t <= tmax {
            (batch, nmax, tmax)
        } else {
            (1, n.div_ceil(64) * 64, t.div_ceil(512) * 512)
        }
    };
    let mut groups: std::collections::BTreeMap<(usize, usize, usize), Vec<usize>> = Default::default();
    for (i, u) in data.iter().enumerate() {
        groups.entry(key(u)).or_default().push(i);
    }
    let mut res: Vec<Option<(Vec<u32>, f64, f64)>> = vec![None; data.len()];
    let mut shapes: HashMap<(usize, usize, usize), Shape> = HashMap::new();
    if let Some(d) = &attn_dir {
        std::fs::create_dir_all(d)?;
    }
    let mut attn_left = attn_n;
    for ((bs, n, t), idx) in &groups {
        let sh = shapes.entry((*bs, *n, *t)).or_insert_with(|| Shape::new(&dev, &ac, &params, *bs, *n, *t));
        for c in idx.chunks(*bs) {
            let mut u: Vec<&Utt> = c.iter().map(|&i| &data[i]).collect();
            let real = u.len();
            while u.len() < *bs {
                u.push(u[0]);
            }
            let mel = sh.mel(&dev, &u);
            let z = sh.z(&dev, &ac, &mel, &u);
            let ls = lens(&u[..real], hop);
            for (j, &(nn, tt)) in ls.iter().enumerate() {
                let prior = align::log_prior(nn, tt, scaling);
                let zf = |ni: usize, ti: usize| z[(j * sh.n + ni) * sh.t + ti] as f64;
                let o = align::utt_host(&zf, nn, tt, &prior, 0.0, blank);
                if let (Some(d), true) = (&attn_dir, attn_left > 0 && sh.batch > 1) {
                    let lp = align::log_attn(&zf, nn, tt, &prior);
                    let at: Vec<f32> = lp.iter().map(|v| v.exp() as f32).collect();
                    write_npy(&d.join(format!("{}.attn.npy", u[j].id)), &[tt, nn], &at)?;
                    let dz: Vec<f32> = o.dur.iter().map(|&v| v as f32).collect();
                    write_npy(&d.join(format!("{}.dur.npy", u[j].id)), &[nn], &dz)?;
                    attn_left -= 1;
                }
                res[c[j]] = Some((o.dur, o.loss_fs, o.sharp));
            }
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out)?);
    writeln!(f, "# utt\tframes\tn_tokens\tloss_fs\tsharp\tdurations (frames per token, Kokoro ids order)")?;
    for (u, r) in data.iter().zip(&res) {
        let Some((d, l, s)) = r else {
            skipped += 1;
            continue;
        };
        let ds = d.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
        writeln!(f, "{}\t{}\t{}\t{l:.5}\t{s:.4}\t{ds}", u.id, u.frames(hop), u.toks.len())?;
    }
    eprintln!("wrote {} ({} utts, {} shapes, {skipped} skipped) in {:.1} s", out.display(), data.len(), shapes.len(), t0.elapsed().as_secs_f64());
    Ok(())
}

// ---------------------------------------------------------------------------
// GPU vs f64 interpreter

fn check(a: &Args) -> Result<()> {
    let name = a.opt("--cfg").unwrap_or_else(|| "check".into());
    let ac = a.cfg()?;
    let specs = ac.params();
    let (nmax, tmax): (usize, usize) = (a.get("--n", 24), a.get("--t", 64));
    let bsz = 3;
    let ls = [(nmax, tmax), (nmax * 2 / 3, tmax * 3 / 4), (nmax / 3, tmax / 2)];
    let mut rng = Rng::new(4);
    let params: Vec<Vec<f32>> = align::init_params(&ac, 1).into_iter().map(|p| p.iter().map(|v| v + (0.05 * rng.normal()) as f32).collect()).collect();
    let toks: Vec<Vec<u16>> = ls.iter().map(|&(n, _)| (0..n).map(|_| rng.below(ac.vocab) as u16).collect()).collect();
    let rows: Vec<&[u16]> = toks.iter().map(|r| r.as_slice()).collect();
    let oh = align::onehot(&rows, ac.vocab, nmax);
    let mel: Vec<f32> = (0..bsz * ac.n_mels * tmax).map(|_| (-5.0 + 2.5 * rng.normal()) as f32).collect();
    let tg = align::train_graph(&ac, bsz, nmax, tmax);
    let base = align::assemble(&tg.ag.b, &specs, &params, None, &[("onehot", &oh), ("mel", &mel)]);
    // interpreter z → host G
    let b64: Vec<Vec<f64>> = base.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
    let z64 = tg.ag.b.g.eval(&b64, &[tg.ag.z])[0].clone();
    let z32: Vec<f32> = z64.iter().map(|&v| v as f32).collect();
    let (gz, lf, lb, _, _) = align::batch_host(&z32, nmax, tmax, &ls, 0.5, 1.0, a.blank());
    let hyper = [1e-3f32, 0.1, 0.001, 0.0, 1.0, 0.0];
    let mut inp = base.clone();
    inp[tg.ag.b.input_index(tg.gz)] = gz;
    inp[tg.ag.b.input_index(tg.hyper)] = hyper.to_vec();
    let i64s: Vec<Vec<f64>> = inp.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
    let mut outs = vec![tg.ag.z];
    outs.extend(&tg.outputs);
    let want = tg.ag.b.g.eval(&i64s, &outs);
    let dev = Device::new();
    let mut prog = Program::new(&dev, &tg.ag.b.g, &outs, &refs(&inp));
    prog.run(&dev);
    let rel = |got: &[f32], want: &[f64]| -> f64 {
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
        got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((*g as f64 - w).abs())) / scale
    };
    let np = specs.len();
    let z_err = rel(&prog.read(&dev, 0), &want[0]);
    let gn_err = rel(&prog.read(&dev, 1), &want[1]);
    let mut p_err: f64 = 0.0;
    let mut m_err: f64 = 0.0;
    for i in 0..np {
        // compare the update (p' − p) and m' (= (1−β₁)·clipped grad)
        let got: Vec<f32> = prog.read(&dev, 2 + i).iter().zip(&params[i]).map(|(a, b)| a - b).collect();
        let w: Vec<f64> = want[2 + i].iter().zip(&params[i]).map(|(a, b)| a - *b as f64).collect();
        p_err = p_err.max(rel(&got, &w));
        m_err = m_err.max(rel(&prog.read(&dev, 2 + np + i), &want[2 + np + i]));
    }
    eprintln!(
        "{} vs f64 interpreter, cfg {name} (B{bsz} N{nmax} T{tmax}; host L_fs {lf:.4} L_bin {lb:.4}):\n  z max rel {z_err:.2e}\n  grad norm rel {gn_err:.2e} ({} vs {})\n  m' (grads) worst rel {m_err:.2e}\n  Δp worst rel {p_err:.2e}",
        ribt::runtime::BACKEND,
        prog.read(&dev, 1)[0],
        want[1][0]
    );
    if z_err > 1e-4 || m_err > 1e-3 {
        return Err(anyhow!("GPU/interpreter mismatch"));
    }
    Ok(())
}

/// The scan path ([`align::scan_loss`]) on the GPU against the host path on
/// synthetic batches of the training shape with per-item lengths:
/// 1. losses, MAS durations and `∂L/∂z` on logits fed as data, for the
///    network's own logits at init and for sharp ones (a trained aligner's);
/// 2. the gradients of the one-program step ([`align::scan_train_graph`])
///    against the two-program step's ([`align::train_graph`] fed host `G`);
/// 3. wall time per step of both, as `train` runs them.
fn scancheck(a: &Args) -> Result<()> {
    let ac = a.cfg()?;
    let specs = ac.params();
    let np = specs.len();
    let (bsz, nmax, tmax): (usize, usize, usize) = (a.get("--batch", 12), a.get("--n", 192), a.get("--t", 1024));
    let (iters, w_bin, blank) = (a.get("--iters", 20usize), a.get("--w-bin", 0.5f64), a.blank());
    let mut rng = Rng::new(a.get("--seed", 4));
    // item 0 fills the padded shape; the rest: 1/3..1 of the frames, 5..8 frames per token
    let ls: Vec<(usize, usize)> = (0..bsz)
        .map(|b| {
            if b == 0 {
                return (nmax, tmax);
            }
            let t = tmax / 3 + rng.below(tmax - tmax / 3 + 1);
            let n = ((t as f64 / (5.0 + 3.0 * rng.uniform())) as usize).clamp(1, nmax.min(t));
            (n, t)
        })
        .collect();
    let params = align::init_params(&ac, 1);
    let toks: Vec<Vec<u16>> = ls.iter().map(|&(n, _)| (0..n).map(|_| rng.below(ac.vocab) as u16).collect()).collect();
    let rows: Vec<&[u16]> = toks.iter().map(|r| r.as_slice()).collect();
    let oh = align::onehot(&rows, ac.vocab, nmax);
    let mel: Vec<f32> = (0..bsz * ac.n_mels * tmax).map(|_| (ac.mel_mu + ac.mel_sd * rng.normal()) as f32).collect();
    let (prior, end, fw) = align::scan_data(&ls, nmax, tmax, 1.0);
    let dev = Device::new();
    eprintln!("{} ({}), B{bsz} N{nmax} T{tmax}, blank {blank:?}, w_bin {w_bin}, lens {ls:?}", ribt::runtime::BACKEND, dev.name());

    // 1. the losses on given logits
    let ag = align::align_graph(&ac, bsz, nmax, tmax);
    let mut fwd = Program::new(&dev, &ag.b.g, &[ag.z], &refs(&align::assemble(&ag.b, &specs, &params, None, &[("onehot", &oh), ("mel", &mel)])));
    fwd.run(&dev);
    let z_init = fwd.read(&dev, 0);
    // sharp logits: a random monotonic path per item, −4 per token of distance, unit noise
    let mut z_sharp = vec![0.0f32; bsz * nmax * tmax];
    for (b, &(n, t)) in ls.iter().enumerate() {
        let mut cuts: Vec<usize> = (0..n - 1).map(|_| 1 + rng.below(t - 1)).collect();
        cuts.sort();
        for ni in 0..nmax {
            for ti in 0..tmax {
                let on = cuts.iter().filter(|&&c| c <= ti).count() as f64;
                z_sharp[(b * nmax + ni) * tmax + ti] = (-4.0 * (ni as f64 - on).abs() + rng.normal()) as f32;
            }
        }
    }
    let mut zb = model::Built::new_pub(&Config::tiny());
    let z = zb.input(&[bsz, nmax, tmax], model::Role::Data("z"));
    let sl = align::scan_loss(&mut zb, z, blank);
    let gz = zb.g.backward(sl.total, &[z]).unwrap()[0];
    let wb = [w_bin as f32];
    let zin = align::assemble(&zb, &[], &[], None, &[("z", &z_init), ("prior", &prior), ("end", &end), ("fw", &fw), ("w_bin", &wb)]);
    let mut zprog = Program::new(&dev, &zb.g, &[sl.loss_fs, sl.loss_bin, sl.sharp, sl.dur, gz], &refs(&zin));
    eprintln!("loss-only program: {} kernels, compile {:.0} ms", zprog.kernels(), zprog.compile_ms);
    let k_z = zb.input_index(z);
    let (mut ok, mut paths_agree) = (true, true);
    for (name, zv) in [("init", &z_init), ("sharp", &z_sharp)] {
        zprog.write_input(&dev, k_z, zv);
        zprog.run(&dev);
        let (g, lf, lb, sp, durs) = align::batch_host(zv, nmax, tmax, &ls, w_bin, 1.0, blank);
        let (glf, glb, gsp, gdur, gg) = (zprog.read(&dev, 0)[0] as f64, zprog.read(&dev, 1)[0] as f64, zprog.read(&dev, 2)[0] as f64, zprog.read(&dev, 3), zprog.read(&dev, 4));
        let (mut tok_off, mut frames_off, mut toks_n) = (0usize, 0.0f64, 0usize);
        for (b, d) in durs.iter().enumerate() {
            for (ni, &want) in d.iter().enumerate() {
                let got = gdur[b * nmax + ni] as f64;
                toks_n += 1;
                if got != want as f64 {
                    tok_off += 1;
                    frames_off += (got - want as f64).abs();
                }
            }
        }
        let scale = g.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
        let max_abs = gg.iter().zip(&g).fold(0.0f64, |m, (a, h)| m.max((*a as f64 - *h as f64).abs()));
        let (num, den) = gg.iter().zip(&g).fold((0.0f64, 0.0f64), |(n, d), (a, h)| (n + (*a as f64 - *h as f64).powi(2), d + (*h as f64).powi(2)));
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
        eprintln!(
            "z {name}: L_fs {glf:.6} vs host {lf:.6} (rel {:.1e}), L_bin {glb:.6} vs {lb:.6} (rel {:.1e}), sharp {gsp:.4} vs {sp:.4}\n  durations: {tok_off}/{toks_n} tokens differ ({frames_off} frames moved)\n  dL/dz: rel l2 {:.2e}, max abs {max_abs:.2e} / max |G| {scale:.2e} = {:.2e}, non-finite {}",
            rel(glf, lf),
            rel(glb, lb),
            (num / den).sqrt(),
            max_abs / scale,
            gg.iter().filter(|v| !v.is_finite()).count()
        );
        // a moved MAS frame moves w_bin/(B·t) of gradient: hold dL/dz to the host's only where the paths agree
        ok &= rel(glf, lf) < 1e-4 && gg.iter().all(|v| v.is_finite()) && (tok_off > 0 || (num / den).sqrt() < 1e-3);
        paths_agree &= tok_off == 0;
    }
    drop(zprog);

    // 2. one step both ways from the same state; m' = (1−β₁)·clipped grad
    let hyper = [1e-3f32, 0.1, 0.001, 0.0, 1.0, 0.0];
    let tg = align::train_graph(&ac, bsz, nmax, tmax);
    let mut inp = align::assemble(&tg.ag.b, &specs, &params, None, &[("onehot", &oh), ("mel", &mel)]);
    inp[tg.ag.b.input_index(tg.hyper)] = hyper.to_vec();
    let mut two = Program::new(&dev, &tg.ag.b.g, &tg.outputs, &refs(&inp));
    let st = align::scan_train_graph(&ac, bsz, nmax, tmax, blank);
    let mut inp = align::assemble(&st.ag.b, &specs, &params, None, &[("onehot", &oh), ("mel", &mel), ("prior", &prior), ("end", &end), ("fw", &fw), ("w_bin", &wb)]);
    inp[st.ag.b.input_index(st.hyper)] = hyper.to_vec();
    let mut one = Program::new(&dev, &st.ag.b.g, &st.outputs, &refs(&inp));
    drop(inp);
    eprintln!("two-program step: fwd {} + train {} kernels (compile {:.0} ms); one-program step: {} kernels (compile {:.0} ms)", fwd.kernels(), two.kernels(), two.compile_ms, one.kernels(), one.compile_ms);
    let k_gz = tg.ag.b.input_index(tg.gz);
    let (gzv, lf, lb, _, _) = align::batch_host(&z_init, nmax, tmax, &ls, w_bin, 1.0, blank);
    two.write_input(&dev, k_gz, &gzv);
    two.run(&dev);
    one.run(&dev);
    let (mut num, mut den, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..np {
        let (a, b) = (one.read(&dev, 1 + np + i), two.read(&dev, 1 + np + i));
        let sc = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-20) as f64;
        for (x, y) in a.iter().zip(&b) {
            num += (*x as f64 - *y as f64).powi(2);
            den += (*y as f64).powi(2);
            worst = worst.max((*x as f64 - *y as f64).abs() / sc);
        }
    }
    let (gn1, gn2) = (one.read(&dev, 0)[0], two.read(&dev, 0)[0]);
    eprintln!(
        "step at init: grad norm {gn1} (one) vs {gn2} (two); grads rel l2 {:.2e}, worst per-tensor max-abs/scale {worst:.2e}; L_fs {} vs host {lf}, L_bin {} vs {lb}",
        (num / den).sqrt(),
        one.read(&dev, 1 + 3 * np)[0],
        one.read(&dev, 2 + 3 * np)[0]
    );
    ok &= !paths_agree || w_bin == 0.0 || (num / den).sqrt() < 1e-3;
    ok &= w_bin != 0.0 || (num / den).sqrt() < 1e-3;

    // 3. wall time per step, as `train` runs each (no frontend, no optimizer swap)
    let (k_mel2, k_oh2) = (tg.ag.b.input_index(tg.ag.mel), tg.ag.b.input_index(tg.ag.onehot));
    let (k_melf, k_ohf) = (ag.b.input_index(ag.mel), ag.b.input_index(ag.onehot));
    let k_p2: Vec<usize> = tg.ag.params.iter().map(|&t| tg.ag.b.input_index(t)).collect();
    let k_pf: Vec<usize> = ag.params.iter().map(|&t| ag.b.input_index(t)).collect();
    let b1 = &st.ag.b;
    let (k_mel1, k_oh1) = (b1.input_index(st.ag.mel), b1.input_index(st.ag.onehot));
    let k_d: Vec<usize> = [st.sl.prior, st.sl.end, st.sl.fw, st.sl.w_bin].iter().map(|&t| b1.input_index(t)).collect();
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (mut t_two, mut t_host, mut t_one) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..iters + 2 {
        let ts = Instant::now();
        fwd.write_input(&dev, k_melf, &mel);
        fwd.write_input(&dev, k_ohf, &align::onehot(&rows, ac.vocab, nmax));
        fwd.run(&dev);
        let z = fwd.read(&dev, 0);
        let th = Instant::now();
        let (gzv, ..) = align::batch_host(&z, nmax, tmax, &ls, w_bin, 1.0, blank);
        t_host.push(th.elapsed().as_secs_f64() * 1e3);
        two.write_input(&dev, k_mel2, &mel);
        two.write_input(&dev, k_oh2, &align::onehot(&rows, ac.vocab, nmax));
        two.write_input(&dev, k_gz, &gzv);
        two.run(&dev);
        let _ = two.read(&dev, 0)[0];
        // the param copy into `fwd` (train swaps p ↔ p' first; the copy is the same size)
        for (kf, k2) in k_pf.iter().zip(&k_p2) {
            fwd.write_input(&dev, *kf, &two.read_input(&dev, *k2).unwrap());
        }
        t_two.push(ts.elapsed().as_secs_f64() * 1e3);

        let ts = Instant::now();
        let (prior, end, fw) = align::scan_data(&ls, nmax, tmax, 1.0);
        one.write_input(&dev, k_mel1, &mel);
        one.write_input(&dev, k_oh1, &align::onehot(&rows, ac.vocab, nmax));
        for (k, v) in k_d.iter().zip([&prior[..], &end[..], &fw[..], &wb[..]]) {
            one.write_input(&dev, *k, v);
        }
        one.run(&dev);
        let _ = (one.read(&dev, 0)[0], one.read(&dev, 1 + 3 * np)[0], one.read(&dev, 2 + 3 * np)[0], one.read(&dev, 3 + 3 * np)[0]);
        t_one.push(ts.elapsed().as_secs_f64() * 1e3);
    }
    let (t_two, t_host, t_one) = (med(t_two.split_off(2)), med(t_host.split_off(2)), med(t_one.split_off(2)));
    let (k_fwd, k_two, k_one) = (fwd.time(&dev, iters), two.time(&dev, iters), one.time(&dev, iters));
    eprintln!(
        "step time, median of {iters}: two-program {t_two:.1} ms (host forward-sum/MAS {t_host:.1} ms of it; kernels only: fwd {k_fwd:.1} + train {k_two:.1} ms), one-program {t_one:.1} ms (kernels only {k_one:.1} ms)"
    );
    if !ok {
        return Err(anyhow!("scan path disagrees with the host"));
    }
    Ok(())
}

/// The padded-batch frontend vs the f64 reference log-mel of each clip.
fn melcheck(a: &Args) -> Result<()> {
    let ac = a.cfg()?;
    let root = PathBuf::from(a.opt("--root").ok_or_else(|| anyhow!("--root"))?);
    let phon = a.opt("--phon").ok_or_else(|| anyhow!("--phon"))?;
    let vc = Config::v1();
    let data = load(&phon, &root, vc.hop, &|n, t| t >= n && t <= 300)?;
    let u: Vec<&Utt> = data.iter().take(2).collect();
    let dev = Device::new();
    let mut sh = Shape::new(&dev, &ac, &align::init_params(&ac, 0), 2, 192, 320);
    let mel = sh.mel(&dev, &u);
    for (b, x) in u.iter().enumerate() {
        let fr = x.frames(vc.hop);
        let xf: Vec<f32> = x.audio[..fr * vc.hop].iter().map(|&v| v as f32 / 32768.0).collect();
        let xp: Vec<f64> = dsp::reflect_pad(&xf, vc.mel_pad()).iter().map(|&v| v as f64).collect();
        let want = dsp::log_mel_ref(&xp, vc.sr, vc.n_fft, vc.hop, vc.n_mels);
        let wf = want.len() / vc.n_mels;
        let mut err: f64 = 0.0;
        let (mut s, mut s2) = (0.0, 0.0);
        for m in 0..vc.n_mels {
            for t in 0..fr.min(wf) {
                let g = mel[(b * vc.n_mels + m) * 320 + t] as f64;
                err = err.max((g - want[m * wf + t]).abs());
                s += g;
                s2 += g * g;
            }
        }
        let k = (vc.n_mels * fr) as f64;
        eprintln!("{}: frames {fr} (ref {wf}), max |err| {err:.2e}, mean {:.3}, sd {:.3}, pad frame value {}", x.id, s / k, (s2 / k - (s / k).powi(2)).sqrt(), mel[(b * vc.n_mels + 50) * 320 + 319]);
    }
    Ok(())
}
