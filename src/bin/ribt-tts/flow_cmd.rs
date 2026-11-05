//! `ribt-tts flow-train | flow-eval` and the flow path of `synth`
//! (the ribt paper): an OT-CFM mel decoder on top of the frozen
//! rung-1 model, and its shortcut-distilled 1–2 step student.
//!
//!   ribt-tts flow-train --ac-run V2A [--ac-ckpt step…] --run DIR --phon … --durs … --root R [--held H]
//!                       [--batch 32] [--t 384] [--steps N] [--total N] [--lr 2e-4] [--warmup 1000]
//!                       [--pred-pe P] [--init FLOW.safetensors]  rung 2b: P = per-item prob. of predicted pitch/energy cond.
//!                       [--chunk C --look L]                  block-causal attention (streaming option)
//!                       [--distill TEACHER.safetensors --pairs 4096 --teacher-nfe 16 --pair-temp 0.5]   student phase
//!                       [--seed 0]   offsets the pair/training RNGs
//!   ribt-tts flow-eval  --ac-run V2A --flow T.safetensors [--student S.safetensors] --held H --durs D --root R
//!                       --voc VOC --out-dir DIR [--n-utts 64] [--nfe 32,16,8] [--temp 1.0] [--seed 0]
//!
//! The flow normalizes mels as `(mel − m) / s` (global scalars, `flow_norm.txt`
//! beside the checkpoint). Teacher time features carry `d = 0`; the student
//! sees `d = 1/nfe` (shortcut models 2410.12557) and regresses the teacher's
//! ODE: `x₀ → x₁ᵀ` (d = 1), `x₀ → x½ᵀ` and `x½ᵀ → x₁ᵀ` (d = ½).

use super::*;
use ribt::flow::{self, FConfig, FDims};

/// Silence in the frontend's log-mel (ln 1e-5), used to pad vocoder input.
pub const MEL_FLOOR: f32 = -11.512925;
const SIGMA_MIN: f64 = 1e-4;

#[derive(Clone, Copy, Debug)]
pub struct FNorm {
    pub m: f64,
    pub s: f64,
}

impl FNorm {
    pub fn load(ckpt: &Path) -> Result<Self> {
        let p = ckpt.parent().unwrap_or(Path::new(".")).join("flow_norm.txt");
        let v: Vec<f64> = std::fs::read_to_string(&p)?.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        Ok(Self { m: v[0], s: v[1] })
    }
    fn save(&self, dir: &Path) -> Result<()> {
        std::fs::write(dir.join("flow_norm.txt"), format!("{} {}\n", self.m, self.s))?;
        Ok(())
    }
    /// `(x − m)/s` on valid frames, 0 elsewhere; `x [B, C, T]`, `fmask [B, T]`.
    pub fn norm(&self, x: &[f32], fmask: &[f32], b: usize, t: usize) -> Vec<f32> {
        let c = x.len() / (b * t);
        let mut out = vec![0.0f32; x.len()];
        for bi in 0..b {
            for ci in 0..c {
                for ti in 0..t {
                    let i = (bi * c + ci) * t + ti;
                    if fmask[bi * t + ti] > 0.0 {
                        out[i] = ((x[i] as f64 - self.m) / self.s) as f32;
                    }
                }
            }
        }
        out
    }
    pub fn denorm(&self, z: &[f32]) -> Vec<f32> {
        z.iter().map(|&v| (v as f64 * self.s + self.m) as f32).collect()
    }
}

/// One velocity evaluation per run; conditioning written once per sample.
pub struct Vel {
    prog: Program,
    map: HashMap<&'static str, usize>,
    pub d: FDims,
    fc: FConfig,
}

impl Vel {
    pub fn new(dev: &Device, fc: &FConfig, params: &[Vec<f32>], d: FDims) -> Self {
        let fg = flow::flow_graph(fc, d, false);
        let inp = flow::assemble(&fg.b, &fc.params(), params, None, &[]);
        let prog = Program::new(dev, &fg.b.g, &[fg.v], &refs(&inp));
        Self { prog, map: input_map(&fg.b), d, fc: fc.clone() }
    }
    pub fn set_params(&mut self, dev: &Device, params: &[Vec<f32>]) {
        let fg = flow::flow_graph(&self.fc, self.d, false);
        for (&pid, p) in fg.params.iter().zip(params) {
            self.prog.write_input(dev, fg.b.input_index(pid), p);
        }
    }
    pub fn set_cond(&mut self, dev: &Device, mu: &[f32], hc: &[f32], fmask: &[f32], abias: &[f32]) {
        for (k, v) in [("mu", mu), ("hc", hc), ("fmask", fmask), ("abias", abias)] {
            self.prog.write_input(dev, self.map[k], v);
        }
    }
    pub fn step(&mut self, dev: &Device, xt: &[f32], t: f64, dstep: f64) -> Vec<f32> {
        let tf = flow::time_feats(&self.fc, t, dstep);
        let temb: Vec<f32> = (0..self.d.batch).flat_map(|_| tf.iter().copied()).collect();
        self.prog.write_input(dev, self.map["xt"], xt);
        self.prog.write_input(dev, self.map["temb"], &temb);
        self.prog.run(dev);
        self.prog.read(dev, 0)
    }
    /// Euler from `x0` over `nfe` steps (`shortcut`: d = 1/nfe). Returns x₁
    /// and (for even `nfe`) the state at t = ½.
    pub fn sample(&mut self, dev: &Device, x0: &[f32], nfe: usize, shortcut: bool) -> (Vec<f32>, Option<Vec<f32>>) {
        let mut x = x0.to_vec();
        let mut half = None;
        let h = 1.0 / nfe as f64;
        for k in 0..nfe {
            if nfe % 2 == 0 && k == nfe / 2 {
                half = Some(x.clone());
            }
            let v = self.step(dev, &x, k as f64 * h, if shortcut { h } else { 0.0 });
            for (xi, vi) in x.iter_mut().zip(&v) {
                *xi += (h as f32) * vi;
            }
        }
        (x, half)
    }
}

pub fn lens_of(fmask: &[f32], b: usize, t: usize) -> Vec<usize> {
    (0..b).map(|i| fmask[i * t..(i + 1) * t].iter().filter(|&&m| m > 0.0).count()).collect()
}

/// Noise `[B, C, T]` (scaled by `temp`) on valid frames only.
pub fn noise(rng: &mut Rng, fmask: &[f32], b: usize, c: usize, t: usize, temp: f64) -> Vec<f32> {
    let mut x = vec![0.0f32; b * c * t];
    for bi in 0..b {
        for ci in 0..c {
            for ti in 0..t {
                if fmask[bi * t + ti] > 0.0 {
                    x[(bi * c + ci) * t + ti] = (temp * rng.normal()) as f32;
                }
            }
        }
    }
    x
}

fn chunk_of(a: &Args) -> Option<(usize, usize)> {
    a.opt("--chunk").and_then(|c| c.parse().ok()).map(|c| (c, a.get("--look", 1usize)))
}

/// Rung-1 conditioning of a host batch: (μ log-mel, hidden, F0 channels, ln(d+1)).
/// `gt_pe`: teacher-forced token pitch/energy, else the predictors'.
pub fn cond(dev: &Device, tts: &mut Tts, hb: &HostBatch, refmel: &[f32], gt_pe: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    cond_mix(dev, tts, hb, refmel, &vec![!gt_pe; hb.get("rw").len() / REF_T])
}

/// As `cond`, choosing per item: `pred[i]` = the predictors' token pitch/energy
/// (what inference sees), else teacher-forced (rung 2b, §17).
pub fn cond_mix(dev: &Device, tts: &mut Tts, hb: &HostBatch, refmel: &[f32], pred: &[bool]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let [h, ld, pp, ep, spk] = tts.encode(dev, hb.get("onehot"), hb.get("tmask"), refmel, hb.get("rw"));
    let (mut p, mut e) = (hb.get("p_tok").to_vec(), hb.get("e_tok").to_vec());
    let n = p.len() / pred.len();
    for (bi, &use_pred) in pred.iter().enumerate() {
        if use_pred {
            p[bi * n..(bi + 1) * n].copy_from_slice(&pp[bi * n..(bi + 1) * n]);
            e[bi * n..(bi + 1) * n].copy_from_slice(&ep[bi * n..(bi + 1) * n]);
        }
    }
    let (mu, f0, hid) = tts.decode(dev, &h, &p, &e, hb.get("tmask"), hb.get("align"), hb.get("fmask"), &spk);
    (mu, hid, f0, ld)
}

// ---------------------------------------------------------------------------
// train (teacher CFM, or shortcut distillation)

struct Pair {
    x0: Vec<f32>,
    xh: Vec<f32>,
    x1: Vec<f32>,
    mu: Vec<f32>,
    hc: Vec<f32>,
    len: usize,
}

/// Item `i` of a `[B, C, T]` batch.
fn item(x: &[f32], i: usize, per: usize) -> Vec<f32> {
    x[i * per..(i + 1) * per].to_vec()
}

pub fn flow_train(a: &Args) -> Result<()> {
    // --seed offsets the pair-generation and training RNGs (default 0 = unchanged); the held eval RNG stays fixed.
    let tseed: u64 = a.get("--seed", 0);
    let tc = a.cfg()?;
    let fc = FConfig::by_name(&a.opt("--fcfg").unwrap_or_else(|| "base".into())).ok_or_else(|| anyhow!("fcfg"))?;
    let fspecs = fc.params();
    let ac_run = PathBuf::from(a.need("--ac-run")?);
    let (ac_params, nm, ac_step) = load_run(&ac_run, &tc, a.opt("--ac-ckpt"))?;
    let run = PathBuf::from(a.need("--run")?);
    std::fs::create_dir_all(&run)?;
    let root = PathBuf::from(a.need("--root")?);
    let steps: usize = a.get("--steps", 1000);
    let total: usize = a.get("--total", 50000);
    let batch: usize = a.get("--batch", 32);
    let (nmax, umax): (usize, usize) = (a.get("--n", 192), a.get("--max-frames", 1024));
    let tmax: usize = a.get("--t", 384);
    let peak: f64 = a.get("--lr", 2e-4);
    let warmup: usize = a.get("--warmup", 1000);
    let clip: f64 = a.get("--clip", 1.0);
    let log_every: usize = a.get("--log-every", 100);
    let ckpt_every: usize = a.get("--ckpt-every", 1000);
    let eval_every: usize = a.get("--eval-every", 2000);
    let keep_every: usize = a.get("--keep-every", 10000);
    let chunk = chunk_of(a);
    let distill = a.opt("--distill").map(PathBuf::from);
    // rung 2b (§17): per-item probability of conditioning on the predictors'
    // pitch/energy instead of the teacher-forced values (scheduled sampling)
    let pred_pe: f64 = a.get("--pred-pe", 0.0);
    let init = a.opt("--init").map(PathBuf::from);
    let keep = |n: usize, t: usize| n <= nmax && t <= umax && t >= 16;
    let d = Dims { batch, n: nmax, t: tmax, r: REF_T };
    let fd = FDims { batch, t: tmax };
    let (c, t) = (tc.n_mels, tmax);

    let t0 = Instant::now();
    let durs = read_durs(&a.need("--durs")?)?;
    let data = load(&a.need("--phon")?, &durs, &root, &keep)?;
    let spk = by_speaker(&data);
    eprintln!("flow-train: {} utts, {} speakers, loaded in {:.1} s; rung-1 step {ac_step}", data.len(), spk.len(), t0.elapsed().as_secs_f64());
    let held: Vec<Utt> = match a.opt("--held") {
        Some(h) => {
            let v = load(&h, &durs, &root, &keep)?;
            let st = (v.len() / (2 * batch)).max(1);
            v.into_iter().step_by(st).take(2 * batch).collect()
        }
        None => Vec::new(),
    };
    let held_spk = by_speaker(&held);
    let dev = Device::new();
    let mut fe = Frontend::new(&dev, batch, tmax + 2 * CTX);
    let mut fe_ref = Frontend::new(&dev, batch, REF_T);
    let mut tts = Tts::new(&dev, &tc, &ac_params, d);
    let mut rng = Rng::new(4242 + tseed);

    // flow normalization: from the teacher's run, or measured on 8 batches
    let fnm = match &distill {
        Some(tp) => FNorm::load(tp)?,
        None if run.join("flow_norm.txt").exists() => FNorm::load(&run.join("x"))?,
        None if init.is_some() => FNorm::load(init.as_ref().unwrap())?,
        None => {
            let (mut s1, mut s2, mut n) = (0.0f64, 0.0f64, 0.0f64);
            for _ in 0..8 {
                let us: Vec<&Utt> = (0..batch).map(|_| &data[rng.below(data.len())]).collect();
                let (hb, mel, _) = prep_crops(&dev, &tc, d, &nm, &mut rng, true, &us, &data, &spk, &mut fe, &mut fe_ref);
                let fm = hb.get("fmask");
                for bi in 0..batch {
                    for ci in 0..c {
                        for ti in 0..t {
                            if fm[bi * t + ti] > 0.0 {
                                let v = mel[(bi * c + ci) * t + ti] as f64;
                                s1 += v;
                                s2 += v * v;
                                n += 1.0;
                            }
                        }
                    }
                }
            }
            let m = s1 / n;
            FNorm { m, s: (s2 / n - m * m).sqrt() }
        }
    };
    fnm.save(&run)?;
    eprintln!("flow norm {fnm:?}; fcfg {fc:?}\nflow params {} ({:.2}M); chunk {chunk:?}; pred-pe {pred_pe}; init {init:?}", fc.n_params(), fc.n_params() as f64 / 1e6);

    let ck_path = run.join("last.safetensors");
    let ck = if ck_path.exists() {
        ckpt::load_specs(&ck_path, &fspecs, false)?
    } else if let Some(tp) = &distill {
        let mut k = ckpt::load_specs(tp, &fspecs, false)?;
        k.adam = None;
        k.step = 0;
        k
    } else if let Some(ip) = &init {
        // warm start (params only) from another flow run
        let mut k = ckpt::load_specs(ip, &fspecs, false)?;
        k.adam = None;
        k.step = 0;
        k
    } else {
        Checkpoint { params: flow::init_params(&fc, 0), adam: None, step: 0 }
    };

    // distillation pairs from the teacher's ODE (GT pitch/energy conditioning,
    // as the teacher was trained)
    let mut pairs: Vec<Pair> = Vec::new();
    if let Some(tp) = &distill {
        let n_pairs: usize = a.get("--pairs", 4096);
        let tnfe: usize = a.get("--teacher-nfe", 16);
        // noise scale of the teacher trajectories; the student is then run
        // at the same temperature (x₀ = τ·z)
        let ptemp: f64 = a.get("--pair-temp", 1.0);
        let tparams = ckpt::load_specs(tp, &fspecs, false)?.params;
        let mut vel = Vel::new(&dev, &fc, &tparams, fd);
        let tg = Instant::now();
        let mut prng = Rng::new(9 + tseed);
        while pairs.len() < n_pairs {
            let us: Vec<&Utt> = (0..batch).map(|_| &data[prng.below(data.len())]).collect();
            let (hb, _mel, refmel) = prep_crops(&dev, &tc, d, &nm, &mut prng, true, &us, &data, &spk, &mut fe, &mut fe_ref);
            let fm = hb.get("fmask").to_vec();
            let lens = lens_of(&fm, batch, t);
            let pred: Vec<bool> = (0..batch).map(|_| prng.uniform() < pred_pe).collect();
            let (mu, hid, _, _) = cond_mix(&dev, &mut tts, &hb, &refmel, &pred);
            let mun = fnm.norm(&mu, &fm, batch, t);
            vel.set_cond(&dev, &mun, &hid, &fm, &flow::attn_bias(&lens, t, chunk));
            let x0 = noise(&mut prng, &fm, batch, c, t, ptemp);
            let (x1, xh) = vel.sample(&dev, &x0, tnfe, false);
            let xh = xh.unwrap();
            for bi in 0..batch {
                let (pc, ph) = (c * t, tc.dd * t);
                pairs.push(Pair { x0: item(&x0, bi, pc), xh: item(&xh, bi, pc), x1: item(&x1, bi, pc), mu: item(&mun, bi, pc), hc: item(&hid, bi, ph), len: lens[bi] });
            }
            if pairs.len() % (batch * 16) == 0 {
                eprintln!("pairs {} / {n_pairs} ({:.0} s)", pairs.len(), tg.elapsed().as_secs_f64());
            }
        }
        eprintln!("{} teacher pairs ({tnfe} Euler steps, temp {ptemp}) in {:.0} s", pairs.len(), tg.elapsed().as_secs_f64());
    }

    let tt = flow::train_graph(&fc, fd);
    let adam = ck.adam.as_ref().map(|(m, v)| (m.as_slice(), v.as_slice()));
    let inputs = flow::assemble(&tt.fg.b, &fspecs, &ck.params, adam, &[]);
    let tcomp = Instant::now();
    let mut prog = Program::new(&dev, &tt.fg.b.g, &tt.outputs, &refs(&inputs));
    drop(inputs);
    eprintln!("flow train program: {} kernels, compile {:.1} s", prog.kernels(), tcomp.elapsed().as_secs_f64());
    // held-out CFM loss (forward only)
    let eg = flow::flow_graph(&fc, fd, true);
    let einp = flow::assemble(&eg.b, &fspecs, &ck.params, None, &[]);
    let mut eprog = Program::new(&dev, &eg.b.g, &[eg.loss.unwrap()], &refs(&einp));
    drop(einp);
    let ek_p: Vec<usize> = eg.params.iter().map(|&p| eg.b.input_index(p)).collect();
    let emap = input_map(&eg.b);
    let b = &tt.fg.b;
    let np = fspecs.len();
    let dmap = input_map(b);
    let k_hyper = b.input_index(tt.hyper);
    let k_p: Vec<usize> = tt.fg.params.iter().map(|&x| b.input_index(x)).collect();
    let k_m: Vec<usize> = tt.m.iter().map(|&x| b.input_index(x)).collect();
    let k_v: Vec<usize> = tt.v.iter().map(|&x| b.input_index(x)).collect();
    let (o_p, o_m, o_v) = (2, 2 + np, 2 + 2 * np);
    let open_log = |name: &str, header: &str| -> Result<std::fs::File> {
        let p = run.join(name);
        let new = !p.exists();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p)?;
        if new {
            writeln!(f, "{header}")?;
        }
        Ok(f)
    };
    let mut log = open_log("log.tsv", "step\tloss\tgrad_norm\tlr\tms")?;
    let mut elog = open_log("eval.tsv", "step\tcfm_t0.1\tcfm_t0.5\tcfm_t0.9\tcfm_mean\twall_s")?;
    let save = |prog: &Program, step: usize, path: &Path| -> Result<()> {
        let rd = |ks: &[usize]| -> Vec<Vec<f32>> { ks.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect() };
        ckpt::save_specs(path, &fspecs, &Checkpoint { params: rd(&k_p), adam: Some((rd(&k_m), rd(&k_v))), step })
    };

    // one flow-matching batch: (xt, ut, temb, mu, hc, fmask, abias, fw)
    type FB = [Vec<f32>; 8];
    let cfm_batch = |rng: &mut Rng, us: &[&Utt], pool: &[Utt], pspk: &HashMap<String, Vec<usize>>, fe: &mut Frontend, fe_ref: &mut Frontend, tts: &mut Tts, tfix: Option<f64>| -> FB {
        let (hb, mel, refmel) = prep_crops(&dev, &tc, d, &nm, rng, true, us, pool, pspk, fe, fe_ref);
        let fm = hb.get("fmask").to_vec();
        let lens = lens_of(&fm, batch, t);
        // eval (tfix) keeps teacher forcing so the held-out loss stays comparable
        let pred: Vec<bool> = (0..batch).map(|_| tfix.is_none() && rng.uniform() < pred_pe).collect();
        let (mu, hid, _, _) = cond_mix(&dev, tts, &hb, &refmel, &pred);
        let mun = fnm.norm(&mu, &fm, batch, t);
        let z1 = fnm.norm(&mel, &fm, batch, t);
        let x0 = noise(rng, &fm, batch, c, t, 1.0);
        let mut xt = vec![0.0f32; batch * c * t];
        let mut ut = vec![0.0f32; batch * c * t];
        let mut temb = Vec::with_capacity(batch * fc.tf);
        for bi in 0..batch {
            // stratified t over the batch (lower-variance loss)
            let tb = tfix.unwrap_or_else(|| (bi as f64 + rng.uniform()) / batch as f64);
            temb.extend(flow::time_feats(&fc, tb, 0.0));
            for i in bi * c * t..(bi + 1) * c * t {
                xt[i] = ((1.0 - (1.0 - SIGMA_MIN) * tb) * x0[i] as f64 + tb * z1[i] as f64) as f32;
                ut[i] = (z1[i] as f64 - (1.0 - SIGMA_MIN) * x0[i] as f64) as f32;
            }
        }
        let nfr: f32 = fm.iter().sum::<f32>().max(1.0);
        let fw = fm.iter().map(|m| m / nfr).collect();
        [xt, ut, temb, mun, hid, fm.clone(), flow::attn_bias(&lens, t, chunk), fw]
    };
    // one distillation batch from the pair pool
    let distill_batch = |rng: &mut Rng| -> FB {
        let (pc, ph) = (c * t, tc.dd * t);
        let mut fb: FB = Default::default();
        let mut lens = Vec::new();
        for _ in 0..batch {
            let p = &pairs[rng.below(pairs.len())];
            // 0: x0 → x1 (d 1); 1: x0 → x½ (d ½); 2: x½ → x1 (d ½)
            let k = rng.below(3);
            let (xs, ts, ds, tgt): (&[f32], f64, f64, Vec<f32>) = match k {
                0 => (&p.x0, 0.0, 1.0, p.x1.iter().zip(&p.x0).map(|(a, b)| a - b).collect()),
                1 => (&p.x0, 0.0, 0.5, p.xh.iter().zip(&p.x0).map(|(a, b)| 2.0 * (a - b)).collect()),
                _ => (&p.xh, 0.5, 0.5, p.x1.iter().zip(&p.xh).map(|(a, b)| 2.0 * (a - b)).collect()),
            };
            fb[0].extend_from_slice(xs);
            fb[1].extend(tgt);
            fb[2].extend(flow::time_feats(&fc, ts, ds));
            fb[3].extend_from_slice(&p.mu);
            fb[4].extend_from_slice(&p.hc);
            fb[5].extend((0..t).map(|i| if i < p.len { 1.0 } else { 0.0 }));
            lens.push(p.len);
        }
        let _ = (pc, ph);
        fb[6] = flow::attn_bias(&lens, t, chunk);
        let nfr: f32 = fb[5].iter().sum::<f32>().max(1.0);
        fb[7] = fb[5].iter().map(|m| m / nfr).collect();
        fb
    };
    const NAMES: [&str; 8] = ["xt", "ut", "temb", "mu", "hc", "fmask", "abias", "fw"];
    let do_eval = |prog: &Program, eprog: &mut Program, step: usize, elog: &mut std::fs::File, fe: &mut Frontend, fe_ref: &mut Frontend, tts: &mut Tts| -> Result<()> {
        if held.len() < batch || distill.is_some() {
            return Ok(());
        }
        for (k, &kp) in ek_p.iter().zip(&k_p) {
            eprog.write_input(&dev, *k, &prog.read_input(&dev, kp).unwrap());
        }
        let mut out = Vec::new();
        for tf in [0.1, 0.5, 0.9] {
            let mut rng = Rng::new(77);
            let (mut s, mut n) = (0.0, 0.0);
            for ch in held.chunks(batch).filter(|ch| ch.len() == batch) {
                let us: Vec<&Utt> = ch.iter().collect();
                let fb = cfm_batch(&mut rng, &us, &held, &held_spk, fe, fe_ref, tts, Some(tf));
                for (nme, v) in NAMES.iter().zip(&fb) {
                    eprog.write_input(&dev, emap[nme], v);
                }
                eprog.run(&dev);
                s += eprog.read(&dev, 0)[0] as f64;
                n += 1.0;
            }
            out.push(s / n);
        }
        let mean = out.iter().sum::<f64>() / 3.0;
        writeln!(elog, "{step}\t{:.5}\t{:.5}\t{:.5}\t{mean:.5}\t{:.0}", out[0], out[1], out[2], t0.elapsed().as_secs_f64())?;
        elog.flush()?;
        eprintln!("eval step {step:>6}  held-out CFM loss t=0.1 {:.4}  t=0.5 {:.4}  t=0.9 {:.4}", out[0], out[1], out[2]);
        Ok(())
    };
    if ck.step == 0 {
        do_eval(&prog, &mut eprog, 0, &mut elog, &mut fe, &mut fe_ref, &mut tts)?;
    }
    let start = ck.step;
    let mut step = ck.step;
    drop(ck);
    let mut rng = Rng::new(1000 + step as u64 + tseed.wrapping_mul(1_000_003));
    let stop = run.join("STOP");
    let (mut acc, mut acc_n, mut skipped) = ([0.0f64; 3], 0usize, 0);
    let mut times = Vec::new();
    while step < start + steps {
        let ts = Instant::now();
        let fb = if distill.is_some() {
            distill_batch(&mut rng)
        } else {
            let us: Vec<&Utt> = (0..batch).map(|_| &data[rng.below(data.len())]).collect();
            cfm_batch(&mut rng, &us, &data, &spk, &mut fe, &mut fe_ref, &mut tts, None)
        };
        let lr = lr_at(step, total, peak, warmup);
        let ti = (step + 1) as i32;
        let hyper = [lr as f32, (1.0 - 0.9f64.powi(ti)) as f32, (1.0 - 0.99f64.powi(ti)) as f32, 0.0, clip as f32, 0.0];
        for (nme, v) in NAMES.iter().zip(&fb) {
            prog.write_input(&dev, dmap[nme], v);
        }
        prog.write_input(&dev, k_hyper, &hyper);
        prog.run(&dev);
        let gn = prog.read(&dev, 0)[0];
        let l = prog.read(&dev, 1)[0];
        if !(gn.is_finite() && l.is_finite()) {
            skipped += 1;
            eprintln!("step {}: non-finite (loss {l}, |g| {gn}), skipped", step + 1);
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
        acc[0] += l as f64;
        acc[1] += gn as f64;
        acc[2] += ms;
        acc_n += 1;
        writeln!(log, "{step}\t{l:.5}\t{gn:.4}\t{lr:.3e}\t{ms:.0}")?;
        if step % log_every == 0 {
            let k = acc_n as f64;
            eprintln!("step {step:>6}  loss {:.4}  |g| {:.3}  lr {lr:.2e}  {:.0} ms/step", acc[0] / k, acc[1] / k, acc[2] / k);
            acc = [0.0; 3];
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
            do_eval(&prog, &mut eprog, step, &mut elog, &mut fe, &mut fe_ref, &mut tts)?;
        }
        if stopping {
            std::fs::remove_file(&stop).ok();
            eprintln!("STOP: saved step {step}");
            return Ok(());
        }
    }
    save(&prog, step, &ck_path)?;
    let mut s: Vec<f64> = times.iter().skip(2).copied().collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !s.is_empty() {
        eprintln!("median {:.0} ms/step (frontends + rung-1 cond + host batch + train step)", s[s.len() / 2]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// eval: held-out whole utterances, GT durations, predicted pitch/energy

/// Vocoder over a mel padded to a multiple of 128 frames (programs cached).
fn vocode(dev: &Device, cache: &mut HashMap<usize, Vocoder>, voc: &Path, mel: &[f32], f0: &[f32], c: usize, fr: usize) -> Result<Vec<f32>> {
    let tp = fr.div_ceil(128) * 128;
    let mut m = vec![MEL_FLOOR; c * tp];
    for ci in 0..c {
        m[ci * tp..ci * tp + fr].copy_from_slice(&mel[ci * fr..(ci + 1) * fr]);
    }
    let mut f = vec![0.0f32; tp];
    f[..fr].copy_from_slice(f0);
    let v = match cache.entry(tp) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => e.insert(Vocoder::new(dev, voc, tp)?),
    };
    let mut y = v.run(dev, &m, &f, tp);
    y.truncate(fr * HOP);
    Ok(y)
}

pub fn flow_eval(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let fc = FConfig::by_name(&a.opt("--fcfg").unwrap_or_else(|| "base".into())).ok_or_else(|| anyhow!("fcfg"))?;
    let fspecs = fc.params();
    let ac_run = PathBuf::from(a.need("--ac-run")?);
    let (ac_params, nm, ac_step) = load_run(&ac_run, &tc, a.opt("--ac-ckpt"))?;
    let tpath = PathBuf::from(a.need("--flow")?);
    let fnm = FNorm::load(&tpath)?;
    let tparams = ckpt::load_specs(&tpath, &fspecs, false)?.params;
    let sparams = match a.opt("--student") {
        Some(s) => Some(ckpt::load_specs(Path::new(&s), &fspecs, false)?.params),
        None => None,
    };
    let root = PathBuf::from(a.need("--root")?);
    let durs = read_durs(&a.need("--durs")?)?;
    let n_utts: usize = a.get("--n-utts", 64);
    let batch: usize = a.get("--batch", 8);
    let temp: f64 = a.get("--temp", 1.0);
    let seed: u64 = a.get("--seed", 0);
    let nfes: Vec<usize> = a.opt("--nfe").unwrap_or_else(|| "32,16,8".into()).split(',').filter_map(|x| x.parse().ok()).collect();
    let (nmax, tmax): (usize, usize) = (a.get("--n", 192), a.get("--t", 1024));
    let chunk = chunk_of(a);
    let voc = a.opt("--voc").map(PathBuf::from);
    let held = load(&a.need("--held")?, &durs, &root, &|n, t| n <= nmax && t <= tmax * 3 / 4 && t >= 16)?;
    let stride = (held.len() / n_utts).max(1);
    let held: Vec<Utt> = held.into_iter().step_by(stride).take(n_utts).collect();
    let hspk = by_speaker(&held);
    let out = PathBuf::from(a.need("--out-dir")?);
    std::fs::create_dir_all(&out)?;
    eprintln!("flow-eval: {} utts ({} speakers), rung-1 step {ac_step}, nfe {nfes:?}, temp {temp}, chunk {chunk:?}", held.len(), hspk.len());
    let dev = Device::new();
    let d = Dims { batch, n: nmax, t: tmax, r: REF_T };
    let (c, t) = (tc.n_mels, tmax);
    let mut tts = Tts::new(&dev, &tc, &ac_params, d);
    let mut fe = Frontend::new(&dev, batch, tmax);
    let mut fe_ref = Frontend::new(&dev, batch, REF_T);
    let mut vel = Vel::new(&dev, &fc, &tparams, FDims { batch, t });
    let mut vcache: HashMap<usize, Vocoder> = HashMap::new();
    let mut rng = Rng::new(77);
    let mut meta = std::fs::File::create(out.join("eval.tsv"))?;
    writeln!(meta, "utt\tframes\tvariants")?;
    let mut variants: Vec<String> = vec!["gt".into(), "r1".into()];
    variants.extend(nfes.iter().map(|n| format!("t{n}")));
    if sparams.is_some() {
        variants.extend(["s1".into(), "s2".into()]);
    }
    let mut timing: HashMap<String, f64> = HashMap::new();
    for ch in held.chunks(batch) {
        let mut us: Vec<&Utt> = ch.iter().collect();
        let real = us.len();
        while us.len() < batch {
            us.push(us[0]);
        }
        let (hb, mel, refmel) = prep_crops(&dev, &tc, d, &nm, &mut rng, false, &us, &held, &hspk, &mut fe, &mut fe_ref);
        let fm = hb.get("fmask").to_vec();
        let lens = lens_of(&fm, batch, t);
        let (mu, hid, f0c, _) = cond(&dev, &mut tts, &hb, &refmel, false);
        let mun = fnm.norm(&mu, &fm, batch, t);
        let ab = flow::attn_bias(&lens, t, chunk);
        let x0 = noise(&mut Rng::new(1234 + us.len() as u64 + seed.wrapping_mul(1_000_003)), &fm, batch, c, t, temp);
        let mut outs: Vec<(String, Vec<f32>)> = vec![("gt".into(), mel.clone()), ("r1".into(), mu.clone())];
        vel.set_params(&dev, &tparams);
        vel.set_cond(&dev, &mun, &hid, &fm, &ab);
        for &n in &nfes {
            let ts = Instant::now();
            let (x1, _) = vel.sample(&dev, &x0, n, false);
            *timing.entry(format!("t{n}")).or_default() += ts.elapsed().as_secs_f64();
            outs.push((format!("t{n}"), fnm.denorm(&x1)));
        }
        if let Some(sp) = &sparams {
            vel.set_params(&dev, sp);
            for n in [1usize, 2] {
                let ts = Instant::now();
                let (x1, _) = vel.sample(&dev, &x0, n, true);
                *timing.entry(format!("s{n}")).or_default() += ts.elapsed().as_secs_f64();
                outs.push((format!("s{n}"), fnm.denorm(&x1)));
            }
        }
        for (bi, u) in us.iter().enumerate().take(real) {
            let fr = lens[bi];
            let grab = |x: &[f32]| -> Vec<f32> { (0..c).flat_map(|ci| x[(bi * c + ci) * t..(bi * c + ci) * t + fr].to_vec()).collect() };
            // F0: YIN for GT copy-synthesis, the rung-1 head for every model
            let yin: Vec<f32> = u.f0[..fr].to_vec();
            let head: Vec<f32> = (0..fr)
                .map(|f| {
                    let (z, v) = (f0c[(bi * 2) * t + f], f0c[(bi * 2 + 1) * t + f]);
                    if v > 0.0 { ((z as f64) * nm.f0_sd + nm.f0_mu).exp() as f32 } else { 0.0 }
                })
                .collect();
            write_npy(&out.join(format!("{}.f0head.npy", u.id)), &[fr], &head)?;
            write_npy(&out.join(format!("{}.f0yin.npy", u.id)), &[fr], &yin)?;
            for (name, x) in &outs {
                let m = grab(x);
                write_npy(&out.join(format!("{}.{name}.npy", u.id)), &[c, fr], &m)?;
                if let Some(vp) = &voc {
                    let f0 = if name == "gt" { &yin } else { &head };
                    let y = vocode(&dev, &mut vcache, vp, &m, f0, c, fr)?;
                    ribt::data::write_wav(&out.join(format!("{}.{name}.wav", u.id)), &y, SR as u32)?;
                }
            }
            // ground-truth recording (trimmed to whole frames)
            ribt::data::write_wav(&out.join(format!("{}.rec.wav", u.id)), &u.audio_f32(0, fr), SR as u32)?;
            writeln!(meta, "{}\t{fr}\t{}", u.id, variants.join(","))?;
        }
        eprintln!("batch done ({} utts)", real);
    }
    let nb = held.len().div_ceil(batch) as f64;
    for (k, v) in &timing {
        eprintln!("{k}: {:.1} ms per batch of {batch} × {t} frames", v / nb * 1e3);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// checks

fn random_flow_data(fc: &FConfig, d: FDims, rng: &mut Rng, chunk: Option<(usize, usize)>) -> Vec<(&'static str, Vec<f32>)> {
    let (b, t, c) = (d.batch, d.t, fc.n_mels);
    let lens: Vec<usize> = (0..b).map(|_| t / 2 + rng.below(t / 2 + 1)).collect();
    let mut fm = vec![0.0f32; b * t];
    for (bi, &n) in lens.iter().enumerate() {
        fm[bi * t..bi * t + n].iter_mut().for_each(|v| *v = 1.0);
    }
    let nfr: f32 = fm.iter().sum();
    let mut temb = Vec::new();
    for _ in 0..b {
        temb.extend(flow::time_feats(fc, rng.uniform(), 0.0));
    }
    vec![
        ("xt", noise(rng, &fm, b, c, t, 1.0)),
        ("ut", noise(rng, &fm, b, c, t, 1.0)),
        ("mu", noise(rng, &fm, b, c, t, 1.0)),
        ("hc", noise(rng, &fm, b, fc.dc, t, 1.0)),
        ("temb", temb),
        ("abias", flow::attn_bias(&lens, t, chunk)),
        ("fw", fm.iter().map(|m| m / nfr).collect()),
        ("fmask", fm),
    ]
}

/// One train step of the check cfg, GPU vs the f64 interpreter.
pub fn flow_check(a: &Args) -> Result<()> {
    let name = a.opt("--fcfg").unwrap_or_else(|| "check".into());
    let fc = FConfig::by_name(&name).ok_or_else(|| anyhow!("fcfg"))?;
    let specs = fc.params();
    let d = FDims { batch: a.get("--batch", 3), t: a.get("--t", 48) };
    let mut rng = Rng::new(4);
    let params: Vec<Vec<f32>> = flow::init_params(&fc, 1).into_iter().map(|p| p.iter().map(|v| v + (0.05 * rng.normal()) as f32).collect()).collect();
    let data = random_flow_data(&fc, d, &mut rng, Some((8, 1)));
    let dr: Vec<(&str, &[f32])> = data.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let tt = flow::train_graph(&fc, d);
    let mut inp = flow::assemble(&tt.fg.b, &specs, &params, None, &dr);
    inp[tt.fg.b.input_index(tt.hyper)] = vec![1e-3, 0.1, 0.02, 0.0, 1.0, 0.0];
    let i64s: Vec<Vec<f64>> = inp.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
    let t = Instant::now();
    let want = tt.fg.b.g.eval(&i64s, &tt.outputs);
    eprintln!("interpreter {:.1} s", t.elapsed().as_secs_f64());
    let dev = Device::new();
    let mut prog = Program::new(&dev, &tt.fg.b.g, &tt.outputs, &refs(&inp));
    prog.run(&dev);
    let rel = |got: &[f32], want: &[f64]| -> f64 {
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
        got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((*g as f64 - w).abs())) / scale
    };
    let np = specs.len();
    let loss_err = rel(&prog.read(&dev, 0), &want[0]).max(rel(&prog.read(&dev, 1), &want[1]));
    let (mut m_err, mut worst) = (0.0f64, String::new());
    for i in 0..np {
        let o = 2 + np + i;
        let e = rel(&prog.read(&dev, o), &want[o]);
        if e > m_err {
            m_err = e;
            worst = specs[i].name.clone();
        }
    }
    eprintln!(
        "{} vs f64 interpreter, fcfg {name} ({d:?}, {} params, chunk-masked attention): loss + |g| worst rel {loss_err:.2e} (loss {} vs {}); m' (grads) worst rel {m_err:.2e} ({worst})",
        ribt::runtime::BACKEND,
        fc.n_params(),
        prog.read(&dev, 1)[0],
        want[1][0]
    );
    if loss_err > 1e-4 || m_err > 1e-3 {
        return Err(anyhow!("GPU/interpreter mismatch"));
    }
    Ok(())
}

/// Train-step and velocity-step kernel ms on random data.
pub fn flow_bench(a: &Args) -> Result<()> {
    let fc = FConfig::by_name(&a.opt("--fcfg").unwrap_or_else(|| "base".into())).ok_or_else(|| anyhow!("fcfg"))?;
    let specs = fc.params();
    let d = FDims { batch: a.get("--batch", 32), t: a.get("--t", 384) };
    let mut rng = Rng::new(4);
    let data = random_flow_data(&fc, d, &mut rng, None);
    let dr: Vec<(&str, &[f32])> = data.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let dev = Device::new();
    let mut params = flow::init_params(&fc, 0);
    for p in params.iter_mut() {
        p.iter_mut().for_each(|v| *v += (0.01 * rng.normal()) as f32);
    }
    if !a.0.iter().any(|x| x == "--infer-only") {
        let tt = flow::train_graph(&fc, d);
        let mut inp = flow::assemble(&tt.fg.b, &specs, &params, None, &dr);
        inp[tt.fg.b.input_index(tt.hyper)] = vec![1e-4, 0.1, 0.02, 0.0, 1.0, 0.0];
        let t = Instant::now();
        let mut prog = Program::new(&dev, &tt.fg.b.g, &tt.outputs, &refs(&inp));
        eprintln!("{} flow params, {d:?}: train {} kernels, compile {:.1} s", fc.n_params(), prog.kernels(), t.elapsed().as_secs_f64());
        let ms = prog.time(&dev, a.get("--iters", 10));
        prog.run(&dev);
        eprintln!("flow train step {ms:.1} ms (kernels only); loss {}", prog.read(&dev, 1)[0]);
    }
    for (b, t) in [(1usize, a.get("--infer-t", 192usize)), (d.batch, d.t)] {
        let fd = FDims { batch: b, t };
        let fg = flow::flow_graph(&fc, fd, false);
        let data = random_flow_data(&fc, fd, &mut rng, None);
        let dr: Vec<(&str, &[f32])> = data.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        let inp = flow::assemble(&fg.b, &specs, &params, None, &dr);
        let mut prog = Program::new(&dev, &fg.b.g, &[fg.v], &refs(&inp));
        let ms = prog.time(&dev, 20);
        #[cfg(feature = "metal")]
        if a.0.iter().any(|x| x == "--profile") && b > 1 {
            let ex = &prog.exe.0;
            let nk = ex.program.kernels.len();
            let mut rows: Vec<(f64, usize)> = (0..nk).map(|i| (ex.time_kernels(&dev.0, i..i + 1, 3), i)).collect();
            let sum: f64 = rows.iter().map(|r| r.0).sum();
            rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            eprintln!("per-kernel sum {sum:.1} ms; top 20:");
            for (ms, i) in rows.iter().take(20) {
                let k = &ex.program.kernels[*i];
                let sh: Vec<usize> = prog.graph.shape(k.node).iter().map(|d| d.n()).collect();
                eprintln!("  {ms:8.2} ms  k{i:<5} shape {sh:?} groups {:?} thr {} frag {} reads {}", k.groups, k.threads, k.fragment, k.reads.len());
            }
        }
        // wall time of one step including host write/read (what synth pays)
        let mut vel = Vel::new(&dev, &fc, &params, fd);
        let get = |k: &str| data.iter().find(|(n, _)| *n == k).unwrap().1.clone();
        vel.set_cond(&dev, &get("mu"), &get("hc"), &get("fmask"), &get("abias"));
        let x = get("xt");
        let _ = vel.step(&dev, &x, 0.5, 0.0);
        let t0 = Instant::now();
        for _ in 0..20 {
            let _ = vel.step(&dev, &x, 0.5, 0.0);
        }
        let wall = t0.elapsed().as_secs_f64() * 1e3 / 20.0;
        eprintln!("velocity step B{b} T{t}: {ms:.2} ms kernels, {wall:.2} ms wall ({} kernels)", prog.kernels());
    }
    Ok(())
}
