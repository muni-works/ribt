//! `ribt-tts ctx-train` and the context path of `synth` (the ribt paper):
//! context-aware prosody. The base acoustic model is frozen; a small context
//! path (previous turn, role, emphasis hints → AdaLN-zero on the variance
//! predictors) learns rule-derived, context-dependent prosody targets laid on
//! top of the measured af_heart prosody of each corpus utterance.
//!
//!   ribt-tts ctx-train --ac-run AC --ac-ckpt step….safetensors --ctx-data ctx.tsv --phon train.phon.tsv
//!                      --durs D.durs.tsv --root R --run DIR [--ref REF.wav] [--batch 32] [--n 192] [--m 128]
//!                      [--steps N] [--total N] [--lr 1e-3] [--warmup 200] [--eval-every 500]
//!                      [--upweight question_yn=16,ack=40] [--init-ctx CTX.safetensors]
//!       ctx.tsv rows from scripts/ctx/ctx_data.py: utt, kind, role, prev, emph, focus, given, rise
//!       (touch RUN/STOP to save + exit; RUN/ctx.safetensors holds the context params only)
//!
//!   synth … --ctx DIR/ctx.safetensors --lines L.tsv   L: name, text, prev ("-" none), role, [emph]
//!       text may mark emphasis as *word*; role in none|answer|question|ack|correction ("none" with
//!       prev "-" = gate 0, exactly the base model). Writes NAME_TAG.durs.txt (tokens, durations).

use super::*;
use ribt::acoustic::{CtxConfig, CTX_ROLES, N_CTX_LOSS};

pub const SPACE: u16 = 16;
const PRIMARY: u16 = 156;

/// Word index per token (pads and spaces: None), words = space-separated runs.
pub fn word_index(toks: &[u16]) -> Vec<Option<usize>> {
    let mut w = 0usize;
    let mut seen = false;
    toks.iter()
        .enumerate()
        .map(|(i, &t)| {
            if i == 0 || i + 1 == toks.len() {
                return None;
            }
            if t == SPACE {
                if seen {
                    w += 1;
                    seen = false;
                }
                return None;
            }
            seen = true;
            Some(w)
        })
        .collect()
}

pub fn role_index(r: &str) -> usize {
    CTX_ROLES.iter().position(|x| *x == r).unwrap_or(0)
}

/// Previous turn → token row (both pads; the last `m − 2` phoneme tokens).
pub fn prev_tokens(prev: &str, m: usize) -> Vec<u16> {
    let mut v = vec![0u16];
    if !prev.is_empty() && prev != "-" {
        let (ps, _) = lilt::phonemize(prev);
        let ids: Vec<u16> = lilt::vocab::ids(&ps).iter().map(|&x| x as u16).collect();
        let k = ids.len().saturating_sub(m - 2);
        v.extend(&ids[k..]);
    }
    v.push(0);
    v
}

fn idx_list(s: &str) -> Vec<usize> {
    if s == "-" {
        return Vec::new();
    }
    s.split(',').filter_map(|x| x.parse().ok()).collect()
}

#[derive(Clone)]
struct Row {
    utt: usize,
    kind: String,
    role: usize,
    prev: Vec<u16>,
    emph: Vec<usize>,
    focus: Vec<usize>,
    given: Vec<usize>,
    rise: bool,
}

/// The §25.2 rules on one utterance's measured token targets (normalized
/// pitch p, energy e, ln(d+1)). Magnitudes follow the focus literature
/// (on-focus F0 +3–5 st and ~+25 % duration, post-focus compression of range
/// and level; Xu & Xu 2005, Xu 2011): focus word +3.5 st (+1.5 st more on the
/// primary-stressed syllable), energy +0.4 σ, duration ×1.25; post-focus words
/// range ×0.5, −1.5 st, energy −0.25 σ, ×0.95; pre-focus given words range
/// ×0.8, −0.5 st; given words without a focus range ×0.6, −1 st, −0.2 σ,
/// ×0.92; yes/no rise: the last word ramps 0 → +5 st (+1 st on the one
/// before); ack: range ×0.6, −1 st, −0.2 σ, ×0.85 (quicker, flatter).
/// Unvoiced tokens (p = 0 exactly) keep their pitch target.
fn rule_targets(toks: &[u16], p: &mut [f32], e: &mut [f32], ld: &mut [f32], nm: &Norm, r: &Row) {
    let st = |s: f64| (s * std::f64::consts::LN_2 / 12.0 / nm.f0_sd) as f32;
    let wi = word_index(toks);
    let voiced: Vec<bool> = p.iter().map(|&v| v != 0.0).collect();
    let nv = voiced.iter().filter(|&&v| v).count().max(1) as f32;
    let pm: f32 = p.iter().zip(&voiced).filter(|(_, &v)| v).map(|(x, _)| *x).sum::<f32>() / nv;
    let scale_d = |ld: &mut f32, s: f32| *ld = ((ld.exp() - 1.0) * s + 1.0).ln();
    let n = toks.len();
    let shape = |p: &mut f32, e: &mut f32, ld: &mut f32, v: bool, range: f32, shift: f64, de: f32, ds: f32| {
        if v {
            *p = pm + range * (*p - pm) + st(shift);
        }
        *e += de;
        scale_d(ld, ds);
    };
    if !r.focus.is_empty() {
        let (f0, f1) = (*r.focus.iter().min().unwrap(), *r.focus.iter().max().unwrap());
        let mut after_primary = 0usize;
        for i in 0..n {
            let Some(w) = wi[i] else { continue };
            if r.focus.contains(&w) {
                if toks[i] == PRIMARY {
                    after_primary = 4;
                }
                let extra = if after_primary > 0 { 1.5 } else { 0.0 };
                after_primary = after_primary.saturating_sub(1);
                shape(&mut p[i], &mut e[i], &mut ld[i], voiced[i], 1.0, 3.5 + extra, 0.4, 1.25);
            } else if w > f1 {
                shape(&mut p[i], &mut e[i], &mut ld[i], voiced[i], 0.5, -1.5, -0.25, 0.95);
            } else if w < f0 && r.given.contains(&w) {
                shape(&mut p[i], &mut e[i], &mut ld[i], voiced[i], 0.8, -0.5, -0.1, 1.0);
            }
        }
    } else if !r.given.is_empty() {
        for i in 0..n {
            if let Some(w) = wi[i] {
                if r.given.contains(&w) {
                    shape(&mut p[i], &mut e[i], &mut ld[i], voiced[i], 0.6, -1.0, -0.2, 0.92);
                }
            }
        }
    }
    if r.rise {
        if let Some(last) = wi.iter().flatten().max().copied() {
            // ramp over the last word's VOICED tokens (the full +5 st on the last vowel; an
            // all-token ramp put its top on "?" and final consonants, whose targets are masked)
            let span: Vec<usize> = (0..n).filter(|&i| wi[i] == Some(last) && voiced[i]).collect();
            let k = span.len().max(2) - 1;
            for (j, &i) in span.iter().enumerate() {
                p[i] += st(if span.len() == 1 { 5.0 } else { 5.0 * j as f64 / k as f64 });
            }
            for i in 0..n {
                if last > 0 && wi[i] == Some(last - 1) && voiced[i] {
                    p[i] += st(1.0);
                }
            }
        }
    }
    if CTX_ROLES[r.role] == "ack" {
        for i in 1..n.saturating_sub(1) {
            shape(&mut p[i], &mut e[i], &mut ld[i], voiced[i], 0.6, -1.0, -0.2, 0.85);
        }
    }
}

/// Context tensors for a batch: prev one-hot `[B,V,M]`, pmask, role, emph `[B,1,N]`, gate.
pub fn ctx_tensors(vocab: usize, n: usize, m: usize, items: &[(&[u16], &[u16], usize, &[usize], f32)]) -> Vec<(&'static str, Vec<f32>)> {
    let b = items.len();
    let nr = CTX_ROLES.len();
    let mut prev = vec![0.0f32; b * vocab * m];
    let mut pmask = vec![0.0f32; b * m];
    let mut role = vec![0.0f32; b * nr];
    let mut emph = vec![0.0f32; b * n];
    let mut gate = vec![0.0f32; b];
    for (bi, (toks, pv, r, em, g)) in items.iter().enumerate() {
        for (i, &id) in pv.iter().enumerate().take(m) {
            prev[(bi * vocab + id as usize) * m + i] = 1.0;
            pmask[bi * m + i] = 1.0;
        }
        role[bi * nr + r] = 1.0;
        for (i, w) in word_index(toks).iter().enumerate() {
            if w.is_some_and(|w| em.contains(&w)) {
                emph[bi * n + i] = 1.0;
            }
        }
        gate[bi] = *g;
    }
    vec![("prev", prev), ("pmask", pmask), ("role", role), ("emph", emph), ("gate", gate)]
}

/// Reference mel of one clip (`REF_T` frames after 0.3 s), tiled over a batch.
pub fn ref_batch(dev: &Device, path: &str, batch: usize, n_mels: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let rp = match path.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(path),
    };
    let (x, sr) = ribt::data::read_wav(&rp)?;
    if sr as usize != SR {
        return Err(anyhow!("{path}: {sr} Hz"));
    }
    let fr = x.len() / HOP;
    let off = if fr > REF_T + 30 { 30 } else { 0 };
    let len = (fr - off).min(REF_T);
    let mut fe = Frontend::new(dev, 1, REF_T);
    let m = fe.run(dev, &[x[off * HOP..(off + len) * HOP].to_vec()]);
    let mut mel = vec![0.0f32; batch * n_mels * REF_T];
    let mut rw = vec![0.0f32; batch * REF_T];
    for b in 0..batch {
        mel[b * n_mels * REF_T..(b + 1) * n_mels * REF_T].copy_from_slice(&m);
        for k in 0..len {
            rw[b * REF_T + k] = 1.0 / len as f32;
        }
    }
    Ok((mel, rw))
}

pub const AF_REF: &str = "~/data/voices/afheart/00000/afheart_00000_000038.wav";

pub fn ctx_train(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let cc = CtxConfig::base();
    let specs = acoustic::ctx_specs(&tc, &cc);
    let nb = tc.params().len();
    let nc = specs.len() - nb;
    let run = PathBuf::from(a.need("--run")?);
    std::fs::create_dir_all(&run)?;
    let ac_run = PathBuf::from(a.need("--ac-run")?);
    let (base_p, nm, _) = load_run(&ac_run, &tc, a.opt("--ac-ckpt"))?;
    nm.save(&run.join("norm.txt"))?;
    let root = PathBuf::from(a.need("--root")?);
    let batch: usize = a.get("--batch", 32);
    let nmax: usize = a.get("--n", 192);
    let m: usize = a.get("--m", 128);
    let steps: usize = a.get("--steps", 3000);
    let total: usize = a.get("--total", steps);
    let peak: f64 = a.get("--lr", 1e-3);
    let warmup: usize = a.get("--warmup", 200);
    let clip: f64 = a.get("--clip", 1.0);
    let eval_every: usize = a.get("--eval-every", 500);
    let t0 = Instant::now();

    // rows → the utterances they need (a filtered phon file, so `load` reads only those)
    let text = std::fs::read_to_string(a.need("--ctx-data")?)?;
    let raw: Vec<Vec<&str>> = text.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).map(|l| l.split('\t').collect()).filter(|c: &Vec<&str>| c.len() >= 8).collect();
    let need: std::collections::HashSet<&str> = raw.iter().map(|c| c[0]).collect();
    let phon = a.need("--phon")?;
    let pdir = run.join("phon");
    std::fs::create_dir_all(&pdir)?;
    let split = Path::new(&phon).file_name().unwrap().to_string_lossy().to_string();
    let mut f = std::fs::File::create(pdir.join(&split))?;
    for line in std::fs::read_to_string(&phon)?.lines() {
        if need.contains(line.split('\t').next().unwrap_or("")) {
            writeln!(f, "{line}")?;
        }
    }
    drop(f);
    let durs = read_durs(&a.need("--durs")?)?;
    let keep = move |n: usize, _fr: usize| n <= nmax;
    let data = load(&pdir.join(&split).to_string_lossy(), &durs, &root, &keep)?;
    let by_id: HashMap<&str, usize> = data.iter().enumerate().map(|(i, u)| (u.id.as_str(), i)).collect();
    let mut rows = Vec::new();
    for c in &raw {
        let Some(&u) = by_id.get(c[0]) else { continue };
        rows.push(Row {
            utt: u,
            // yes/no questions are their own kind (sampling weight, eval): same role as wh ones
            kind: if c[1] == "question" && c[7] == "1" { "question_yn".into() } else { c[1].to_string() },
            role: role_index(c[2]),
            prev: prev_tokens(c[3], m),
            emph: idx_list(c[4]),
            focus: idx_list(c[5]),
            given: idx_list(c[6]),
            rise: c[7] == "1",
        });
    }
    // held-out: every utterance whose index ≡ 0 (mod 50)
    let (held, train): (Vec<Row>, Vec<Row>) = rows.into_iter().partition(|r| r.utt % 50 == 0);
    eprintln!("ctx-train: {} utts, {} train rows, {} held rows, loaded in {:.1} s", data.len(), train.len(), held.len(), t0.elapsed().as_secs_f64());
    if train.is_empty() {
        return Err(anyhow!("no rows"));
    }
    // targets per row (rules on the measured prosody), cached
    let targets = |r: &Row| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let u = &data[r.utt];
        let (mut p, mut e) = token_targets(u, &nm);
        let mut ld: Vec<f32> = u.durs.iter().map(|&d| (d as f32 + 1.0).ln()).collect();
        rule_targets(&u.toks, &mut p, &mut e, &mut ld, &nm, r);
        (p, e, ld)
    };
    let tr_t: Vec<_> = train.iter().map(targets).collect();
    // diagnostics: how far the rules move each kind's pitch targets (st, voiced tokens)
    {
        let mut by: HashMap<String, (f64, f64, usize, usize)> = HashMap::new();
        for (r, t) in train.iter().zip(&tr_t) {
            let (p0, _) = token_targets(&data[r.utt], &nm);
            let e = by.entry(format!("{}{}", r.kind, if r.rise { "+rise" } else { "" })).or_default();
            let sc = nm.f0_sd * 12.0 / std::f64::consts::LN_2;
            let d: Vec<f64> = p0.iter().zip(&t.0).filter(|(a, _)| **a != 0.0).map(|(a, b)| ((b - a) as f64 * sc).abs()).collect();
            e.0 += d.iter().sum::<f64>();
            e.1 = e.1.max(d.iter().cloned().fold(0.0, f64::max));
            e.2 += d.len();
            e.3 += 1;
        }
        let mut ks: Vec<_> = by.into_iter().collect();
        ks.sort_by(|a, b| a.0.cmp(&b.0));
        for (k, (s, mx, n, rows)) in ks {
            eprintln!("targets {k:<14} {rows:>6} rows  mean |Δpitch| {:.2} st over voiced tokens, max {:.1} st", s / n.max(1) as f64, mx);
        }
    }
    let he_t: Vec<_> = held.iter().map(targets).collect();

    let dev = Device::new();
    let d = Dims { batch, n: nmax, t: 64, r: REF_T };
    let (refmel, rw) = ref_batch(&dev, &a.opt("--ref").unwrap_or_else(|| AF_REF.into()), batch, tc.n_mels)?;
    let ck_path = run.join("ctx.safetensors");
    let cspecs = cc.params(&tc);
    let (cparams, adam, start) = if ck_path.exists() {
        let ck = ckpt::load_specs(&ck_path, &cspecs, false)?;
        (ck.params, ck.adam, ck.step)
    } else if let Some(i) = a.opt("--init-ctx") {
        (ckpt::load_specs(Path::new(&i), &cspecs, false)?.params, None, 0)
    } else {
        let mut rng = Rng::new(0);
        (cspecs.iter().map(|s| s.init_data(&mut rng)).collect(), None, 0)
    };
    let mut all = base_p.clone();
    all.extend(cparams);
    let ct = acoustic::ctx_train_graph(&tc, &cc, d, m);
    // Adam state: zeros for the frozen base (never read), the ctx moments from the checkpoint
    let (mut am, mut av): (Vec<Vec<f32>>, Vec<Vec<f32>>) = specs.iter().map(|s| (vec![0.0; s.numel()], vec![0.0; s.numel()])).unzip();
    if let Some((m0, v0)) = adam {
        for j in 0..nc {
            am[nb + j] = m0[j].clone();
            av[nb + j] = v0[j].clone();
        }
    }
    let inputs = acoustic::assemble(&ct.b, &specs, &all, Some((&am, &av)), &[("refmel", &refmel), ("rw", &rw)]);
    let tcomp = Instant::now();
    let mut prog = Program::new(&dev, &ct.b.g, &ct.outputs, &refs(&inputs));
    drop(inputs);
    eprintln!("ctx train program: {} kernels, compile {:.1} s; ctx params {}", prog.kernels(), tcomp.elapsed().as_secs_f64(), cspecs.iter().map(|s| s.numel()).sum::<usize>());
    let dmap = input_map(&ct.b);
    let n_out = ct.outputs.len();
    let k_hyper = ct.b.input_index(ct.hyper);
    let k_p: Vec<usize> = ct.params[nb..].iter().map(|&t| ct.b.input_index(t)).collect();
    let k_m: Vec<usize> = ct.m.iter().map(|&t| ct.b.input_index(t)).collect();
    let k_v: Vec<usize> = ct.v.iter().map(|&t| ct.b.input_index(t)).collect();
    let (o_p, o_m, o_v) = (1 + N_CTX_LOSS, 1 + N_CTX_LOSS + nc, 1 + N_CTX_LOSS + 2 * nc);

    let fill = |prog: &mut Program, rs: &[&Row], ts: &[&(Vec<f32>, Vec<f32>, Vec<f32>)], gate: f32| {
        let n = nmax;
        let rows_t: Vec<&[u16]> = rs.iter().map(|r| data[r.utt].toks.as_slice()).collect();
        let (oh, tm) = onehot_rows(&tc, &rows_t, n);
        let mut p_tok = vec![0.0f32; batch * n];
        let mut e_tok = vec![0.0f32; batch * n];
        let mut ld_tok = vec![0.0f32; batch * n];
        for (bi, t) in ts.iter().enumerate() {
            p_tok[bi * n..bi * n + t.0.len()].copy_from_slice(&t.0);
            e_tok[bi * n..bi * n + t.1.len()].copy_from_slice(&t.1);
            ld_tok[bi * n..bi * n + t.2.len()].copy_from_slice(&t.2);
        }
        let ntok: f32 = tm.iter().sum::<f32>().max(1.0);
        let tw: Vec<f32> = tm.iter().map(|v| v / ntok).collect();
        let items: Vec<(&[u16], &[u16], usize, &[usize], f32)> = rs.iter().map(|r| (data[r.utt].toks.as_slice(), r.prev.as_slice(), r.role, r.emph.as_slice(), gate)).collect();
        for (k, v) in [("onehot", oh), ("tmask", tm), ("p_tok", p_tok), ("e_tok", e_tok), ("ld_tok", ld_tok), ("tw", tw)].into_iter().chain(ctx_tensors(tc.vocab, n, m, &items)) {
            prog.write_input(&dev, dmap[k], &v);
        }
    };
    let open_log = |name: &str, header: &str| -> Result<std::fs::File> {
        let p = run.join(name);
        let new = !p.exists();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p)?;
        if new {
            writeln!(f, "{header}")?;
        }
        Ok(f)
    };
    let mut log = open_log("log.tsv", "step\ttotal\tdur\tpitch\tenergy\tgrad_norm\tlr\tms")?;
    let mut elog = open_log("eval.tsv", "step\tkind\trows\tbase_total\tbase_dur\tbase_pitch\tbase_energy\tctx_total\tctx_dur\tctx_pitch\tctx_energy")?;
    // held-out by kind: base (gate 0) vs context path, lr 0 (no swap: params unchanged)
    let mut kinds: Vec<String> = held.iter().map(|r| r.kind.clone()).collect();
    kinds.sort();
    kinds.dedup();
    let do_eval = |prog: &mut Program, step: usize, elog: &mut std::fs::File| -> Result<()> {
        prog.write_input(&dev, k_hyper, &[0.0, 0.1, 0.02, 0.0, clip as f32, 0.0]);
        for kind in &kinds {
            let idx: Vec<usize> = (0..held.len()).filter(|&i| &held[i].kind == kind).collect();
            let mut acc = [[0.0f64; N_CTX_LOSS]; 2];
            let mut nb_ = 0.0;
            let mut dp = 0.0;
            for c in idx.chunks(batch) {
                // short last chunk: repeat rows (a mean over duplicates)
                let ids: Vec<usize> = (0..batch).map(|j| c[j % c.len()]).collect();
                let rs: Vec<&Row> = ids.iter().map(|&i| &held[i]).collect();
                let ts: Vec<_> = ids.iter().map(|&i| &he_t[i]).collect();
                let mut pr = Vec::new();
                for (gi, gate) in [0.0f32, 1.0].into_iter().enumerate() {
                    fill(prog, &rs, &ts, gate);
                    prog.run(&dev);
                    for (l, x) in acc[gi].iter_mut().enumerate() {
                        *x += prog.read(&dev, 1 + l)[0] as f64;
                    }
                    pr.push(prog.read(&dev, n_out - 1));
                }
                let sc = nm.f0_sd * 12.0 / std::f64::consts::LN_2;
                dp += pr[0].iter().zip(&pr[1]).map(|(a, b)| ((a - b) as f64 * sc).abs()).sum::<f64>() / pr[0].len() as f64;
                nb_ += 1.0;
            }
            let f = |v: &[f64; N_CTX_LOSS]| v.iter().map(|x| format!("{:.5}", x / nb_)).collect::<Vec<_>>().join("\t");
            writeln!(elog, "{step}\t{kind}\t{}\t{}\t{}", idx.len(), f(&acc[0]), f(&acc[1]))?;
            eprintln!("eval {step:>5} {kind:<9} ({:>4} rows)  base {:.4}  ctx {:.4}  mean |Δ pred pitch| {:.3} st/token", idx.len(), acc[0][0] / nb_, acc[1][0] / nb_, dp / nb_);
        }
        elog.flush()?;
        Ok(())
    };
    let save = |prog: &Program, step: usize, path: &Path| -> Result<()> {
        let rd = |ks: &[usize]| -> Vec<Vec<f32>> { ks.iter().map(|&k| prog.read_input(&dev, k).unwrap()).collect() };
        ckpt::save_specs(path, &cspecs, &Checkpoint { params: rd(&k_p), adam: Some((rd(&k_m), rd(&k_v))), step })
    };
    if start == 0 {
        do_eval(&mut prog, 0, &mut elog)?;
    }
    // --upweight kind=k,…: sample those kinds' rows k× as often (rare roles: question, ack)
    let up: HashMap<String, usize> = a
        .opt("--upweight")
        .unwrap_or_default()
        .split(',')
        .filter_map(|kv| kv.split_once('=').and_then(|(k, v)| Some((k.to_string(), v.parse().ok()?))))
        .collect();
    let pool: Vec<usize> = (0..train.len()).flat_map(|i| std::iter::repeat(i).take(*up.get(&train[i].kind).unwrap_or(&1))).collect();
    eprintln!("sampling pool {} (upweight {up:?})", pool.len());
    let mut rng = Rng::new(4242 + start as u64);
    let stop = run.join("STOP");
    let mut step = start;
    let mut acc = [0.0f64; N_CTX_LOSS + 2];
    let mut acc_n = 0usize;
    while step < start + steps {
        let ts = Instant::now();
        let ids: Vec<usize> = (0..batch).map(|_| pool[rng.below(pool.len())]).collect();
        let rs: Vec<&Row> = ids.iter().map(|&i| &train[i]).collect();
        let tt: Vec<_> = ids.iter().map(|&i| &tr_t[i]).collect();
        fill(&mut prog, &rs, &tt, 1.0);
        let lr = lr_at(step, total, peak, warmup);
        let t = (step + 1) as i32;
        prog.write_input(&dev, k_hyper, &[lr as f32, (1.0 - 0.9f64.powi(t)) as f32, (1.0 - 0.98f64.powi(t)) as f32, 0.0, clip as f32, 0.0]);
        prog.run(&dev);
        let gn = prog.read(&dev, 0)[0];
        let ls: Vec<f32> = (0..N_CTX_LOSS).map(|i| prog.read(&dev, 1 + i)[0]).collect();
        if !(gn.is_finite() && ls[0].is_finite()) {
            eprintln!("step {}: non-finite, skipped", step + 1);
            continue;
        }
        for i in 0..nc {
            prog.swap_input_output(k_p[i], o_p + i);
            prog.swap_input_output(k_m[i], o_m + i);
            prog.swap_input_output(k_v[i], o_v + i);
        }
        step += 1;
        let ms = ts.elapsed().as_secs_f64() * 1e3;
        writeln!(log, "{step}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{gn:.4}\t{lr:.3e}\t{ms:.0}", ls[0], ls[1], ls[2], ls[3])?;
        for (i, x) in ls.iter().enumerate() {
            acc[i] += *x as f64;
        }
        acc[N_CTX_LOSS] += gn as f64;
        acc[N_CTX_LOSS + 1] += ms;
        acc_n += 1;
        if step % 100 == 0 {
            let k = acc_n as f64;
            eprintln!("step {step:>5}  total {:.4}  dur {:.4}  pitch {:.4}  energy {:.4}  |g| {:.3}  lr {lr:.2e}  {:.0} ms/step", acc[0] / k, acc[1] / k, acc[2] / k, acc[3] / k, acc[4] / k, acc[5] / k);
            acc = [0.0; N_CTX_LOSS + 2];
            acc_n = 0;
            log.flush()?;
        }
        let stopping = stop.exists();
        if step % 500 == 0 || stopping {
            save(&prog, step, &ck_path)?;
        }
        if eval_every > 0 && step % eval_every == 0 {
            do_eval(&mut prog, step, &mut elog)?;
        }
        if stopping {
            std::fs::remove_file(&stop).ok();
            eprintln!("STOP: saved step {step}");
            return Ok(());
        }
    }
    save(&prog, step, &ck_path)?;
    if eval_every > 0 && step % eval_every != 0 {
        do_eval(&mut prog, step, &mut elog)?;
    }
    Ok(())
}

/// Context encoder for `synth`: base + ctx params, compiled per (n, m).
pub struct CtxEnc {
    prog: Program,
    map: HashMap<&'static str, usize>,
}

impl CtxEnc {
    pub fn new(dev: &Device, tc: &TConfig, params: &[Vec<f32>], n: usize, m: usize, r: usize) -> Self {
        let cc = CtxConfig::base();
        let eg = acoustic::ctx_enc_graph(tc, &cc, Dims { batch: 1, n, t: 64, r }, m);
        let ei = acoustic::assemble(&eg.b, &acoustic::ctx_specs(tc, &cc), params, None, &[]);
        let prog = Program::new(dev, &eg.b.g, &[eg.enc.h, eg.enc.ld, eg.enc.pitch, eg.enc.energy, eg.enc.spk_dec], &refs(&ei));
        Self { prog, map: input_map(&eg.b) }
    }
    pub fn encode(&mut self, dev: &Device, data: &[(&str, &[f32])]) -> [Vec<f32>; 5] {
        for (k, v) in data {
            self.prog.write_input(dev, self.map[k], v);
        }
        self.prog.run(dev);
        [0, 1, 2, 3, 4].map(|i| self.prog.read(dev, i))
    }
}

/// `*word*` marks → (clean text, marked word indices over whitespace words).
pub fn strip_emph(text: &str) -> (String, Vec<usize>) {
    let mut out = Vec::new();
    let mut idx = Vec::new();
    for (i, w) in text.split_whitespace().enumerate() {
        if w.contains('*') {
            idx.push(i);
        }
        out.push(w.replace('*', ""));
    }
    (out.join(" "), idx)
}
