//! `ribt-tts voc-pairs`: predicted-mel / real-audio pairs for the vocoder
//! fine-tune (the ribt paper).
//!
//!   ribt-tts voc-pairs --run V2A [--ckpt step…] --phon A.phon.tsv[,B…] --durs D --root R --out DIR
//!                      [--batch 16] [--t 1024] [--pe gt|pred] [--flow STUDENT.safetensors --nfe 1 --temp 0.5]
//!
//! Teacher-forced: the aligner's (ground-truth) durations, so frame f of the
//! predicted mel is frame f of the recording. Token pitch/energy are the
//! recording's (`--pe gt`, default) or the predictors' (`--pe pred`). With
//! `--flow`, the rung-2 shortcut student (1 step) replaces rung 1's mel.
//! F0 is the rung-1 head's (Hz, 0 unvoiced). Writes `DIR/index.tsv` (id,
//! frames, frame offset), `DIR/mel.i16` (per utt `[frames, n_mels]`,
//! log-mel × 1000, little-endian) and `DIR/f0.f32`. Each phon file is loaded
//! and written in turn (memory); utts longer than `--t` are skipped.

use super::*;
use ribt::flow::{FConfig, FDims};

pub fn voc_pairs(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let run = PathBuf::from(a.need("--run")?);
    let (params, nm, step) = load_run(&run, &tc, a.opt("--ckpt"))?;
    let root = PathBuf::from(a.need("--root")?);
    let durs = read_durs(&a.need("--durs")?)?;
    let out = PathBuf::from(a.need("--out")?);
    std::fs::create_dir_all(&out)?;
    let batch: usize = a.get("--batch", 16);
    let (nmax, tmax): (usize, usize) = (a.get("--n", 192), a.get("--t", 1024));
    let pred_pe = a.opt("--pe").as_deref() == Some("pred");
    let fc = FConfig::by_name(&a.opt("--fcfg").unwrap_or_else(|| "base".into())).ok_or_else(|| anyhow!("fcfg"))?;
    let flow = match a.opt("--flow") {
        Some(f) => {
            let p = PathBuf::from(f);
            Some((ckpt::load_specs(&p, &fc.params(), false)?.params, flow_cmd::FNorm::load(&p)?))
        }
        None => None,
    };
    let (nfe, temp): (usize, f64) = (a.get("--nfe", 1), a.get("--temp", 0.5));
    let dev = Device::new();
    let d = Dims { batch, n: nmax, t: tmax, r: REF_T };
    let m = tc.n_mels;
    let mut tts = Tts::new(&dev, &tc, &params, d);
    let mut fe_ref = Frontend::new(&dev, batch, REF_T);
    let mut vel = flow.as_ref().map(|(fp, _)| flow_cmd::Vel::new(&dev, &fc, fp, FDims { batch, t: tmax }));
    let mut rng = Rng::new(4242);
    let mut idx = std::fs::File::create(out.join("index.tsv"))?;
    writeln!(idx, "id\tframes\toffset")?;
    let mut fmel = std::io::BufWriter::new(std::fs::File::create(out.join("mel.i16"))?);
    let mut ff0 = std::io::BufWriter::new(std::fs::File::create(out.join("f0.f32"))?);
    // hidden-decoder rung 1: also the decoder's last hidden layer, `hid.f32` per utt `[frames, dd]`
    let mut fhid = match a.0.iter().any(|x| x == "--hid") {
        true => Some(std::io::BufWriter::new(std::fs::File::create(out.join("hid.f32"))?)),
        false => None,
    };
    std::fs::write(out.join("source.txt"), format!("run {} step {step} pe {} flow {:?} nfe {nfe} temp {temp}\n", run.display(), if pred_pe { "pred" } else { "gt" }, a.opt("--flow")))?;
    let (mut off, mut n_utt) = (0usize, 0usize);
    let t0 = Instant::now();
    for phon in a.need("--phon")?.split(',').filter(|s| !s.is_empty()) {
        let data = load(phon, &durs, &root, &|n, t| n <= nmax && t <= tmax && t >= 16)?;
        let spk = by_speaker(&data);
        eprintln!("{phon}: {} utts ({:.0} s)", data.len(), t0.elapsed().as_secs_f64());
        for c in data.chunks(batch) {
            let mut us: Vec<&Utt> = c.iter().collect();
            let real = us.len();
            while us.len() < batch {
                us.push(us[0]);
            }
            let mut rclips = Vec::new();
            let mut rlens = Vec::new();
            for u in &us {
                let (r, o, len) = pick_ref(&mut rng, u, &data, &spk);
                rclips.push(r.audio_f32(o, len));
                rlens.push(len);
            }
            let refmel = fe_ref.run(&dev, &rclips);
            let hb = host_batch(&tc, d, &us, &nm, &rlens, &vec![0; batch]);
            let [h, _ld, pp, ep, spkv] = tts.encode(&dev, hb.get("onehot"), hb.get("tmask"), &refmel, hb.get("rw"));
            let (p, e) = if pred_pe { (pp.as_slice(), ep.as_slice()) } else { (hb.get("p_tok"), hb.get("e_tok")) };
            let (mut mel, f0, hid) = tts.decode(&dev, &h, p, e, hb.get("tmask"), hb.get("align"), hb.get("fmask"), &spkv);
            if let (Some(v), Some((_, fnm))) = (vel.as_mut(), flow.as_ref()) {
                let fm = hb.get("fmask");
                let lens: Vec<usize> = us.iter().map(|u| u.frames()).collect();
                let ab = ribt::flow::attn_bias(&lens, tmax, None);
                let mun = fnm.norm(&mel, fm, batch, tmax);
                let x0 = flow_cmd::noise(&mut rng, fm, batch, m, tmax, temp);
                v.set_cond(&dev, &mun, &hid, fm, &ab);
                let (x1, _) = v.sample(&dev, &x0, nfe, true);
                mel = fnm.denorm(&x1);
            }
            for (b, u) in us.iter().enumerate().take(real) {
                let fr = u.frames();
                for f in 0..fr {
                    for cc in 0..m {
                        let v = (mel[(b * m + cc) * tmax + f] * 1000.0).round().clamp(-32768.0, 32767.0) as i16;
                        fmel.write_all(&v.to_le_bytes())?;
                    }
                    let base = b * 2 * tmax;
                    let hz = if f0[base + tmax + f] > 0.0 { ((f0[base + f] as f64) * nm.f0_sd + nm.f0_mu).exp() as f32 } else { 0.0 };
                    ff0.write_all(&hz.to_le_bytes())?;
                    if let Some(fh) = fhid.as_mut() {
                        for cc in 0..tc.dd {
                            fh.write_all(&hid[(b * tc.dd + cc) * tmax + f].to_le_bytes())?;
                        }
                    }
                }
                writeln!(idx, "{}\t{fr}\t{off}", u.id)?;
                off += fr;
                n_utt += 1;
            }
            if n_utt % (batch * 100) < batch {
                eprintln!("{n_utt} utts, {off} frames ({:.0} s)", t0.elapsed().as_secs_f64());
            }
        }
    }
    fmel.flush()?;
    ff0.flush()?;
    if let Some(fh) = fhid.as_mut() {
        fh.flush()?;
    }
    eprintln!("voc-pairs: {n_utt} utts, {off} frames -> {} ({:.0} s)", out.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

/// Hidden-decoder rung 1: a hidden-input G (`--cfg v1f0hid`) whose step-0
/// function is the mel G on the acoustic decoder's mel, exactly.
///
///   ribt-tts hid-init --run ACRUN --ckpt acoustic_af_bon.safetensors --voc MEL_G.safetensors --out G_HID.safetensors
///
/// The decoder's mel is `W_m·h + b_m`, and G's first layer is the k-tap conv
/// `W_e`. On G's input rows (h, 1, ln gain): `W'[:, h] = W_e·W_m`,
/// `W'[:, 1] = W_e·b_m`, `W'[:, gain] = W_e·1`, so `conv(W', [h; 1; g]) =
/// conv(W_e, mel + g)` with the same zero padding at the edges (the ones row
/// is zero there too). Every other tensor is copied.
pub fn hid_init(a: &Args) -> Result<()> {
    let tc = a.cfg()?;
    let run = PathBuf::from(a.need("--run")?);
    let (ap, _, step) = load_run(&run, &tc, a.opt("--ckpt"))?;
    let spec = tc.params();
    let at = |n: &str| spec.iter().position(|s| s.name == n).ok_or_else(|| anyhow!("acoustic has no {n}"));
    let (wm, bm) = (&ap[at("mel.w")?], &ap[at("mel.b")?]);
    let (mc, hc) = (Config::by_name("v1f0").unwrap(), Config::by_name("v1f0hid").unwrap());
    let (m, dd) = (mc.n_mels, tc.dd);
    assert_eq!(hc.hid, dd, "v1f0hid expects the decoder width");
    let src = PathBuf::from(a.need("--voc")?);
    let vp = ckpt::load(&src, &mc)?.params;
    let (d, k, ci) = (mc.dim, mc.embed_k, hc.in_ch());
    let we = &vp[0];
    let mut w = vec![0.0f32; d * ci * k];
    for o in 0..d {
        for kk in 0..k {
            let e = |mm: usize| we[(o * m + mm) * k + kk] as f64;
            for c in 0..dd {
                w[(o * ci + c) * k + kk] = (0..m).map(|mm| e(mm) * wm[mm * dd + c] as f64).sum::<f64>() as f32;
            }
            w[(o * ci + dd) * k + kk] = (0..m).map(|mm| e(mm) * bm[mm] as f64).sum::<f64>() as f32;
            w[(o * ci + dd + 1) * k + kk] = (0..m).map(e).sum::<f64>() as f32;
        }
    }
    let mut params = vp;
    params[0] = w;
    let out = PathBuf::from(a.need("--out")?);
    ckpt::save_specs(&out, &hc.params(), &ckpt::Checkpoint { params, adam: None, step: 0 })?;
    eprintln!("hid-init: {} (acoustic step {step}) + {} -> {} ({} G params)", run.display(), src.display(), out.display(), hc.n_params());
    Ok(())
}
