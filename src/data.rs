//! Wav IO and the smoke dataset (random fixed-length crops).

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

use crate::dsp;
use crate::model::Rng;

pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
        }
    };
    let mono = raw.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect();
    Ok((mono, spec.sample_rate))
}

pub fn write_wav(path: &Path, x: &[f32], sr: u32) -> Result<()> {
    let spec = hound::WavSpec { channels: 1, sample_rate: sr, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &v in x {
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}

/// Minimal `.npy` reader: little-endian f32 / f64, C order, any rank.
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path)?;
    if &b[..6] != b"\x93NUMPY" {
        return Err(anyhow!("not an npy file"));
    }
    let (hlen, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let h = std::str::from_utf8(&b[off..off + hlen])?;
    if h.contains("'fortran_order': True") {
        return Err(anyhow!("fortran order unsupported"));
    }
    let shape_s = h.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).ok_or_else(|| anyhow!("no shape"))?;
    let shape: Vec<usize> = shape_s.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
    let data = &b[off + hlen..];
    let v = if h.contains("<f4") {
        data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    } else if h.contains("<f8") {
        data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect()
    } else {
        return Err(anyhow!("dtype unsupported: {h}"));
    };
    Ok((shape, v))
}

pub fn list_wavs(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(list_wavs(&p));
            } else if p.extension().is_some_and(|x| x == "wav") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

pub struct Dataset {
    pub clips: Vec<(PathBuf, Vec<f32>)>,
}

impl Dataset {
    /// Every wav under `dir` (resampling is not done: all must be at `sr`).
    /// Clips whose index is `≡ 0 mod holdout` go to the held-out split.
    pub fn load(dir: &Path, sr: u32, holdout: usize) -> Result<(Dataset, Dataset)> {
        let (mut tr, mut ho) = (Vec::new(), Vec::new());
        for (i, p) in list_wavs(dir).into_iter().enumerate() {
            let (x, r) = read_wav(&p)?;
            if r != sr {
                return Err(anyhow!("{}: {r} Hz, want {sr}", p.display()));
            }
            if holdout > 0 && i % holdout == 0 { ho.push((p, x)) } else { tr.push((p, x)) }
        }
        Ok((Dataset { clips: tr }, Dataset { clips: ho }))
    }

    pub fn seconds(&self, sr: usize) -> f64 {
        self.clips.iter().map(|c| c.1.len()).sum::<usize>() as f64 / sr as f64
    }

    /// `batch` random crops of `n` samples (clips shorter than `n` are
    /// zero-padded), returned as (reflect-padded by `pad`, raw), row-major.
    pub fn batch(&self, rng: &mut Rng, batch: usize, n: usize, pad: usize) -> (Vec<f32>, Vec<f32>) {
        crop_batch(self.clips.len(), |i, s, out| out.copy_from_slice(&self.clips[i].1[s..s + out.len()]), |i| self.clips[i].1.len(), rng, batch, n, pad)
    }

    /// Fixed held-out set from a corpus split: `n` utterances, round-robin
    /// over speakers (LibriTTS layout `split/speaker/chapter/utt.wav`) so
    /// many voices are covered; deterministic.
    pub fn heldout(dirs: &[PathBuf], sr: u32, n: usize) -> Result<Dataset> {
        let mut by_spk: std::collections::BTreeMap<String, Vec<PathBuf>> = Default::default();
        for d in dirs {
            for p in list_wavs(d) {
                by_spk.entry(speaker_of(&p)).or_default().push(p);
            }
        }
        let mut picks = Vec::new();
        let mut round = 0;
        while picks.len() < n && by_spk.values().any(|v| v.len() > round) {
            for v in by_spk.values() {
                // middle-length-ish pick per speaker: skip the very first utterances
                if let Some(p) = v.get(round * 7 % v.len().max(1)).filter(|_| v.len() > round) {
                    if picks.len() < n && !picks.contains(p) {
                        picks.push(p.clone());
                    }
                }
            }
            round += 1;
        }
        let mut clips = Vec::new();
        for p in picks {
            let (x, r) = read_wav(&p)?;
            if r != sr {
                return Err(anyhow!("{}: {r} Hz, want {sr}", p.display()));
            }
            clips.push((p, x));
        }
        Ok(Dataset { clips })
    }
}

/// Speaker id of a LibriTTS path (`.../speaker/chapter/utt.wav`), or the
/// parent directory name otherwise.
pub fn speaker_of(p: &Path) -> String {
    p.parent().and_then(|c| c.parent()).and_then(|s| s.file_name()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// A training corpus held as i16 (LibriTTS-R train-clean-100 is ~9 GB so,
/// vs ~19 GB as f32). Loaded with parallel readers.
pub struct Corpus {
    pub clips: Vec<(PathBuf, Vec<i16>)>,
    pub speakers: usize,
}

impl Corpus {
    /// Every wav under each of `dirs`, except speakers in `exclude`.
    pub fn load(dirs: &[PathBuf], sr: u32, exclude: &std::collections::HashSet<String>, min_samples: usize) -> Result<Corpus> {
        let mut paths: Vec<PathBuf> = dirs.iter().flat_map(|d| list_wavs(d)).filter(|p| !exclude.contains(&speaker_of(p))).collect();
        paths.sort();
        let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(12);
        let chunk = paths.len().div_ceil(nt).max(1);
        let parts: Vec<Result<Vec<(PathBuf, Vec<i16>)>>> = std::thread::scope(|s| {
            let hs: Vec<_> = paths
                .chunks(chunk)
                .map(|c| {
                    s.spawn(move || {
                        let mut out = Vec::new();
                        for p in c {
                            let mut r = hound::WavReader::open(p)?;
                            let spec = r.spec();
                            if spec.sample_rate != sr || spec.channels != 1 {
                                return Err(anyhow!("{}: {} Hz × {} ch, want {sr} mono", p.display(), spec.sample_rate, spec.channels));
                            }
                            let v: Vec<i16> = match (spec.sample_format, spec.bits_per_sample) {
                                (hound::SampleFormat::Int, 16) => r.samples::<i16>().collect::<Result<_, _>>()?,
                                _ => r.samples::<f32>().map(|x| x.map(|v| (v * 32767.0).clamp(-32768.0, 32767.0) as i16)).collect::<Result<_, _>>()?,
                            };
                            if v.len() >= min_samples {
                                out.push((p.clone(), v));
                            }
                        }
                        Ok(out)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut clips = Vec::new();
        for p in parts {
            clips.extend(p?);
        }
        let speakers = clips.iter().map(|(p, _)| speaker_of(p)).collect::<std::collections::HashSet<_>>().len();
        Ok(Corpus { clips, speakers })
    }

    pub fn seconds(&self, sr: usize) -> f64 {
        self.clips.iter().map(|c| c.1.len()).sum::<usize>() as f64 / sr as f64
    }

    pub fn batch(&self, rng: &mut Rng, batch: usize, n: usize, pad: usize) -> (Vec<f32>, Vec<f32>) {
        let (xp, x, _) = self.batch_meta(rng, batch, n, pad, 1);
        (xp, x)
    }

    /// `batch` plus per crop `(clip, start sample, gain)`; starts are
    /// multiples of `align` (the hop, so crop frame t is clip frame
    /// start/hop + t for per-frame labels).
    pub fn batch_meta(&self, rng: &mut Rng, batch: usize, n: usize, pad: usize, align: usize) -> (Vec<f32>, Vec<f32>, Vec<(usize, usize, f32)>) {
        crop_batch_meta(
            self.clips.len(),
            |i, s, out| {
                for (o, &v) in out.iter_mut().zip(&self.clips[i].1[s..]) {
                    *o = v as f32 / 32768.0;
                }
            },
            |i| self.clips[i].1.len(),
            rng,
            batch,
            n,
            pad,
            align,
        )
    }
}

/// Random crops: `read(clip, start, out)` fills `out` from `clip` at `start`.
fn crop_batch(
    n_clips: usize,
    read: impl Fn(usize, usize, &mut [f32]),
    len: impl Fn(usize) -> usize,
    rng: &mut Rng,
    batch: usize,
    n: usize,
    pad: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (xp, x, _) = crop_batch_meta(n_clips, read, len, rng, batch, n, pad, 1);
    (xp, x)
}

#[allow(clippy::too_many_arguments)]
fn crop_batch_meta(
    n_clips: usize,
    read: impl Fn(usize, usize, &mut [f32]),
    len: impl Fn(usize) -> usize,
    rng: &mut Rng,
    batch: usize,
    n: usize,
    pad: usize,
    align: usize,
) -> (Vec<f32>, Vec<f32>, Vec<(usize, usize, f32)>) {
    {
        let mut meta = Vec::with_capacity(batch);
        let mut xp = Vec::with_capacity(batch * (n + 2 * pad));
        let mut x = Vec::with_capacity(batch * n);
        for _ in 0..batch {
            let ci = rng.below(n_clips);
            let cl = len(ci);
            // re-draw near-silent crops: a silent target makes the
            // spectral-convergence denominator ~0 (its gradient overflows)
            let mut seg = vec![0.0f32; n];
            let mut start = 0;
            for _ in 0..20 {
                seg.iter_mut().for_each(|v| *v = 0.0);
                if cl > n {
                    let s = rng.below(cl - n) / align * align;
                    start = s;
                    read(ci, s, &mut seg);
                } else {
                    read(ci, 0, &mut seg[..cl]);
                }
                let rms = (seg.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
                if rms > 1e-3 {
                    break;
                }
            }
            // random gain, like Vocos' training augmentation
            let peak = seg.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-4);
            let gain = (0.3 + 0.6 * rng.uniform() as f32) / peak;
            for v in &mut seg {
                *v *= gain.min(30.0);
            }
            meta.push((ci, start, gain.min(30.0)));
            xp.extend(dsp::reflect_pad(&seg, pad));
            x.extend(seg);
        }
        (xp, x, meta)
    }
}
