//! Checkpoints as safetensors: parameters (`name`) plus optional Adam state
//! (`adam.m.name`, `adam.v.name`) and the step (`meta.step`, one f32).

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use crate::model::{Config, ParamSpec};

pub struct Checkpoint {
    pub params: Vec<Vec<f32>>,
    pub adam: Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
    pub step: usize,
}

fn bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

pub fn save(path: &Path, cfg: &Config, ck: &Checkpoint) -> Result<()> {
    save_specs(path, &cfg.params(), ck)
}

pub fn save_specs(path: &Path, specs: &[ParamSpec], ck: &Checkpoint) -> Result<()> {
    let step = [ck.step as f32];
    let mut views: Vec<(String, TensorView)> = Vec::new();
    for (s, p) in specs.iter().zip(&ck.params) {
        views.push((s.name.clone(), TensorView::new(Dtype::F32, s.shape.clone(), bytes(p))?));
    }
    if let Some((m, v)) = &ck.adam {
        for (s, x) in specs.iter().zip(m) {
            views.push((format!("adam.m.{}", s.name), TensorView::new(Dtype::F32, s.shape.clone(), bytes(x))?));
        }
        for (s, x) in specs.iter().zip(v) {
            views.push((format!("adam.v.{}", s.name), TensorView::new(Dtype::F32, s.shape.clone(), bytes(x))?));
        }
    }
    views.push(("meta.step".into(), TensorView::new(Dtype::F32, vec![1], bytes(&step))?));
    let tmp = path.with_extension("tmp");
    safetensors::serialize_to_file(views, &None, &tmp)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn load(path: &Path, cfg: &Config) -> Result<Checkpoint> {
    load_specs(path, &cfg.params(), false)
}

/// Number of f32 values of tensor `name`, if present.
pub fn numel(path: &Path, name: &str) -> Option<usize> {
    let b = std::fs::read(path).ok()?;
    let st = SafeTensors::deserialize(&b).ok()?;
    st.tensor(name).ok().map(|t| t.data().len() / 4)
}

/// Does the checkpoint hold tensor `name`?
pub fn has(path: &Path, name: &str) -> bool {
    std::fs::read(path).ok().is_some_and(|b| SafeTensors::deserialize(&b).is_ok_and(|st| st.tensor(name).is_ok()))
}

/// Load `specs` from `path`. With `init_missing`, parameters absent from
/// the file get their spec's init (seed 0) and the Adam state is dropped
/// unless complete (warm-starting V1.1 from a V1 checkpoint).
pub fn load_specs(path: &Path, specs: &[ParamSpec], init_missing: bool) -> Result<Checkpoint> {
    let buf = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&buf)?;
    let get = |name: &str| -> Option<Vec<f32>> {
        let t = st.tensor(name).ok()?;
        Some(t.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    };
    let mut params = Vec::new();
    let mut rng = crate::model::Rng::new(0);
    for s in specs {
        let p = match get(&s.name) {
            Some(p) => p,
            None if init_missing => {
                eprintln!("ckpt {}: {} not found, initialized", path.display(), s.name);
                s.init_data(&mut rng)
            }
            None => return Err(anyhow!("missing {}", s.name)),
        };
        if p.len() != s.numel() {
            return Err(anyhow!("{}: {} values, want {}", s.name, p.len(), s.numel()));
        }
        params.push(p);
    }
    let m: Option<Vec<_>> = specs.iter().map(|s| get(&format!("adam.m.{}", s.name))).collect();
    let v: Option<Vec<_>> = specs.iter().map(|s| get(&format!("adam.v.{}", s.name))).collect();
    let step = get("meta.step").map_or(0, |s| s[0] as usize);
    Ok(Checkpoint { params, adam: m.zip(v), step })
}
