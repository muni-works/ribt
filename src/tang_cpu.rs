//! CPU runtime for tensor-IR programs, for a robot with gcc and no usable GPU
//! path (the K1). Same idea as `tang_gpu::cpu::CpuProgram` (`to_c`, the
//! system compiler, `dlopen`), plus the two things a robot needs:
//!
//! - **a persistent build cache**: the shared object is kept under
//!   `$TANG_CPU_CACHE` (default `~/.cache/tang-cpu`), named by a hash of the
//!   C source and the flags, so a bucket compiled once (seconds of gcc) loads
//!   in milliseconds on every later run. The streaming TTS compiles one
//!   program per shape bucket, so a cold run compiles on the hot path.
//! - **threads**: with `TANG_THREADS=N` (N > 1) each kernel's outermost
//!   non-trivial *point* loop (`p*`, `q`, `j*`: output points, never a
//!   reduction `r*`) gets `#pragma omp parallel for`, and the unit is built
//!   with `-fopenmp`. Only a perfectly nested run of point loops at the very
//!   top of a kernel is considered, so every iteration writes disjoint
//!   outputs and declares its accumulators inside the loop.
//!
//! Shared by `ribt` and `hear` (hear includes this file by `#[path]`), so one
//! copy of the cache/threading rule exists. Upstream target: tang-gpu.

use std::path::{Path, PathBuf};
use std::process::Command;

use tang_compile::tensor::c::CProgram;
use tang_compile::tensor::{precision, Format, TensorGraph, TensorId};

type RunFn = unsafe extern "C" fn();
type SetThreadsFn = unsafe extern "C" fn(i32);

thread_local! {
    /// OpenMP team size wanted by this OS thread (0 = libgomp's default)
    static WANT: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    static APPLIED: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
}

/// Team size for programs run *from the calling thread* (OpenMP's
/// nthreads ICV is per thread, so e.g. ASR and TTS threads can split the
/// cores). Only meaningful with `TANG_THREADS` > 1.
pub fn set_threads(n: usize) {
    WANT.with(|w| w.set(n as i32));
}
type PtrFn = unsafe extern "C" fn(i32) -> *mut core::ffi::c_void;

pub struct CpuProgram {
    pub program: CProgram,
    _lib: libloading::Library,
    run: RunFn,
    input: PtrFn,
    output: PtrFn,
    set_threads: Option<SetThreadsFn>,
}

fn threads() -> usize {
    std::env::var("TANG_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(1)
}

fn cflags() -> Vec<String> {
    let mut f: Vec<String> = match std::env::var("TANG_CFLAGS") {
        Ok(f) => f.split_whitespace().map(str::to_string).collect(),
        Err(_) => {
            let cpu = if cfg!(target_arch = "aarch64") { "-mcpu=native" } else { "-march=native" };
            ["-O3", cpu, "-ffp-contract=fast", "-fno-strict-aliasing"].iter().map(|s| s.to_string()).collect()
        }
    };
    if threads() > 1 {
        f.push("-fopenmp".into());
    }
    f
}

/// `for (int X = 0; X < N; ...` → (X, N)
fn loop_head(line: &str) -> Option<(&str, usize)> {
    let s = line.trim_start().strip_prefix("for (int ")?;
    let (var, rest) = s.split_once(" = 0; ")?;
    let rest = rest.strip_prefix(var)?.strip_prefix(" < ")?;
    let n: usize = rest.split(';').next()?.trim().parse().ok()?;
    Some((var, n))
}

fn point_var(v: &str) -> bool {
    let tail = |p: &str| v.strip_prefix(p).is_some_and(|t| t.chars().all(|c| c.is_ascii_digit()));
    v == "q" || tail("p") || tail("j")
}

/// Insert `#pragma omp parallel for` per kernel (see module doc). Returns
/// the new source and how many kernels were parallelised.
pub fn parallelise(src: &str, min_iters: usize) -> (String, usize) {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = String::with_capacity(src.len() + 4096);
    let mut n = 0;
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        out.push_str(l);
        out.push('\n');
        i += 1;
        if !(l.contains("TANG_KERNEL void") && l.trim_end().ends_with('{')) {
            continue;
        }
        // leading, perfectly nested point loops
        let mut j = i;
        let mut pick = None;
        while j < lines.len() {
            let Some((v, it)) = loop_head(lines[j]) else { break };
            if !point_var(v) {
                break;
            }
            if it >= min_iters {
                pick = Some(j);
                break;
            }
            j += 1;
        }
        if let Some(p) = pick {
            for l in &lines[i..p] {
                out.push_str(l);
                out.push('\n');
            }
            out.push_str("#pragma omp parallel for schedule(static)\n");
            n += 1;
            i = p;
        }
    }
    (out, n)
}

fn cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("TANG_CPU_CACHE") {
        return PathBuf::from(d);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    home.join(".cache/tang-cpu")
}

/// Build (or find in the cache) the shared object for `source`.
fn build(source: &str) -> Result<PathBuf, String> {
    use std::hash::{Hash, Hasher};
    let flags = cflags();
    let source = if threads() > 1 { parallelise(source, 2).0 } else { source.to_string() };
    let mut h = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut h);
    flags.hash(&mut h);
    let key = format!("{:016x}", h.finish());
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let so = dir.join(format!("{key}.so"));
    if so.is_file() {
        return Ok(so);
    }
    let tmp = dir.join(format!("{key}.{}", std::process::id()));
    let src = tmp.with_extension("c");
    std::fs::write(&src, &source).map_err(|e| e.to_string())?;
    let cc = std::env::var("TANG_CC").unwrap_or_else(|_| "cc".into());
    let o = Command::new(&cc)
        .args(&flags)
        .args(["-shared", "-fPIC", "-o"])
        .arg(&tmp)
        .arg(&src)
        .arg("-lm")
        .output()
        .map_err(|e| format!("{cc}: {e}"))?;
    let _ = std::fs::remove_file(&src);
    if !o.status.success() {
        return Err(format!("{cc} failed:\n{}", String::from_utf8_lossy(&o.stderr)));
    }
    std::fs::rename(&tmp, &so).map_err(|e| e.to_string())?;
    Ok(so)
}

impl CpuProgram {
    pub fn compile(g: &TensorGraph, outputs: &[TensorId], inputs: &[&[f32]]) -> Result<Self, String> {
        let p = Self::load(g.to_c(outputs))?;
        for (k, x) in inputs.iter().enumerate() {
            p.write_input(k, x);
        }
        Ok(p)
    }

    pub fn load(program: CProgram) -> Result<Self, String> {
        let so = build(&program.source)?;
        Self::open(program, &so)
    }

    fn open(program: CProgram, so: &Path) -> Result<Self, String> {
        // SAFETY: the library is a unit generated by to_c; its entry points
        // have the signatures declared above.
        unsafe {
            let lib = libloading::Library::new(so).map_err(|e| e.to_string())?;
            let run = *lib.get::<RunFn>(b"tang_run\0").map_err(|e| e.to_string())?;
            let input = *lib.get::<PtrFn>(b"tang_input\0").map_err(|e| e.to_string())?;
            let output = *lib.get::<PtrFn>(b"tang_output\0").map_err(|e| e.to_string())?;
            let set_threads = lib.get::<SetThreadsFn>(b"omp_set_num_threads\0").ok().map(|f| *f);
            Ok(Self { program, _lib: lib, run, input, output, set_threads })
        }
    }

    fn input_spec(&self, k: usize) -> (usize, Format) {
        let b = self.program.program.buffers.iter().find(|b| b.input == Some(k)).expect("graph input has a buffer");
        (b.len, b.format)
    }

    pub fn write_input(&self, k: usize, data: &[f32]) {
        let (len, format) = self.input_spec(k);
        assert_eq!(data.len(), len, "input {k} length");
        // SAFETY: tang_input(k) points at a static array of `len` elements of `format`.
        unsafe {
            let p = (self.input)(k as i32);
            match format {
                Format::F32 => std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut f32, len),
                Format::F16 => {
                    let d = std::slice::from_raw_parts_mut(p as *mut u16, len);
                    for (o, &x) in d.iter_mut().zip(data) {
                        *o = precision::f16_bits(x as f64);
                    }
                }
                Format::BF16 => {
                    let d = std::slice::from_raw_parts_mut(p as *mut u16, len);
                    for (o, &x) in d.iter_mut().zip(data) {
                        *o = precision::bf16_bits(x as f64);
                    }
                }
            }
        }
    }

    pub fn run(&self) {
        if let Some(f) = self.set_threads {
            let w = WANT.with(|w| w.get());
            if w > 0 && APPLIED.with(|a| a.get()) != w {
                // SAFETY: libgomp's omp_set_num_threads(int).
                unsafe { f(w) };
                APPLIED.with(|a| a.set(w));
            }
        }
        // SAFETY: tang_run touches only the library's own buffers.
        unsafe { (self.run)() }
    }

    pub fn read_output(&self, i: usize) -> Vec<f32> {
        let b = &self.program.program.buffers[self.program.outputs[i]];
        // SAFETY: as for write_input.
        unsafe {
            let p = (self.output)(i as i32);
            match b.format {
                Format::F32 => std::slice::from_raw_parts(p as *const f32, b.len).to_vec(),
                Format::F16 => std::slice::from_raw_parts(p as *const u16, b.len).iter().map(|h| precision::f16_from_bits(*h) as f32).collect(),
                Format::BF16 => std::slice::from_raw_parts(p as *const u16, b.len).iter().map(|h| f32::from_bits((*h as u32) << 16)).collect(),
            }
        }
    }

    /// Mean seconds per run over `iters` after one warm-up.
    pub fn time(&self, iters: usize) -> f64 {
        self.run();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            self.run();
        }
        t.elapsed().as_secs_f64() / iters.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pragma_goes_on_first_nontrivial_point_loop() {
        let src = "static TANG_KERNEL void k0(float* restrict b0) {\n    for (int p0 = 0; p0 < 1; p0++) {\n        for (int q = 0; q < 384; q += 4) {\n            for (int r0 = 0; r0 < 9; r0++) {\n";
        let (s, n) = parallelise(src, 2);
        assert_eq!(n, 1);
        let l: Vec<&str> = s.lines().collect();
        assert!(l[2].starts_with("#pragma omp"), "{s}");
        assert!(l[3].contains("int q"));
    }

    #[test]
    fn reductions_and_trivial_kernels_are_left_alone() {
        let src = "static TANG_KERNEL void k0(float* restrict b0) {\n    for (int r0 = 0; r0 < 100; r0++) {\nstatic TANG_KERNEL void k1(float* restrict b0) {\n    for (int p0 = 0; p0 < 1; p0++) {\n        float acc = 0;\n";
        assert_eq!(parallelise(src, 2).1, 0);
    }
}
