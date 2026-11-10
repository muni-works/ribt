//! Compile a built graph (passes → schedule → kernels) and run it on a GPU.
//! Inputs are addressed by their graph input number; outputs by the ids the
//! optimizer returns.
//!
//! One backend per build, chosen by cargo feature:
//! - `metal` (default): MSL via tang-bench's `ir_msl` runtime, vendored in
//!   `tang_msl` (macOS).
//! - `cuda`: `TensorGraph::to_cuda` (wmma TF32 tiles + WGSL→CUDA transpiled
//!   kernels) on a small cudarc runtime here. tang-bench's `ir_cuda` keeps
//!   its buffers private (no input write / swap), so it is not reused.
//! - `wgpu`: `to_wgsl` on tang-bench's `ir_gpu` runtime (Vulkan on Linux).
//!
//! All three expose the same `Device` / `Program` API.

use std::time::Instant;

use tang_compile::tensor::{Op, TensorGraph, TensorId};

#[cfg(any(
    all(feature = "metal", feature = "cuda"),
    all(feature = "metal", feature = "wgpu"),
    all(feature = "cuda", feature = "wgpu"),
    all(feature = "opencl", any(feature = "metal", feature = "cuda", feature = "wgpu", feature = "cpu")),
    all(feature = "cpu", any(feature = "metal", feature = "cuda", feature = "wgpu"))
))]
compile_error!("enable exactly one backend: --no-default-features --features cuda|wgpu|metal|cpu|opencl");
#[cfg(not(any(feature = "metal", feature = "cuda", feature = "wgpu", feature = "cpu", feature = "opencl")))]
compile_error!("enable a backend feature: metal, cuda, wgpu, cpu or opencl");

/// The pass pipeline tang's own benches use, then `schedule_auto`.
///
/// A graph with scans gets every live scan's inputs appended to its outputs
/// (after the given ones, so output indices stay as given). At tang
/// 73a78dcb a scan reads its inputs as buffers in the layout it was built
/// with, but `relayout` permutes a scan input's dims and the schedule
/// inlines a map read only by a scan; an output is exempt from both.
pub fn optimize(g: &TensorGraph, outs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let pinned = g.nodes().iter().any(|n| matches!(n.op, Op::Scan(_))).then(|| {
        let (g, mut outs) = g.eliminate_dead(outs);
        for i in 0..g.nodes().len() {
            for t in g.scan_at(TensorId::from_index(i)).map(|s| s.inputs.clone()).into_iter().flatten() {
                if !outs.contains(&t) {
                    outs.push(t);
                }
            }
        }
        (g, outs)
    });
    let (g, outs) = pinned.as_ref().map_or((g, outs), |(g, o)| (g, &o[..]));
    let (g, outs) = g.substitute_maps(outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    let (mut g, outs) = g.merge_reduce_dims(&outs);
    g.schedule_auto(&outs);
    #[cfg(feature = "opencl")]
    crate::tang_cl::adreno_schedule(&mut g);
    (g, outs)
}

pub const BACKEND: &str = if cfg!(feature = "metal") {
    "metal"
} else if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "cpu") {
    "cpu"
} else if cfg!(feature = "opencl") {
    "opencl"
} else {
    "wgpu"
};

// ---------------------------------------------------------------------------
// metal

#[cfg(feature = "metal")]
mod backend {
    use crate::tang_msl::{Executable, MslDevice};
    use tang_compile::tensor::msl::FragmentTarget;
    use tang_compile::tensor::{precision, Format, TensorGraph, TensorId};

    pub struct Device(pub MslDevice);

    impl Device {
        pub fn new() -> Self {
            Device(MslDevice::new().expect("no Metal device"))
        }
        pub fn name(&self) -> String {
            "Metal".into()
        }
    }

    pub struct Exe(pub Executable);

    impl Exe {
        pub fn new(dev: &Device, graph: &TensorGraph, outs: &[TensorId], inputs: &[&[f32]]) -> Self {
            // no nested contractions: at tang 73a78dcb a contraction nested in a
            // tile is staged outside its own reduce loop (undeclared `q1_0` in
            // the vocoder train step's MSL)
            Exe(Executable::new(&dev.0, graph.to_msl_with(outs, &FragmentTarget::APPLE, false), inputs))
        }
        pub fn kernels(&self) -> usize {
            self.0.program.kernels.len()
        }
        pub fn kernel_buffers(&self) -> Vec<usize> {
            self.0.program.kernels.iter().map(|k| k.reads.len() + 1).collect()
        }
        pub fn input_slot(&self, k: usize) -> Option<usize> {
            self.0.program.buffers.iter().position(|b| b.input == Some(k))
        }
        pub fn slot_of(&self, t: TensorId) -> usize {
            let t = self.0.program.storage(t);
            self.0.program.buffers.iter().position(|b| b.node == t).unwrap()
        }
        pub fn spec(&self, i: usize) -> (usize, Format) {
            let b = &self.0.program.buffers[i];
            (b.len, b.format)
        }
        pub fn write_slot(&self, _dev: &Device, i: usize, data: &[f32]) {
            let (len, format) = self.spec(i);
            assert_eq!(len, data.len());
            let p = self.0.buffers[i].contents();
            unsafe {
                match format {
                    Format::F32 => std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut f32, data.len()),
                    Format::F16 => {
                        let d = std::slice::from_raw_parts_mut(p as *mut u16, data.len());
                        for (o, &x) in d.iter_mut().zip(data) {
                            *o = precision::f16_bits(x as f64);
                        }
                    }
                    Format::BF16 => {
                        let d = std::slice::from_raw_parts_mut(p as *mut u16, data.len());
                        for (o, &x) in d.iter_mut().zip(data) {
                            *o = precision::bf16_bits(x as f64);
                        }
                    }
                    f => unimplemented!("metal: writing a {f:?} buffer"),
                }
            }
        }
        pub fn read_slot(&self, _dev: &Device, i: usize) -> Vec<f32> {
            let (len, format) = self.spec(i);
            assert_eq!(format, Format::F32);
            let p = self.0.buffers[i].contents();
            unsafe { std::slice::from_raw_parts(p as *const f32, len).to_vec() }
        }
        pub fn read(&self, _dev: &Device, t: TensorId) -> Vec<f32> {
            self.0.read(t)
        }
        pub fn swap(&mut self, a: usize, b: usize) {
            self.0.buffers.swap(a, b);
        }
        pub fn run(&mut self, dev: &Device) {
            self.0.run(&dev.0);
        }
        pub fn time(&mut self, dev: &Device, iters: usize) -> f64 {
            self.0.time(&dev.0, iters)
        }
    }
}

// ---------------------------------------------------------------------------
// source fixes for transpiled kernels (WGSL -> CUDA / OpenCL C)

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub mod fix {
use tang_compile::tensor::{TensorGraph, TensorId};

/// Workaround for a `wgsl_to_cuda` gap (tang 1e076822): a typed WGSL
/// binding `let u1: u32 = 0u;` is emitted as `auto u1: u32 = 0u;`,
/// which nvrtc rejects. Rewrite to `unsigned u1 = 0u;`. Repro:
/// `examples/ir_gaps.rs cuda-typed-let`.
pub fn fix_typed_auto(src: &str) -> String {
    src.lines()
        .map(|l| {
            let t = l.trim_start();
            let indent = &l[..l.len() - t.len()];
            let Some(rest) = t.strip_prefix("auto ") else { return l.to_string() };
            let Some((name, rhs)) = rest.split_once(": ") else { return l.to_string() };
            let Some((ty, init)) = rhs.split_once(" = ") else { return l.to_string() };
            let ty = match ty {
                "u32" => "unsigned",
                "f32" => "float",
                "i32" => "int",
                "bool" => "bool",
                _ => return l.to_string(),
            };
            format!("{indent}{ty} {name} = {init}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Workaround for a third `wgsl_to_cuda` gap (§20, gap 12): a tiled
/// kernel stages a guarded gather as `if (live) { stage_k[i] = …; }`.
/// WGSL zero-initializes workgroup memory, so a skipped slot reads 0;
/// CUDA `__shared__` is not initialized, so the slot keeps whatever the
/// previous block left. A conv's left halo at the very first element
/// (`a = -1`) fails the `(unsigned)a < len` guard: the rung-1 conv
/// backward read garbage (~1e-2 rel, then inf in the grad norm). The
/// patch adds `else { stage_k[i] = 0.0f; }` (the slot's index is
/// in range whatever the guard; later chunks never read rows a failed
/// guard skipped, so 0 matches WGSL for every slot that is read).
pub fn fix_stage_guards(src: &str) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len() + 8);
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        if i + 2 < lines.len() && l.trim_start().starts_with("if (") && l.trim_end().ends_with('{') && lines[i + 2].trim() == "}" {
            let st = lines[i + 1].trim_start();
            if st.starts_with("stage_") {
                if let Some(eq) = st.find("] = ") {
                    let indent = &lines[i + 2][..lines[i + 2].len() - lines[i + 2].trim_start().len()];
                    out.push(l.to_string());
                    out.push(lines[i + 1].to_string());
                    out.push(format!("{indent}}} else {{ {}] = 0.0f; }}", &st[..eq]));
                    i += 3;
                    continue;
                }
            }
        }
        out.push(l.to_string());
        i += 1;
    }
    let mut r = out.join("\n");
    if src.ends_with('\n') {
        r.push('\n');
    }
    r
}

/// Workaround for a second `wgsl_to_cuda` gap: guarded gathers in tiled
/// kernels bound-check with WGSL's `arrayLength(&bJ)` (tensor_tile.rs
/// `buffer_len_expr`), which CUDA lacks. The length is static, as the
/// MSL dialect already emits. `bJ` is the kernel's J-th read binding
/// (`reads[J]`), NOT graph node J: reading node J's shape (the original
/// patch) bounded the tiled attention `p@v` gather by an unrelated
/// buffer's length, skipped most staging loads and left shared memory
/// uninitialized — the whole rung-1 encoder was wrong on CUDA (§20, gap
/// 11). Repros: `examples/ir_gaps.rs cuda-array-length`, `cuda-mha 7`.
pub fn fix_array_length(src: &str, g: &TensorGraph, reads: &[TensorId]) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find("arrayLength(&b") {
        out.push_str(&rest[..i]);
        let tail = &rest[i + "arrayLength(&b".len()..];
        let end = tail.find(')').expect("arrayLength )");
        let j: usize = tail[..end].parse().expect("arrayLength binding");
        let node = reads.get(j).copied().expect("arrayLength binding out of range");
        let n: usize = g.nodes()[node.index()].shape.iter().map(|d| d.n()).product();
        out.push_str(&format!("{n}u"));
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    out
}

}

// ---------------------------------------------------------------------------
// cuda

#[cfg(feature = "cuda")]
mod backend {
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
    use tang_compile::tensor::wgsl::WgslProgram;
    use tang_compile::tensor::{Format, TensorGraph, TensorId};
    use super::fix::{fix_array_length, fix_stage_guards, fix_typed_auto};

    pub struct Device {
        pub ctx: Arc<CudaContext>,
        pub stream: Arc<CudaStream>,
    }

    impl Device {
        pub fn new() -> Self {
            let ord: usize = std::env::var("RIBT_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let ctx = CudaContext::new(ord).expect("no CUDA device");
            let stream = ctx.default_stream();
            Device { ctx, stream }
        }
        pub fn name(&self) -> String {
            format!("CUDA:{}", self.ctx.name().unwrap_or_default())
        }
    }

    /// `#define TANG_SMEM n` in a kernel's source: its dynamic shared
    /// memory in bytes (0 when absent).
    fn dynamic_smem(src: &str) -> u32 {
        src.lines().find_map(|l| l.strip_prefix("#define TANG_SMEM ")).and_then(|v| v.trim().parse().ok()).unwrap_or(0)
    }

    struct Launch {
        f: CudaFunction,
        cfg: LaunchConfig,
        reads: Vec<usize>,
        out: usize,
        points: u32,
    }

    pub struct Exe {
        pub program: WgslProgram,
        buffers: Vec<CudaSlice<f32>>,
        launches: Vec<Launch>,
    }

    fn arch() -> String {
        std::env::var("RIBT_CUDA_ARCH").unwrap_or_else(|_| "sm_86".into())
    }

    impl Exe {
        pub fn new(dev: &Device, graph: &TensorGraph, outs: &[TensorId], inputs: &[&[f32]]) -> Self {
            // default: transpile every kernel. RIBT_CUDA_MMA=1 uses the wmma
            // TF32 tiles: ~10% faster but gradients ~10% off (docs §10, ir_gaps)
            let mma = std::env::var("RIBT_CUDA_MMA").is_ok_and(|v| v == "1");
            let mut program = if mma {
                graph.to_cuda(outs)
            } else {
                let mut p = graph.to_program(outs, tang_compile::tensor::wgsl::Target::Wgsl);
                for k in &mut p.kernels {
                    k.source = tang_compile::tensor::cuda::wgsl_to_cuda(&k.source, k.workgroup_size);
                }
                p
            };
            for k in &mut program.kernels {
                k.source = fix_stage_guards(&fix_array_length(&fix_typed_auto(&k.source), graph, &k.reads));
            }
            let buffers = program
                .buffers
                .iter()
                .map(|b| {
                    assert_eq!(b.format, Format::F32, "cuda runtime is f32-only");
                    match b.input {
                        Some(k) => dev.stream.memcpy_stod(inputs[k]).expect("upload"),
                        None => dev.stream.alloc_zeros::<f32>(b.len.max(1)).expect("alloc"),
                    }
                })
                .collect();
            let include = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
            let arch: &'static str = Box::leak(arch().into_boxed_str());
            // nvrtc is single-threaded per call; compile kernels in parallel
            let srcs: Vec<&str> = program.kernels.iter().map(|k| k.source.as_str()).collect();
            let ptxs: Vec<_> = std::thread::scope(|s| {
                let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
                let chunk = srcs.len().div_ceil(n).max(1);
                let hs: Vec<_> = srcs
                    .chunks(chunk)
                    .map(|c| {
                        let include = include.clone();
                        s.spawn(move || {
                            c.iter()
                                .map(|src| {
                                    let opts = CompileOptions { arch: Some(arch), include_paths: vec![format!("{include}/include")], ..Default::default() };
                                    compile_ptx_with_opts(*src, opts).unwrap_or_else(|e| panic!("nvrtc: {e:?}\n{src}"))
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
            });
            let slot = |t: TensorId| {
                let t = program.storage(t);
                program.buffers.iter().position(|b| b.node == t).expect("buffer")
            };
            let launches = program
                .kernels
                .iter()
                .zip(ptxs)
                .map(|(k, ptx)| {
                    let module = dev.ctx.load_module(ptx).expect("load module");
                    let f = module.load_function(tang_compile::tensor::cuda::CUDA_ENTRY).expect("entry");
                    let smem = dynamic_smem(&k.source);
                    if smem > 48 * 1024 {
                        f.set_attribute(
                            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            smem as i32,
                        )
                        .expect("smem attribute");
                    }
                    let cfg = LaunchConfig { grid_dim: (k.count.div_ceil(k.workgroup_size), 1, 1), block_dim: (k.workgroup_size, 1, 1), shared_mem_bytes: smem };
                    Launch { f, cfg, reads: k.reads.iter().map(|t| slot(*t)).collect(), out: slot(k.node), points: k.points }
                })
                .collect();
            Exe { program, buffers, launches }
        }
        pub fn kernels(&self) -> usize {
            self.program.kernels.len()
        }
        pub fn input_slot(&self, k: usize) -> Option<usize> {
            self.program.buffers.iter().position(|b| b.input == Some(k))
        }
        pub fn slot_of(&self, t: TensorId) -> usize {
            let t = self.program.storage(t);
            self.program.buffers.iter().position(|b| b.node == t).unwrap()
        }
        pub fn spec(&self, i: usize) -> (usize, Format) {
            let b = &self.program.buffers[i];
            (b.len, b.format)
        }
        pub fn write_slot(&mut self, dev: &Device, i: usize, data: &[f32]) {
            dev.stream.memcpy_htod(data, &mut self.buffers[i]).expect("write");
        }
        pub fn read_slot(&self, dev: &Device, i: usize) -> Vec<f32> {
            dev.stream.synchronize().expect("sync");
            let mut v = dev.stream.memcpy_dtov(&self.buffers[i]).expect("download");
            v.truncate(self.program.buffers[i].len);
            v
        }
        pub fn read(&self, dev: &Device, t: TensorId) -> Vec<f32> {
            self.read_slot(dev, self.slot_of(t))
        }
        pub fn swap(&mut self, a: usize, b: usize) {
            self.buffers.swap(a, b);
        }
        pub fn run(&mut self, dev: &Device) {
            for l in &self.launches {
                let mut launch = dev.stream.launch_builder(&l.f);
                for &r in &l.reads {
                    launch.arg(&self.buffers[r]);
                }
                launch.arg(&self.buffers[l.out]);
                launch.arg(&l.points);
                unsafe { launch.launch(l.cfg) }.expect("launch");
            }
        }
        pub fn time(&mut self, dev: &Device, iters: usize) -> f64 {
            for _ in 0..3 {
                self.run(dev);
            }
            dev.stream.synchronize().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..iters {
                self.run(dev);
            }
            dev.stream.synchronize().unwrap();
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        }
    }
}

// ---------------------------------------------------------------------------
// wgpu

#[cfg(feature = "wgpu")]
mod backend {
    use std::cell::RefCell;

    use tang_bench::ir_gpu::Executable;
    use tang_compile::tensor::{Format, TensorGraph, TensorId};
    use tang_gpu::{GpuDevice, KernelCache};

    pub struct Device {
        pub dev: GpuDevice,
        pub cache: RefCell<KernelCache>,
    }

    impl Device {
        pub fn new() -> Self {
            Device { dev: GpuDevice::new_sync().expect("no wgpu adapter"), cache: RefCell::new(KernelCache::new()) }
        }
        pub fn name(&self) -> String {
            format!("wgpu:{}", self.dev.name)
        }
    }

    pub struct Exe(pub Executable);

    impl Exe {
        pub fn new(dev: &Device, graph: &TensorGraph, outs: &[TensorId], inputs: &[&[f32]]) -> Self {
            Exe(Executable::new(&dev.dev, graph.to_wgsl(outs), inputs))
        }
        pub fn kernels(&self) -> usize {
            self.0.program.kernels.len()
        }
        pub fn input_slot(&self, k: usize) -> Option<usize> {
            self.0.program.buffers.iter().position(|b| b.input == Some(k))
        }
        pub fn slot_of(&self, t: TensorId) -> usize {
            let t = self.0.program.storage(t);
            self.0.program.buffers.iter().position(|b| b.node == t).unwrap()
        }
        pub fn spec(&self, i: usize) -> (usize, Format) {
            let b = &self.0.program.buffers[i];
            (b.len, b.format)
        }
        pub fn write_slot(&mut self, dev: &Device, i: usize, data: &[f32]) {
            assert_eq!(self.spec(i).1, Format::F32);
            self.0.buffers[i].write(&dev.dev, data);
        }
        pub fn read_slot(&self, dev: &Device, i: usize) -> Vec<f32> {
            let mut v = self.0.buffers[i].to_vec_sync(&dev.dev);
            v.truncate(self.spec(i).0);
            v
        }
        pub fn read(&self, dev: &Device, t: TensorId) -> Vec<f32> {
            self.0.read(&dev.dev, t)
        }
        pub fn swap(&mut self, a: usize, b: usize) {
            self.0.buffers.swap(a, b);
        }
        pub fn run(&mut self, dev: &Device) {
            self.0.run(&dev.dev, &mut dev.cache.borrow_mut());
        }
        pub fn time(&mut self, dev: &Device, iters: usize) -> f64 {
            for _ in 0..3 {
                self.run(dev);
            }
            let _ = self.read_slot(dev, 0);
            let t = std::time::Instant::now();
            for _ in 0..iters {
                self.run(dev);
            }
            let _ = self.read_slot(dev, 0);
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        }
    }
}


// ---------------------------------------------------------------------------
// cpu: tang-gpu's C lowering, compiled by the system `cc` at load time and
// dlopen'ed, cached on disk, optionally OpenMP-threaded (src/tang_cpu.rs). For a robot with gcc and no GPU path (the K1).
// Only graph inputs and outputs are addressable (no ping-pong `swap`): enough
// for inference, not for the trainers.

#[cfg(feature = "cpu")]
mod backend {
    use tang_compile::tensor::{Format, TensorGraph, TensorId};
    use crate::tang_cpu::CpuProgram;

    pub struct Device;

    impl Device {
        pub fn new() -> Self {
            Device
        }
        pub fn name(&self) -> String {
            "cpu(C)".into()
        }
    }

    pub struct Exe(pub CpuProgram);

    impl Exe {
        pub fn new(_dev: &Device, graph: &TensorGraph, outs: &[TensorId], inputs: &[&[f32]]) -> Self {
            Exe(CpuProgram::compile(graph, outs, inputs).expect("cpu compile"))
        }
        pub fn kernels(&self) -> usize {
            self.0.program.program.kernels.len()
        }
        pub fn input_slot(&self, k: usize) -> Option<usize> {
            self.0.program.program.buffers.iter().position(|b| b.input == Some(k))
        }
        pub fn slot_of(&self, t: TensorId) -> usize {
            let t = self.0.program.program.storage(t);
            self.0.program.program.buffers.iter().position(|b| b.node == t).unwrap()
        }
        pub fn spec(&self, i: usize) -> (usize, Format) {
            let b = &self.0.program.program.buffers[i];
            (b.len, b.format)
        }
        pub fn write_slot(&self, _dev: &Device, i: usize, data: &[f32]) {
            let k = self.0.program.program.buffers[i].input.expect("cpu: only graph inputs are writable");
            self.0.write_input(k, data);
        }
        pub fn read_slot(&self, _dev: &Device, i: usize) -> Vec<f32> {
            if let Some(o) = self.0.program.outputs.iter().position(|&b| b == i) {
                return self.0.read_output(o);
            }
            panic!("cpu: buffer {i} is not an output (inputs are not readable back)")
        }
        pub fn read(&self, dev: &Device, t: TensorId) -> Vec<f32> {
            self.read_slot(dev, self.slot_of(t))
        }
        pub fn swap(&mut self, _a: usize, _b: usize) {
            panic!("cpu backend: buffer swap (training) is unsupported")
        }
        pub fn run(&mut self, _dev: &Device) {
            self.0.run();
        }
        pub fn time(&mut self, _dev: &Device, iters: usize) -> f64 {
            self.0.time(iters) * 1e3
        }
    }
}

// ---------------------------------------------------------------------------
// opencl: `to_opencl` on the dlopen'ed OpenCL runtime with a binary cache
// (src/tang_cl.rs, shared with ribt), for the K1's Adreno 740.

#[cfg(feature = "opencl")]
mod backend {
    use tang_compile::tensor::{Format, TensorGraph, TensorId};
    use crate::tang_cl::{Cl, ClProgram};

    pub struct Device(pub &'static Cl);

    impl Device {
        pub fn new() -> Self {
            Device(Cl::get().unwrap_or_else(|e| panic!("{e}")))
        }
        pub fn name(&self) -> String {
            format!("opencl:{}", self.0.name)
        }
    }

    pub struct Exe(pub ClProgram);

    impl Exe {
        pub fn new(_dev: &Device, graph: &TensorGraph, outs: &[TensorId], inputs: &[&[f32]]) -> Self {
            let patch = |src: &str, reads: &[TensorId]| super::fix::fix_stage_guards(&super::fix::fix_array_length(src, graph, reads));
            Exe(ClProgram::compile(graph, outs, inputs, &patch).unwrap_or_else(|e| panic!("{e}")))
        }
        pub fn kernels(&self) -> usize {
            self.0.kernels()
        }
        pub fn input_slot(&self, k: usize) -> Option<usize> {
            self.0.program.buffers.iter().position(|b| b.input == Some(k))
        }
        pub fn slot_of(&self, t: TensorId) -> usize {
            let t = self.0.program.storage(t);
            self.0.program.buffers.iter().position(|b| b.node == t).unwrap()
        }
        pub fn spec(&self, i: usize) -> (usize, Format) {
            let b = &self.0.program.buffers[i];
            (b.len, b.format)
        }
        pub fn write_slot(&mut self, _dev: &Device, i: usize, data: &[f32]) {
            self.0.write_slot(i, data);
        }
        pub fn read_slot(&self, _dev: &Device, i: usize) -> Vec<f32> {
            self.0.read_slot(i)
        }
        pub fn read(&self, dev: &Device, t: TensorId) -> Vec<f32> {
            self.read_slot(dev, self.slot_of(t))
        }
        pub fn swap(&mut self, a: usize, b: usize) {
            self.0.swap(a, b);
        }
        pub fn run(&mut self, _dev: &Device) {
            self.0.run();
        }
        pub fn time(&mut self, _dev: &Device, iters: usize) -> f64 {
            self.0.time(iters)
        }
    }
}

pub use backend::Device;
#[cfg(feature = "cuda")]
pub use fix::{fix_array_length, fix_stage_guards, fix_typed_auto};

pub struct Program {
    pub graph: TensorGraph,
    pub exe: backend::Exe,
    pub outs: Vec<TensorId>,
    pub compile_ms: f64,
}

impl Program {
    /// Optimize `g` for `outs` and upload `inputs` (one slice per graph input).
    pub fn new(dev: &Device, g: &TensorGraph, outs: &[TensorId], inputs: &[&[f32]]) -> Self {
        let t = Instant::now();
        // cpu: tang's C lowering wants every graph input live, so dead ones
        // (unused weights) are kept by listing each input node as an extra
        // output after the real ones (output indices stay as given).
        #[cfg(feature = "cpu")]
        let outs = &{
            use tang_compile::tensor::Op;
            let mut all = outs.to_vec();
            let mut seen = std::collections::HashSet::new();
            for (i, n) in g.nodes().iter().enumerate() {
                if let Op::Input(k) = n.op {
                    if seen.insert(k) {
                        all.push(TensorId::from_index(i));
                    }
                }
            }
            all
        };
        let (graph, outs) = optimize(g, outs);
        Self::from_optimized(dev, graph, outs, inputs, t)
    }

    pub fn from_optimized(dev: &Device, graph: TensorGraph, outs: Vec<TensorId>, inputs: &[&[f32]], t: Instant) -> Self {
        let exe = backend::Exe::new(dev, &graph, &outs, inputs);
        Self { graph, exe, outs, compile_ms: t.elapsed().as_secs_f64() * 1e3 }
    }

    pub fn kernels(&self) -> usize {
        self.exe.kernels()
    }

    /// Buffers each kernel binds (its reads + its output). Metal allows 31.
    #[cfg(feature = "metal")]
    pub fn kernel_buffers(&self) -> Vec<usize> {
        self.exe.kernel_buffers()
    }

    /// Overwrite graph input `k` (a dead input has no buffer: ignored).
    pub fn write_input(&mut self, dev: &Device, k: usize, data: &[f32]) {
        let Some(i) = self.exe.input_slot(k) else { return };
        assert_eq!(self.exe.spec(i).0, data.len(), "input {k} length");
        self.exe.write_slot(dev, i, data);
    }

    /// Read graph input `k` back (e.g. parameters after training steps).
    pub fn read_input(&self, dev: &Device, k: usize) -> Option<Vec<f32>> {
        let i = self.exe.input_slot(k)?;
        Some(self.exe.read_slot(dev, i))
    }

    /// Output `i` (an index into the `outs` given at construction).
    pub fn read(&self, dev: &Device, i: usize) -> Vec<f32> {
        self.exe.read(dev, self.outs[i])
    }

    /// Exchange the buffers behind graph input `k` and output `i`: after a
    /// step, `p'` becomes the next step's `p` with no copy, and the old
    /// `p` buffer is overwritten by the next run.
    pub fn swap_input_output(&mut self, k: usize, i: usize) {
        let a = self.exe.input_slot(k).expect("input has a buffer");
        let b = self.exe.slot_of(self.outs[i]);
        assert_ne!(a, b);
        let (sa, sb) = (self.exe.spec(a), self.exe.spec(b));
        assert_eq!(sa, sb, "ping-pong buffers must share length and format");
        self.exe.swap(a, b);
    }

    pub fn run(&mut self, dev: &Device) {
        self.exe.run(dev);
    }

    /// Debug (CUDA): after `run`, compare every kernel's output buffer with
    /// the f64 interpreter on the optimized graph. A kernel is a *source*
    /// when its output error exceeds `tol` while every buffer it reads is
    /// within `tol / 10` (graph inputs are exact): those are printed, and
    /// the first one's source goes to `dump`. `inputs` = f64 graph inputs.
    #[cfg(feature = "cuda")]
    pub fn bisect(&self, dev: &Device, inputs: &[Vec<f64>], tol: f64, dump: &std::path::Path) {
        let ks = &self.exe.program.kernels;
        let nodes: Vec<TensorId> = ks.iter().map(|k| k.node).collect();
        let want = self.graph.eval(inputs, &nodes);
        let mut err: std::collections::HashMap<TensorId, f64> = std::collections::HashMap::new();
        let (mut off, mut sources) = (0, 0);
        for (i, k) in ks.iter().enumerate() {
            let got = self.exe.read_slot(dev, self.exe.slot_of(k.node));
            let w = &want[i];
            let sc = w.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
            let e = got.iter().zip(w).map(|(g, w)| if g.is_finite() { (*g as f64 - w).abs() / sc } else { f64::INFINITY }).fold(0.0f64, f64::max);
            err.insert(k.node, e);
            let ein = k.reads.iter().map(|r| *err.get(&self.exe.program.storage(*r)).unwrap_or(&0.0)).fold(0.0f64, f64::max);
            if e > tol {
                off += 1;
                if ein < tol / 10.0 {
                    sources += 1;
                    let shape: Vec<usize> = self.graph.nodes()[k.node.index()].shape.iter().map(|d| d.n()).collect();
                    eprintln!("  SOURCE kernel {i} node {:?} shape {shape:?}: rel {e:.2e} (reads max {ein:.1e}), wg {}, count {}", k.node, k.workgroup_size, k.count);
                    if sources == 1 {
                        let reads: Vec<String> = k.reads.iter().map(|r| format!("{r:?}")).collect();
                        std::fs::write(dump, format!("// kernel {i}, node {:?}, reads {}\n{}", k.node, reads.join(","), k.source)).ok();
                    }
                }
            }
        }
        eprintln!("bisect: {off}/{} kernels off (tol {tol:.0e}), {sources} sources", ks.len());
    }

    /// Kernel-only milliseconds per run.
    pub fn time(&mut self, dev: &Device, iters: usize) -> f64 {
        self.exe.time(dev, iters)
    }
}
