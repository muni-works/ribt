//! OpenCL runtime for tensor-IR programs, for GPUs that have OpenCL and no
//! usable Vulkan/Metal/CUDA (the K1's Adreno 740). tang lowers the program
//! (`TensorGraph::to_opencl`: the WGSL kernels transpiled to OpenCL C, with
//! the CUDA calling convention `main0(reads…, out, uint points)`, launched as
//! `count` items in groups of `workgroup_size`); this file runs it.
//!
//! - **libOpenCL is dlopen'ed** at first use, so nothing links against it and
//!   a build without the library still works until a program is made. Tried
//!   in order: `$TANG_OPENCL_LIB`, `libOpenCL.so`, the Qualcomm path
//!   `/opt/qti/usr/lib/libOpenCL.so.adreno`, `libOpenCL.so.1`. On the K1,
//!   `libOpenCL.so` is Qualcomm's library itself, and `libOpenCL.so.1` is the
//!   generic ocl-icd loader with no vendor file (it finds no platform).
//! - **a persistent binary cache**: each distinct kernel source is built once
//!   (`clBuildProgram`, ~0.1–0.4 s on the Adreno) and its device binary
//!   (`CL_PROGRAM_BINARIES`) is kept under `$TANG_CL_CACHE` (default
//!   `~/.cache/tang-cl`), keyed by a hash of the device, driver, build options
//!   and source. Cold builds run on a few threads.
//! - **one context, one in-order queue** per process; `run` only enqueues
//!   (no wait between kernels), a read blocks.
//!
//! f32 buffers only, like the CUDA runtime in hear. Shared by `ribt` and
//! `hear` (hear includes this file by `#[path]`). Upstream target: tang-gpu.

use std::ffi::{c_char, c_void, CString};
use std::path::PathBuf;
use std::ptr::null_mut;
use std::sync::OnceLock;

use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{Format, TensorGraph, TensorId};

type H = *mut c_void;
type Int = i32;

const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_DRIVER_VERSION: u32 = 0x102D;
const CL_MEM_READ_WRITE: u64 = 1;
const CL_MEM_COPY_HOST_PTR: u64 = 1 << 5;
const CL_PROGRAM_BINARY_SIZES: u32 = 0x1165;
const CL_PROGRAM_BINARIES: u32 = 0x1166;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
// cl_qcom_perf_hint / cl_qcom_priority_hint context properties
const CL_CONTEXT_PERF_HINT_QCOM: isize = 0x40C2;
const CL_PERF_HINT_HIGH_QCOM: isize = 0x40C3;
const CL_CONTEXT_PRIORITY_HINT_QCOM: isize = 0x40C9;
const CL_PRIORITY_HINT_HIGH_QCOM: isize = 0x40CA;

#[allow(non_snake_case)]
struct Api {
    _lib: libloading::Library,
    clGetPlatformIDs: unsafe extern "C" fn(u32, *mut H, *mut u32) -> Int,
    clGetDeviceIDs: unsafe extern "C" fn(H, u64, u32, *mut H, *mut u32) -> Int,
    clGetDeviceInfo: unsafe extern "C" fn(H, u32, usize, *mut c_void, *mut usize) -> Int,
    clCreateContext: unsafe extern "C" fn(*const isize, u32, *const H, *const c_void, *const c_void, *mut Int) -> H,
    clCreateCommandQueue: unsafe extern "C" fn(H, H, u64, *mut Int) -> H,
    clCreateBuffer: unsafe extern "C" fn(H, u64, usize, *mut c_void, *mut Int) -> H,
    clEnqueueWriteBuffer: unsafe extern "C" fn(H, H, u32, usize, usize, *const c_void, u32, *const H, *mut H) -> Int,
    clEnqueueReadBuffer: unsafe extern "C" fn(H, H, u32, usize, usize, *mut c_void, u32, *const H, *mut H) -> Int,
    clCreateProgramWithSource: unsafe extern "C" fn(H, u32, *const *const c_char, *const usize, *mut Int) -> H,
    clCreateProgramWithBinary: unsafe extern "C" fn(H, u32, *const H, *const usize, *const *const u8, *mut Int, *mut Int) -> H,
    clBuildProgram: unsafe extern "C" fn(H, u32, *const H, *const c_char, *const c_void, *const c_void) -> Int,
    clGetProgramBuildInfo: unsafe extern "C" fn(H, H, u32, usize, *mut c_void, *mut usize) -> Int,
    clGetProgramInfo: unsafe extern "C" fn(H, u32, usize, *mut c_void, *mut usize) -> Int,
    clCreateKernel: unsafe extern "C" fn(H, *const c_char, *mut Int) -> H,
    clSetKernelArg: unsafe extern "C" fn(H, u32, usize, *const c_void) -> Int,
    clEnqueueNDRangeKernel: unsafe extern "C" fn(H, H, u32, *const usize, *const usize, *const usize, u32, *const H, *mut H) -> Int,
    clFlush: unsafe extern "C" fn(H) -> Int,
    clGetEventProfilingInfo: unsafe extern "C" fn(H, u32, usize, *mut c_void, *mut usize) -> Int,
    clReleaseEvent: unsafe extern "C" fn(H) -> Int,
    clFinish: unsafe extern "C" fn(H) -> Int,
    clReleaseMemObject: unsafe extern "C" fn(H) -> Int,
    clReleaseKernel: unsafe extern "C" fn(H) -> Int,
    clReleaseProgram: unsafe extern "C" fn(H) -> Int,
}

/// The process's OpenCL device: one context and one in-order queue.
pub struct Cl {
    api: Api,
    device: H,
    context: H,
    queue: H,
    pub name: String,
    driver: String,
    /// `TANG_CL_PROFILE=1`: the queue records event times (`ClProgram::profile`)
    pub profiling: bool,
}
// The handles are thread-safe OpenCL objects (the spec makes every call but
// clSetKernelArg on a shared kernel thread-safe; kernels are per program).
unsafe impl Send for Cl {}
unsafe impl Sync for Cl {}

fn check(e: Int, what: &str) -> Result<(), String> {
    if e == 0 {
        Ok(())
    } else {
        Err(format!("OpenCL {what}: error {e}"))
    }
}

fn load_api() -> Result<Api, String> {
    let mut names: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("TANG_OPENCL_LIB") {
        names.push(p);
    }
    names.extend(["libOpenCL.so", "/opt/qti/usr/lib/libOpenCL.so.adreno", "libOpenCL.so.1", "/System/Library/Frameworks/OpenCL.framework/OpenCL"].map(String::from));
    let mut errs = Vec::new();
    for n in &names {
        let lib = match unsafe { libloading::Library::new(n) } {
            Ok(l) => l,
            Err(e) => {
                errs.push(format!("{n}: {e}"));
                continue;
            }
        };
        macro_rules! sym {
            ($name:ident) => {
                *unsafe { lib.get(concat!(stringify!($name), "\0").as_bytes()) }.map_err(|e| format!("{n}: {}: {e}", stringify!($name)))?
            };
        }
        return Ok(Api {
            clGetPlatformIDs: sym!(clGetPlatformIDs),
            clGetDeviceIDs: sym!(clGetDeviceIDs),
            clGetDeviceInfo: sym!(clGetDeviceInfo),
            clCreateContext: sym!(clCreateContext),
            clCreateCommandQueue: sym!(clCreateCommandQueue),
            clCreateBuffer: sym!(clCreateBuffer),
            clEnqueueWriteBuffer: sym!(clEnqueueWriteBuffer),
            clEnqueueReadBuffer: sym!(clEnqueueReadBuffer),
            clCreateProgramWithSource: sym!(clCreateProgramWithSource),
            clCreateProgramWithBinary: sym!(clCreateProgramWithBinary),
            clBuildProgram: sym!(clBuildProgram),
            clGetProgramBuildInfo: sym!(clGetProgramBuildInfo),
            clGetProgramInfo: sym!(clGetProgramInfo),
            clCreateKernel: sym!(clCreateKernel),
            clSetKernelArg: sym!(clSetKernelArg),
            clEnqueueNDRangeKernel: sym!(clEnqueueNDRangeKernel),
            clFlush: sym!(clFlush),
            clGetEventProfilingInfo: sym!(clGetEventProfilingInfo),
            clReleaseEvent: sym!(clReleaseEvent),
            clFinish: sym!(clFinish),
            clReleaseMemObject: sym!(clReleaseMemObject),
            clReleaseKernel: sym!(clReleaseKernel),
            clReleaseProgram: sym!(clReleaseProgram),
            _lib: lib,
        });
    }
    Err(format!("no OpenCL library ({})", errs.join("; ")))
}

impl Cl {
    fn new() -> Result<Cl, String> {
        let api = load_api()?;
        unsafe {
            let mut plats = [null_mut(); 8];
            let mut np = 0u32;
            check((api.clGetPlatformIDs)(8, plats.as_mut_ptr(), &mut np), "clGetPlatformIDs")?;
            let mut device = null_mut();
            for &p in &plats[..np as usize] {
                if (api.clGetDeviceIDs)(p, CL_DEVICE_TYPE_GPU, 1, &mut device, null_mut()) == 0 {
                    break;
                }
            }
            if device.is_null() {
                return Err(format!("OpenCL: no GPU device on {np} platform(s)"));
            }
            let info = |param: u32| {
                let mut b = vec![0u8; 1024];
                let mut n = 0usize;
                (api.clGetDeviceInfo)(device, param, b.len(), b.as_mut_ptr() as *mut c_void, &mut n);
                b.truncate(n.saturating_sub(1));
                String::from_utf8_lossy(&b).into_owned()
            };
            let (name, driver) = (info(CL_DEVICE_NAME), info(CL_DRIVER_VERSION));
            let mut err = 0;
            // Qualcomm hints (high clocks, high priority) when the driver takes them; plain context otherwise
            let hints = std::env::var("TANG_CL_HINTS").map(|v| v != "0").unwrap_or(true) && name.contains("Adreno");
            let props = [CL_CONTEXT_PERF_HINT_QCOM, CL_PERF_HINT_HIGH_QCOM, CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM, 0];
            let mut context = if hints { (api.clCreateContext)(props.as_ptr(), 1, &device, std::ptr::null(), std::ptr::null(), &mut err) } else { null_mut() };
            if context.is_null() {
                context = (api.clCreateContext)(std::ptr::null(), 1, &device, std::ptr::null(), std::ptr::null(), &mut err);
            }
            check(err, "clCreateContext")?;
            let profiling = std::env::var("TANG_CL_PROFILE").is_ok_and(|v| v == "1");
            let queue = (api.clCreateCommandQueue)(context, device, if profiling { 2 } else { 0 }, &mut err);
            check(err, "clCreateCommandQueue")?;
            Ok(Cl { api, device, context, queue, name, driver, profiling })
        }
    }

    /// The process-wide device (created on first use).
    pub fn get() -> Result<&'static Cl, String> {
        static CL: OnceLock<Result<Cl, String>> = OnceLock::new();
        CL.get_or_init(Cl::new).as_ref().map_err(|e| e.clone())
    }

    pub fn finish(&self) {
        unsafe { (self.api.clFinish)(self.queue) };
    }
    pub fn flush(&self) {
        unsafe { (self.api.clFlush)(self.queue) };
    }

    fn build_log(&self, p: H) -> String {
        let mut b = vec![0u8; 1 << 16];
        let mut n = 0usize;
        unsafe { (self.api.clGetProgramBuildInfo)(p, self.device, CL_PROGRAM_BUILD_LOG, b.len(), b.as_mut_ptr() as *mut c_void, &mut n) };
        b.truncate(n.saturating_sub(1));
        String::from_utf8_lossy(&b).into_owned()
    }

    /// A built program for `source`: from the binary cache, else compiled
    /// (and its binary written to the cache).
    fn program(&self, source: &str) -> Result<H, String> {
        use std::hash::{Hash, Hasher};
        let opts = build_options();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (&self.name, &self.driver, &opts, source).hash(&mut h);
        let dir = cache_dir();
        let file = dir.join(format!("{:016x}.bin", h.finish()));
        let copts = CString::new(opts.clone()).unwrap();
        let mut err = 0;
        if let Ok(bin) = std::fs::read(&file) {
            let (len, ptr) = (bin.len(), bin.as_ptr());
            let mut status = 0;
            let p = unsafe { (self.api.clCreateProgramWithBinary)(self.context, 1, &self.device, &len, &ptr, &mut status, &mut err) };
            if err == 0 && status == 0 && unsafe { (self.api.clBuildProgram)(p, 1, &self.device, copts.as_ptr(), std::ptr::null(), std::ptr::null()) } == 0 {
                return Ok(p);
            }
            if !p.is_null() {
                unsafe { (self.api.clReleaseProgram)(p) };
            }
            let _ = std::fs::remove_file(&file); // stale (driver update): rebuild below
        }
        let src = CString::new(source).map_err(|e| e.to_string())?;
        let sp = src.as_ptr();
        let p = unsafe { (self.api.clCreateProgramWithSource)(self.context, 1, &sp, std::ptr::null(), &mut err) };
        check(err, "clCreateProgramWithSource")?;
        if unsafe { (self.api.clBuildProgram)(p, 1, &self.device, copts.as_ptr(), std::ptr::null(), std::ptr::null()) } != 0 {
            let log = self.build_log(p);
            unsafe { (self.api.clReleaseProgram)(p) };
            return Err(format!("clBuildProgram failed:\n{log}\n--- source ---\n{source}"));
        }
        let mut len = 0usize;
        unsafe { (self.api.clGetProgramInfo)(p, CL_PROGRAM_BINARY_SIZES, std::mem::size_of::<usize>(), &mut len as *mut usize as *mut c_void, null_mut()) };
        if len > 0 && std::fs::create_dir_all(&dir).is_ok() {
            let mut bin = vec![0u8; len];
            let mut bp = bin.as_mut_ptr();
            if unsafe { (self.api.clGetProgramInfo)(p, CL_PROGRAM_BINARIES, std::mem::size_of::<*mut u8>(), &mut bp as *mut *mut u8 as *mut c_void, null_mut()) } == 0 {
                let tmp = file.with_extension(format!("{}", std::process::id()));
                if std::fs::write(&tmp, &bin).is_ok() {
                    let _ = std::fs::rename(&tmp, &file);
                }
            }
        }
        Ok(p)
    }

    fn buffer(&self, len: usize, init: Option<&[f32]>) -> Result<H, String> {
        let bytes = len.max(1) * 4;
        let mut err = 0;
        let m = match init {
            Some(d) if !d.is_empty() => {
                assert_eq!(d.len(), len);
                unsafe { (self.api.clCreateBuffer)(self.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, d.as_ptr() as *mut c_void, &mut err) }
            }
            _ => {
                let m = unsafe { (self.api.clCreateBuffer)(self.context, CL_MEM_READ_WRITE, bytes, null_mut(), &mut err) };
                if err == 0 {
                    // zero it, as the CUDA/wgpu runtimes do
                    let z = vec![0f32; len.max(1)];
                    check(unsafe { (self.api.clEnqueueWriteBuffer)(self.queue, m, 1, 0, bytes, z.as_ptr() as *const c_void, 0, std::ptr::null(), null_mut()) }, "zero")?;
                }
                m
            }
        };
        check(err, &format!("clCreateBuffer {bytes} B"))?;
        Ok(m)
    }
}

fn cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("TANG_CL_CACHE") {
        return PathBuf::from(d);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    home.join(".cache/tang-cl")
}

fn build_options() -> String {
    std::env::var("TANG_CL_OPTS").unwrap_or_default()
}

struct Launch {
    kernel: H,
    reads: Vec<usize>,
    out: usize,
    points: u32,
    global: usize,
    local: usize,
}

/// A lowered program on the device: its buffers and one kernel per launch.
pub struct ClProgram {
    pub program: WgslProgram,
    cl: &'static Cl,
    buffers: Vec<H>,
    launches: Vec<Launch>,
    programs: Vec<H>,
    args_dirty: bool,
    /// distinct kernel sources, and how many were built (not cached)
    pub distinct: usize,
}

struct Send2<T>(T);
unsafe impl<T> Send for Send2<T> {}

impl ClProgram {
    /// Lower `g` with `to_opencl`, apply `patch(source, reads)` to every
    /// kernel (text fixes for transpiler gaps), build and upload `inputs`.
    pub fn compile(g: &TensorGraph, outputs: &[TensorId], inputs: &[&[f32]], patch: &dyn Fn(&str, &[TensorId]) -> String) -> Result<Self, String> {
        let cl = Cl::get()?;
        let mut program = g.to_opencl(outputs);
        for k in &mut program.kernels {
            k.source = patch(&k.source, &k.reads);
        }
        if let Some(dir) = std::env::var_os("TANG_CL_DUMP") {
            // every kernel's source, as p{program}_k{index}_wg{wg}_n{count}.cl
            static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let pn = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&dir);
            for (i, k) in program.kernels.iter().enumerate() {
                let n = &g.nodes()[k.node.index()];
                let shape: Vec<usize> = n.shape.iter().map(|d| d.n()).collect();
                let head = format!("// program {pn} kernel {i}: shape {shape:?} split {} tile {:?} reg {:?} inline {}\n", n.split, n.tile, n.reg, n.inline);
                let _ = std::fs::write(dir.join(format!("p{pn}_k{i}_wg{}_n{}.cl", k.workgroup_size, k.count)), head + &k.source);
            }
        }
        let mut buffers = Vec::with_capacity(program.buffers.len());
        for b in &program.buffers {
            if b.format != Format::F32 {
                return Err(format!("tang_cl: f32 buffers only (got {:?})", b.format));
            }
            buffers.push(cl.buffer(b.len, b.input.map(|k| inputs[k]))?);
        }
        // one program per distinct source, built on a few threads
        let mut uniq: Vec<&str> = program.kernels.iter().map(|k| k.source.as_str()).collect();
        uniq.sort_unstable();
        uniq.dedup();
        let n = std::env::var("TANG_CL_BUILD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(3usize).max(1);
        let chunk = uniq.len().div_ceil(n).max(1);
        let built: Vec<Result<Send2<H>, String>> = std::thread::scope(|s| {
            let hs: Vec<_> = uniq.chunks(chunk).map(|c| s.spawn(move || c.iter().map(|src| cl.program(src).map(Send2)).collect::<Vec<_>>())).collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let mut programs = Vec::with_capacity(built.len());
        for b in built {
            programs.push(b?.0);
        }
        let slot = |t: TensorId| {
            let t = program.storage(t);
            program.buffers.iter().position(|b| b.node == t).expect("buffer")
        };
        let entry = CString::new(tang_compile::tensor::opencl::OPENCL_ENTRY).unwrap();
        let mut launches = Vec::with_capacity(program.kernels.len());
        for k in &program.kernels {
            let pi = uniq.binary_search(&k.source.as_str()).unwrap();
            let mut err = 0;
            // a kernel object per launch: its arguments are set once
            let kernel = unsafe { (cl.api.clCreateKernel)(programs[pi], entry.as_ptr(), &mut err) };
            check(err, "clCreateKernel")?;
            let local = k.workgroup_size as usize;
            launches.push(Launch { kernel, reads: k.reads.iter().map(|t| slot(*t)).collect(), out: slot(k.node), points: k.points, global: (k.count as usize).div_ceil(local) * local, local });
        }
        let distinct = uniq.len();
        let mut p = ClProgram { program, cl, buffers, launches, programs, args_dirty: true, distinct };
        p.set_args()?;
        Ok(p)
    }

    fn set_args(&mut self) -> Result<(), String> {
        let api = &self.cl.api;
        for l in &self.launches {
            let mut a = 0u32;
            for &r in l.reads.iter().chain(std::iter::once(&l.out)) {
                check(unsafe { (api.clSetKernelArg)(l.kernel, a, std::mem::size_of::<H>(), &self.buffers[r] as *const H as *const c_void) }, "clSetKernelArg")?;
                a += 1;
            }
            check(unsafe { (api.clSetKernelArg)(l.kernel, a, 4, &l.points as *const u32 as *const c_void) }, "clSetKernelArg points")?;
        }
        self.args_dirty = false;
        Ok(())
    }

    pub fn kernels(&self) -> usize {
        self.launches.len()
    }

    pub fn write_slot(&self, i: usize, data: &[f32]) {
        assert_eq!(self.program.buffers[i].len, data.len());
        let e = unsafe { (self.cl.api.clEnqueueWriteBuffer)(self.cl.queue, self.buffers[i], 1, 0, data.len() * 4, data.as_ptr() as *const c_void, 0, std::ptr::null(), null_mut()) };
        check(e, "clEnqueueWriteBuffer").unwrap();
    }

    pub fn read_slot(&self, i: usize) -> Vec<f32> {
        let mut v = vec![0f32; self.program.buffers[i].len];
        let e = unsafe { (self.cl.api.clEnqueueReadBuffer)(self.cl.queue, self.buffers[i], 1, 0, v.len() * 4, v.as_mut_ptr() as *mut c_void, 0, std::ptr::null(), null_mut()) };
        check(e, "clEnqueueReadBuffer").unwrap();
        v
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.buffers.swap(a, b);
        self.args_dirty = true;
    }

    /// Enqueue every kernel (no wait); a later read or `finish` waits.
    pub fn run(&mut self) {
        // TANG_CL_PROFILE_RUNS=1 (with TANG_CL_PROFILE=1): every run is a profiled run, one line each
        if self.cl.profiling && std::env::var("TANG_CL_PROFILE_RUNS").is_ok_and(|v| v == "1") {
            if let Some((sum, span, host, top)) = self.profile(3) {
                let t: Vec<String> = top.iter().map(|(ms, i)| {
                    let k = &self.program.kernels[*i];
                    format!("k{i} {ms:.2}ms wg{} n{}", k.workgroup_size, k.count)
                }).collect();
                eprintln!("tang_cl run: {} kernels, device {sum:.2} ms, span {span:.2} ms, host {host:.2} ms; top: {}", self.launches.len(), t.join(", "));
            }
            return;
        }
        if self.args_dirty {
            self.set_args().unwrap();
        }
        let api = &self.cl.api;
        for l in &self.launches {
            let e = unsafe { (api.clEnqueueNDRangeKernel)(self.cl.queue, l.kernel, 1, std::ptr::null(), &l.global, &l.local, 0, std::ptr::null(), null_mut()) };
            check(e, "clEnqueueNDRangeKernel").unwrap();
        }
        self.cl.flush();
    }

    pub fn finish(&self) {
        self.cl.finish();
    }

    /// Needs `TANG_CL_PROFILE=1`. One run with an event per kernel: (sum of
    /// kernel device times, first start to last end, host enqueue time), ms,
    /// and the `top` slowest kernels as (ms, kernel index).
    pub fn profile(&mut self, top: usize) -> Option<(f64, f64, f64, Vec<(f64, usize)>)> {
        if !self.cl.profiling {
            return None;
        }
        if self.args_dirty {
            self.set_args().unwrap();
        }
        self.finish();
        let api = &self.cl.api;
        let t = std::time::Instant::now();
        let mut evs = vec![null_mut(); self.launches.len()];
        for (l, e) in self.launches.iter().zip(&mut evs) {
            check(unsafe { (api.clEnqueueNDRangeKernel)(self.cl.queue, l.kernel, 1, std::ptr::null(), &l.global, &l.local, 0, std::ptr::null(), e) }, "enqueue").unwrap();
        }
        let host = t.elapsed().as_secs_f64() * 1e3;
        self.finish();
        let q = |e: H, w: u32| {
            let mut v = 0u64;
            unsafe { (api.clGetEventProfilingInfo)(e, w, 8, &mut v as *mut u64 as *mut c_void, null_mut()) };
            v
        };
        // CL_PROFILING_COMMAND_START / END
        let (mut sum, mut first, mut last) = (0.0, u64::MAX, 0u64);
        let mut per: Vec<(f64, usize)> = Vec::with_capacity(evs.len());
        for (i, &e) in evs.iter().enumerate() {
            let (s, f) = (q(e, 0x1282), q(e, 0x1283));
            let ms = f.saturating_sub(s) as f64 * 1e-6;
            sum += ms;
            first = first.min(s);
            last = last.max(f);
            per.push((ms, i));
            unsafe { (api.clReleaseEvent)(e) };
        }
        if std::env::var("TANG_CL_PROFILE_BY_SOURCE").is_ok_and(|v| v == "1") {
            // one row per distinct source: (total ms, first kernel index); count printed by caller
            let mut by: std::collections::HashMap<&str, (f64, usize, usize)> = std::collections::HashMap::new();
            for &(ms, i) in &per {
                let e = by.entry(self.program.kernels[i].source.as_str()).or_insert((0.0, i, 0));
                e.0 += ms;
                e.2 += 1;
            }
            per = by.into_values().map(|(ms, i, n)| {
                eprintln!("    source of kernel {i}: {n} launches, {ms:.2} ms total");
                (ms, i)
            }).collect();
        }
        per.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        per.truncate(top);
        Some((sum, last.saturating_sub(first) as f64 * 1e-6, host, per))
    }

    /// Milliseconds per run (enqueue all + finish), after 3 warm-up runs.
    pub fn time(&mut self, iters: usize) -> f64 {
        for _ in 0..3 {
            self.run();
        }
        self.finish();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            self.run();
        }
        self.finish();
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    }
}

impl Drop for ClProgram {
    fn drop(&mut self) {
        let api = &self.cl.api;
        self.cl.finish();
        unsafe {
            for l in &self.launches {
                (api.clReleaseKernel)(l.kernel);
            }
            for &p in &self.programs {
                (api.clReleaseProgram)(p);
            }
            for &m in &self.buffers {
                (api.clReleaseMemObject)(m);
            }
        }
    }
}

/// Reschedule every real reduction of an optimized graph for the Adreno 740
/// (6 CUs), after `schedule_auto`. Measured on the K1:
/// - Reductions hear's optimize() materializes were inline when schedule_auto ran, so they have no
///   schedule at all: one thread per point and a serial 512–2048 loop (a [512, 2]
///   projection ran as 4 work-groups, 3.7 ms; 259 ms of device time per 160 ms chunk).
/// - schedule_auto's choices are desktop ones: 256-thread splits (tang's Adreno tuning:
///   64–128 win), and 16×16 shared-memory tiles even when a point axis is 2 wide
///   ([1024, 2]: 64 tiles of which 7/8 of the lanes are dead).
/// Rule: few points (< 16k) or a short point axis (< 16): split the reduction across
/// TANG_CL_SPLIT threads (default 128); otherwise a TANG_CL_TILE thread tile (default 8x8x16: 8×8 threads,
/// k chunk 16) with TANG_CL_REG registers (default 4x8). TANG_CL_SCHED=0 keeps schedule_auto's choices.
pub fn adreno_schedule(g: &mut TensorGraph) {
    if std::env::var("TANG_CL_SCHED").is_ok_and(|v| v == "0") {
        return;
    }
    {
        let env = |k: &str, d: u32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let split = env("TANG_CL_SPLIT", 128);
        let reg: (u32, u32) = std::env::var("TANG_CL_REG").ok().and_then(|v| {
            let (a, b) = v.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        }).unwrap_or((4, 8));
        // matmul-shaped: threads over (last, second-to-last) point axes and the k chunk
        let tile: (u32, u32, u32) = std::env::var("TANG_CL_TILE").ok().and_then(|v| {
            let p: Vec<u32> = v.split('x').filter_map(|x| x.parse().ok()).collect();
            (p.len() == 3).then(|| (p[0], p[1], p[2]))
        }).unwrap_or((8, 8, 16));
        // Materialize every real contraction (>= 128 terms) that schedule_auto inlined into a
        // consumer: inlined, it runs with its consumer's schedule (in RIBT's decoder a [1, 64,
        // 384] output with a 1536-term dot product inline per point ran one thread per point,
        // 40–85 ms per kernel). Its own buffer is small next to the weights it reads.
        for i in 0..g.nodes().len() {
            let big = g.comp_at(i).is_some_and(|c| c.monoid.is_some() && c.red.iter().map(|d| d.n()).product::<usize>() >= 128);
            if big && g.nodes()[i].inline {
                g.set_inline(TensorId::from_index(i), false);
            }
        }
        // an epilogue map tiled/split over a reduction just materialized: untile it (as hear's optimize)
        for j in 0..g.nodes().len() {
            let n = &g.nodes()[j];
            if (n.tile != (0, 0, 0) || n.split != 0) && g.comp_at(j).is_some() && g.tiled_reduction(j).is_none() {
                let reg = n.reg;
                g.set_schedule(TensorId::from_index(j), 0, (0, 0, 0), reg);
            }
        }
        for i in 0..g.nodes().len() {
            let Some(c) = g.comp_at(i) else { continue };
            let red: usize = c.red.iter().map(|d| d.n()).product();
            if c.monoid.is_none() || red < 128 || g.nodes()[i].inline {
                continue;
            }
            let shape: Vec<usize> = g.nodes()[i].shape.iter().map(|d| d.n()).collect();
            let pts: usize = shape.iter().product();
            let short = shape.len() < 2 || shape[shape.len() - 1] < 16 || shape[shape.len() - 2] < 16;
            if pts < (1 << 14) || short {
                g.set_schedule(TensorId::from_index(i), split, (0, 0, 0), (1, 1));
            } else {
                g.set_schedule(TensorId::from_index(i), 0, tile, reg);
            }
        }
    }
}
