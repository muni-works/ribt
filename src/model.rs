//! The V1 vocoder (Vocos-style: frame-rate ConvNeXt backbone + iSTFT head),
//! the log-mel frontend and the training losses, all as one tang tensor
//! graph. Nothing here runs anything: it builds graphs and records which
//! graph input is what.

use tang::Scalar;
use tang_compile::tensor::{Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

use crate::dsp;
use crate::glottis::{self, PConsts};

/// Physics-range inputs of the vocoder head.
pub struct PhysIn {
    pub pc: PConsts,
    /// `[B, P, T]` physical params (data "phys")
    pub th: TensorId,
    /// `[B, 1, T]` F0 in Hz (data "f0hz")
    pub f0hz: TensorId,
    /// the log-mel the output must reproduce exactly
    pub target: TensorId,
}

/// Adds the range inputs for `cfg.range` (none for 0).
pub fn phys_inputs(b: &mut Built, batch: usize, frames: usize, mel: TensorId) -> Option<PhysIn> {
    if b.cfg.range == 0 {
        return None;
    }
    let pc = glottis::pconsts(b);
    let np = glottis::n_params(&b.cfg);
    let th = b.input(&[batch, np, frames], Role::Data("phys"));
    let f0hz = b.input(&[batch, 1, frames], Role::Data("f0hz"));
    Some(PhysIn { pc, th, f0hz, target: mel })
}
use crate::ir::{self, arg, form, lit, pick, view};

#[derive(Clone, Debug)]
pub struct Config {
    pub sr: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub n_mels: usize,
    pub dim: usize,
    pub ff: usize,
    pub layers: usize,
    pub embed_k: usize,
    pub dw_k: usize,
    /// MR-STFT loss resolutions `(n_fft, hop)`; hop must divide n_fft and
    /// the training segment length.
    pub loss_res: Vec<(usize, usize)>,
    pub mel_weight: f64,
    /// V1.1: F0 source conditioning. A frame-rate harmonic excitation
    /// (a comb at f0's harmonics, projected onto the mel bands) plus voicing
    /// and log-f0 are added to the backbone input by zero-initialized convs,
    /// so a V1 checkpoint loads and starts as exactly the V1 function.
    pub f0: bool,
    /// §11.5 range operator: 0 = V1.2 head (exp(a)·e^{iφ}), 1 = exact
    /// projection of |F⁺e|·exp(a) (RNDVoC pseudo-inverse range), 2 = exact
    /// projection of phys_mag(θ)·exp(a) (physics range).
    pub range: u8,
    /// cepstral envelope order (physics range)
    pub n_cep: usize,
    /// §29.5 harmonic source: the lowest `hsrc` STFT bins of a sample-rate
    /// sine excitation at f0's harmonics (`dsp::harmonic_feats`, phase
    /// accumulated on the host), added to the backbone input by a
    /// zero-initialised conv `hsrc.w` (0 = off). Needs `f0`.
    pub hsrc: usize,
    /// hidden-decoder rung 1 (docs/hidden-decoder-plan.md): G reads the
    /// acoustic decoder's last hidden layer (`hid` channels) plus a ones row
    /// and a ln-gain row instead of a log-mel (0 = off). `embed.w` is then
    /// `[d, hid + 2, k]`; `ribt-tts hid-init` sets it to `W_embed·W_melproj`
    /// (ones row: `W_embed·b_mel`, gain row: `W_embed·1`), so step 0 is the
    /// mel path exactly, conv edges included.
    pub hid: usize,
}

impl Config {
    /// V1: 24 kHz, hop 256 (Vocos), 100 mels, d512 × 8 blocks, ff 1536.
    pub fn v1() -> Self {
        Self {
            sr: 24000,
            n_fft: 1024,
            hop: 256,
            n_mels: 100,
            dim: 512,
            ff: 1536,
            layers: 8,
            embed_k: 7,
            dw_k: 7,
            loss_res: vec![(512, 128), (1024, 256), (2048, 512)],
            mel_weight: 1.0,
            f0: false,
            range: 0,
            n_cep: 30,
            hsrc: 0,
            hid: 0,
        }
    }

    /// Small enough for the f64 interpreter (gradient checks).
    pub fn tiny() -> Self {
        Self {
            sr: 8000,
            n_fft: 16,
            hop: 4,
            n_mels: 6,
            dim: 8,
            ff: 12,
            layers: 2,
            embed_k: 3,
            dw_k: 3,
            loss_res: vec![(8, 2), (16, 4)],
            mel_weight: 1.0,
            f0: false,
            range: 0,
            n_cep: 30,
            hsrc: 0,
            hid: 0,
        }
    }

    /// Big enough that the GPU schedules take their tiled paths, small
    /// enough for the f64 interpreter (`gpucheck --cfg check`).
    pub fn check() -> Self {
        Self {
            sr: 16000,
            n_fft: 128,
            hop: 32,
            n_mels: 20,
            dim: 64,
            ff: 192,
            layers: 2,
            embed_k: 7,
            dw_k: 7,
            loss_res: vec![(64, 16), (128, 32), (256, 64)],
            mel_weight: 1.0,
            f0: false,
            range: 0,
            n_cep: 30,
            hsrc: 0,
            hid: 0,
        }
    }

    /// A mid-size config for quick GPU smoke runs.
    pub fn small() -> Self {
        Self {
            dim: 256,
            ff: 768,
            layers: 4,
            ..Self::v1()
        }
    }

    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "v1" => Some(Self::v1()),
            "small" => Some(Self::small()),
            "tiny" => Some(Self::tiny()),
            "check" => Some(Self::check()),
            "v1f0" => Some(Self { f0: true, ..Self::v1() }),
            "tinyf0" => Some(Self { f0: true, ..Self::tiny() }),
            "checkf0" => Some(Self { f0: true, ..Self::check() }),
            // §29.5: + harmonic source on bins < 2 kHz (86 × 23.4 Hz)
            "v1f0h" => Some(Self { f0: true, hsrc: 86, ..Self::v1() }),
            "checkf0h" => Some(Self { f0: true, hsrc: 12, ..Self::check() }),
            // hidden-decoder rung 1: the release decoder's 384-d last layer in
            "v1f0hid" => Some(Self { f0: true, hid: 384, ..Self::v1() }),
            "checkf0hid" => Some(Self { f0: true, hid: 24, ..Self::check() }),
            "v1f0pinv" => Some(Self { f0: true, range: 1, ..Self::v1() }),
            "v1f0phys" => Some(Self { f0: true, range: 2, ..Self::v1() }),
            "checkf0pinv" => Some(Self { f0: true, range: 1, n_cep: 12, ..Self::check() }),
            "checkf0phys" => Some(Self { f0: true, range: 2, n_cep: 12, ..Self::check() }),
            _ => None,
        }
    }

    /// rows of the "src" input: the frame-rate source (`n_mels + 2`), then
    /// the §29.5 harmonic source (re, im of `hsrc` bins)
    pub fn src_ch(&self) -> usize {
        self.n_mels + 2 + 2 * self.hsrc
    }

    /// rows of G's main input: the log-mel, or (`hid`) hidden + ones + ln gain
    pub fn in_ch(&self) -> usize {
        if self.hid > 0 { self.hid + 2 } else { self.n_mels }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Reflect padding each side for the "same" mel frontend.
    pub fn mel_pad(&self) -> usize {
        (self.n_fft - self.hop) / 2
    }

    /// Parameters in a fixed order: (name, shape, init std or constant, decay).
    pub fn params(&self) -> Vec<ParamSpec> {
        let (d, f, m, k) = (self.dim, self.ff, self.n_mels, self.bins());
        let ci = self.in_ch();
        let mut p = vec![
            ParamSpec::normal("embed.w", &[d, ci, self.embed_k], (1.0 / (ci * self.embed_k) as f64).sqrt()),
            ParamSpec::zeros("embed.b", &[d]),
            ParamSpec::ones("norm.g", &[d]),
            ParamSpec::zeros("norm.b", &[d]),
        ];
        let ls = 1.0 / self.layers as f64;
        for i in 0..self.layers {
            let n = |s: &str| format!("blocks.{i}.{s}");
            p.push(ParamSpec::normal(&n("dw.w"), &[d, self.dw_k], (1.0 / self.dw_k as f64).sqrt()));
            p.push(ParamSpec::zeros(&n("dw.b"), &[d]));
            p.push(ParamSpec::ones(&n("ln.g"), &[d]));
            p.push(ParamSpec::zeros(&n("ln.b"), &[d]));
            p.push(ParamSpec::normal(&n("pw1.w"), &[f, d], (1.0 / d as f64).sqrt()));
            p.push(ParamSpec::zeros(&n("pw1.b"), &[f]));
            p.push(ParamSpec::normal(&n("pw2.w"), &[d, f], (1.0 / f as f64).sqrt()));
            p.push(ParamSpec::zeros(&n("pw2.b"), &[d]));
            p.push(ParamSpec::constant(&n("gamma"), &[d], ls));
        }
        p.extend([
            ParamSpec::ones("final.g", &[d]),
            ParamSpec::zeros("final.b", &[d]),
            ParamSpec::normal("head.mag.w", &[k, d], 0.02),
            ParamSpec::zeros("head.mag.b", &[k]),
            ParamSpec::normal("head.ph.w", &[k, d], 0.02),
            ParamSpec::zeros("head.ph.b", &[k]),
        ]);
        if self.f0 {
            // zero init: a V1 checkpoint starts as the V1 function
            p.push(ParamSpec { name: "src.w".into(), shape: vec![d, m + 2, self.embed_k], init: Init::Const(0.0), decay: true });
        }
        if self.hsrc > 0 {
            assert!(self.f0, "hsrc needs f0");
            p.push(ParamSpec { name: "hsrc.w".into(), shape: vec![d, 2 * self.hsrc, self.embed_k], init: Init::Const(0.0), decay: true });
        }
        p
    }

    pub fn n_params(&self) -> usize {
        self.params().iter().map(|p| p.numel()).sum()
    }
}

#[derive(Clone, Debug)]
pub enum Init {
    Normal(f64),
    Const(f64),
}

#[derive(Clone, Debug)]
pub struct ParamSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub init: Init,
    pub decay: bool,
}

impl ParamSpec {
    pub fn normal(name: &str, shape: &[usize], std: f64) -> Self {
        Self { name: name.into(), shape: shape.to_vec(), init: Init::Normal(std), decay: true }
    }
    pub fn constant(name: &str, shape: &[usize], c: f64) -> Self {
        Self { name: name.into(), shape: shape.to_vec(), init: Init::Const(c), decay: false }
    }
    pub fn zeros(name: &str, shape: &[usize]) -> Self {
        Self::constant(name, shape, 0.0)
    }
    pub fn ones(name: &str, shape: &[usize]) -> Self {
        Self::constant(name, shape, 1.0)
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn init_data(&self, rng: &mut Rng) -> Vec<f32> {
        (0..self.numel())
            .map(|_| match self.init {
                Init::Normal(s) => (rng.normal() * s) as f32,
                Init::Const(c) => c as f32,
            })
            .collect()
    }
}

/// xorshift + Box–Muller; deterministic inits and crops.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// What a graph input holds.
#[derive(Clone, Debug)]
pub enum Role {
    /// Parameter `i` of `Config::params`.
    Param(usize),
    /// Adam first / second moment of parameter `i`.
    AdamM(usize),
    AdamV(usize),
    /// `[lr, 1−β₁ᵗ, 1−β₂ᵗ, weight decay, grad-clip norm, 0]`.
    Hyper,
    /// Discriminator parameter `i` of `Built::dspecs`, and its Adam moments.
    DParam(usize),
    DAdamM(usize),
    DAdamV(usize),
    /// Data fed per run (audio, mel).
    Data(&'static str),
    /// A constant table, uploaded once.
    Const(Vec<f32>),
}

#[derive(Clone, Debug)]
pub struct InputSpec {
    pub id: TensorId,
    pub role: Role,
}

/// A graph plus the meaning of each of its inputs (in creation order,
/// which is the order `Executable::new` expects).
pub struct Built {
    pub g: TensorGraph,
    pub inputs: Vec<InputSpec>,
    pub cfg: Config,
    /// Discriminator parameter specs (empty for generator-only graphs).
    pub dspecs: Vec<ParamSpec>,
}

impl Built {
    pub fn new(cfg: &Config) -> Self {
        Self { g: TensorGraph::new(), inputs: Vec::new(), cfg: cfg.clone(), dspecs: Vec::new() }
    }

    pub fn new_pub(cfg: &Config) -> Self {
        Self::new(cfg)
    }

    pub fn input(&mut self, shape: &[usize], role: Role) -> TensorId {
        let id = self.g.input(shape);
        self.inputs.push(InputSpec { id, role });
        id
    }

    pub fn constant(&mut self, shape: &[usize], data: Vec<f64>) -> TensorId {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        self.input(shape, Role::Const(data.iter().map(|&v| v as f32).collect()))
    }

    /// One vector per graph input, in input order. Params/state come from
    /// the given slices (zeros when absent), data by name.
    pub fn assemble(
        &self,
        params: &[Vec<f32>],
        adam: Option<(&[Vec<f32>], &[Vec<f32>])>,
        data: &[(&str, &[f32])],
    ) -> Vec<Vec<f32>> {
        self.assemble_gd(params, adam, &[], None, data)
    }

    /// `assemble` plus discriminator parameters / Adam state.
    pub fn assemble_gd(
        &self,
        params: &[Vec<f32>],
        adam: Option<(&[Vec<f32>], &[Vec<f32>])>,
        dparams: &[Vec<f32>],
        dadam: Option<(&[Vec<f32>], &[Vec<f32>])>,
        data: &[(&str, &[f32])],
    ) -> Vec<Vec<f32>> {
        let specs = self.cfg.params();
        let ds = &self.dspecs;
        self.inputs
            .iter()
            .map(|s| match &s.role {
                Role::Param(i) => params[*i].clone(),
                Role::DParam(i) => dparams.get(*i).cloned().unwrap_or_else(|| vec![0.0; ds[*i].numel()]),
                Role::DAdamM(i) => dadam.map_or_else(|| vec![0.0; ds[*i].numel()], |a| a.0[*i].clone()),
                Role::DAdamV(i) => dadam.map_or_else(|| vec![0.0; ds[*i].numel()], |a| a.1[*i].clone()),
                Role::AdamM(i) => adam.map_or_else(|| vec![0.0; specs[*i].numel()], |a| a.0[*i].clone()),
                Role::AdamV(i) => adam.map_or_else(|| vec![0.0; specs[*i].numel()], |a| a.1[*i].clone()),
                Role::Hyper => vec![0.0; 6],
                Role::Const(v) => v.clone(),
                Role::Data(name) => {
                    let n: usize = self.g.shape(s.id).iter().map(|d| d.n()).product();
                    data.iter()
                        .find(|(k, _)| k == name)
                        .map_or_else(|| vec![0.0; n], |(_, v)| v.to_vec())
                }
            })
            .collect()
    }

    /// Index of graph input `id` (its position in `inputs`).
    pub fn input_index(&self, id: TensorId) -> usize {
        self.inputs.iter().position(|s| s.id == id).unwrap()
    }

    pub fn find(&self, pred: impl Fn(&Role) -> bool) -> Option<TensorId> {
        self.inputs.iter().find(|s| pred(&s.role)).map(|s| s.id)
    }
}

/// Constant tables shared by the frontend, the head and the losses.
pub struct Consts {
    /// analysis bases per n_fft: (cos, −sin), `[n_fft, bins]`
    pub stft: Vec<(usize, TensorId, TensorId)>,
    pub fbank: TensorId,
    pub ib_c: TensorId,
    pub ib_s: TensorId,
    /// bin centre frequencies in Hz, `[bins]` (F0 source)
    pub freqs: TensorId,
}

/// Parameters as graph inputs, in `Config::params` order.
pub fn param_inputs(b: &mut Built) -> Vec<TensorId> {
    let specs = b.cfg.params();
    specs
        .iter()
        .enumerate()
        .map(|(i, s)| b.input(&s.shape, Role::Param(i)))
        .collect()
}

pub fn consts(b: &mut Built, analysis: &[usize], frames: usize) -> (Consts, TensorId) {
    let cfg = b.cfg.clone();
    let mut stft = Vec::new();
    for &n in analysis {
        if stft.iter().any(|(m, _, _)| *m == n) {
            continue;
        }
        let (c, s) = dsp::stft_bases(n);
        let bins = n / 2 + 1;
        let ci = b.constant(&[n, bins], c);
        let si = b.constant(&[n, bins], s);
        stft.push((n, ci, si));
    }
    let fbank = b.constant(&[cfg.bins(), cfg.n_mels], dsp::mel_fbank(cfg.sr, cfg.n_fft, cfg.n_mels));
    let (c, s) = dsp::istft_bases(cfg.n_fft);
    let ib_c = b.constant(&[cfg.bins(), cfg.n_fft], c);
    let ib_s = b.constant(&[cfg.bins(), cfg.n_fft], s);
    let env = b.constant(&[frames, cfg.hop], dsp::istft_inv_env(cfg.n_fft, cfg.hop, frames));
    let df = cfg.sr as f64 / cfg.n_fft as f64;
    let freqs = b.constant(&[cfg.bins()], (0..cfg.bins()).map(|k| k as f64 * df).collect());
    (Consts { stft, fbank, ib_c, ib_s, freqs }, env)
}

impl Consts {
    pub fn bases(&self, n: usize) -> (TensorId, TensorId) {
        let (_, c, s) = self.stft.iter().find(|(m, _, _)| *m == n).unwrap();
        (*c, *s)
    }
}

/// `|STFT(x)|` (Hann, no centering) as `[B, frames, bins]`, framed
/// polyphase: `x` viewed as `[B, L/hop, hop]`, frame `t` = rows `t..t+J`.
/// `sqrt(re² + im² + eps)`.
pub fn stft_mag(
    g: &mut TensorGraph,
    x: TensorId,
    n_fft: usize,
    hop: usize,
    bases: (TensorId, TensorId),
    eps: f64,
) -> TensorId {
    let (re, im) = stft_ri(g, x, n_fft, hop, bases);
    g.map(&[re, im], |v| (v[0] * v[0] + v[1] * v[1] + lit(eps)).sqrt()).unwrap()
}

/// `STFT(x)` (Hann, no centering) as `(re, im)`, each `[B, frames, bins]`.
pub fn stft_ri(
    g: &mut TensorGraph,
    x: TensorId,
    n_fft: usize,
    hop: usize,
    (bc, bs): (TensorId, TensorId),
) -> (TensorId, TensorId) {
    let [b, l] = ir::dims(g, x)[..] else { panic!() };
    assert_eq!(l % hop, 0);
    assert_eq!(n_fft % hop, 0);
    let (q, j) = (l / hop, n_fft / hop);
    let frames = q - j + 1;
    let bins = n_fft / 2 + 1;
    let x3 = g.reshape(x, &[b, q, hop]).unwrap();
    // unfold u[b, t, j, r] = x3[b, t + j, r]  (guarded: see `ir`)
    let uv = view(
        vec![form(4, &[(0, 1)], 0), form(4, &[(1, 1), (2, 1)], 0), form(4, &[(3, 1)], 0)],
        true,
    );
    let u = g
        .comp(&[b, frames, j, hop], &[], vec![arg(x3, uv)], None, |v| vec![v[0]])
        .unwrap();
    let u = g.reshape(u, &[b, frames, n_fft]).unwrap();
    let dft = |g: &mut TensorGraph, basis: TensorId| {
        // space (b, t, k, n)
        g.comp(
            &[b, frames, bins],
            &[n_fft],
            vec![
                arg(u, pick(4, &[Some(0), Some(1), Some(3)])),
                arg(basis, pick(4, &[Some(3), Some(2)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap()
    };
    (dft(g, bc), dft(g, bs))
}

/// `ln(max(fbᵀ·|STFT|, 1e-5))` as `[B, n_mels, frames]` (no padding: pass a
/// reflect-padded signal for the model's "same" frontend).
pub fn log_mel(g: &mut TensorGraph, cfg: &Config, c: &Consts, x: TensorId, eps: f64) -> TensorId {
    let mag = stft_mag(g, x, cfg.n_fft, cfg.hop, c.bases(cfg.n_fft), eps);
    let [b, f, k] = ir::dims(g, mag)[..] else { panic!() };
    // space (b, m, t, k)
    let mel = g
        .comp(
            &[b, cfg.n_mels, f],
            &[k],
            vec![
                arg(mag, pick(4, &[Some(0), Some(2), Some(3)])),
                arg(c.fbank, pick(4, &[Some(3), Some(1)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    g.map(&[mel], |v| {
        let floor = lit(1e-5);
        ExprId::select(v[0] - floor, v[0], floor).ln()
    })
    .unwrap()
}

/// `conv1d_nobias` over input rows `c0 .. c0 + C` of `x` (C from `w [O, C, K]`).
fn conv1d_rows(g: &mut TensorGraph, x: TensorId, w: TensorId, c0: i64) -> TensorId {
    let [b, _, t] = ir::dims(g, x)[..] else { panic!() };
    let [co, ci, k] = ir::dims(g, w)[..] else { panic!() };
    let p = (k as i64 - 1) / 2;
    let xv = view(vec![form(5, &[(0, 1)], 0), form(5, &[(3, 1)], c0), form(5, &[(2, 1), (4, 1)], -p)], true);
    g.comp(&[b, co, t], &[ci, k], vec![arg(x, xv), arg(w, pick(5, &[Some(1), Some(3), Some(4)]))], Some(Monoid::sum()), |v| vec![v[0] * v[1]]).unwrap()
}

/// mel `[B, n_mels, T]` → audio `[B, T·hop]`.
fn vocoder(
    g: &mut TensorGraph,
    cfg: &Config,
    c: &Consts,
    env: TensorId,
    p: &[TensorId],
    mel: TensorId,
    f0: Option<TensorId>,
    phys: Option<&PhysIn>,
) -> (TensorId, Vec<TensorId>, TensorId) {
    let mut resid = Vec::new();
    let mut it = p.iter().copied();
    let mut next = || it.next().unwrap();
    let (ew, eb, ng, nb) = (next(), next(), next(), next());
    let mut x = ir::conv1d(g, mel, ew, eb);
    if cfg.f0 {
        let src = f0.expect("cfg.f0 needs a source input");
        let ks = p.len() - 1 - (cfg.hsrc > 0) as usize;
        let y = if cfg.hsrc > 0 { conv1d_rows(g, src, p[ks], 0) } else { ir::conv1d_nobias(g, src, p[ks]) };
        x = g.map(&[x, y], |v| v[0] + v[1]).unwrap();
        if cfg.hsrc > 0 {
            // §29.5: the harmonic source rows follow the frame-rate source's
            let y = conv1d_rows(g, src, p[p.len() - 1], (cfg.n_mels + 2) as i64);
            x = g.map(&[x, y], |v| v[0] + v[1]).unwrap();
        }
    }
    x = ir::layer_norm(g, x, ng, nb, 1e-6);
    for _ in 0..cfg.layers {
        resid.push(x);
        let (dw, db, lg, lb, w1, b1, w2, b2, gm) =
            (next(), next(), next(), next(), next(), next(), next(), next(), next());
        let h = ir::dwconv(g, x, dw, db);
        let h = ir::layer_norm(g, h, lg, lb, 1e-6);
        let h = ir::linear(g, h, w1);
        let h = ir::channel_map(g, h, &[b1], |v, q| ir::gelu(v + q[0]));
        let h = ir::linear(g, h, w2);
        let sh = ir::dims(g, h);
        x = g
            .comp(
                &sh,
                &[],
                vec![
                    arg(h, View::identity(3, 3)),
                    arg(x, View::identity(3, 3)),
                    arg(b2, pick(3, &[Some(1)])),
                    arg(gm, pick(3, &[Some(1)])),
                ],
                None,
                |v| vec![v[1] + v[3] * (v[0] + v[2])],
            )
            .unwrap();
    }
    let (fg, fb, mw, mb, pw, pb) = (next(), next(), next(), next(), next(), next());
    x = ir::layer_norm(g, x, fg, fb, 1e-6);
    let a = ir::linear(g, x, mw);
    let a = ir::add_channel(g, a, mb);
    let ph = ir::linear(g, x, pw);
    let ph = ir::add_channel(g, ph, pb);
    resid.push(x);
    if cfg.range > 0 {
        let p = phys.expect("range > 0 needs PhysIn");
        let r = match cfg.range {
            1 => glottis::pinv_range(g, &p.pc, mel),
            _ => glottis::phys_mag(g, cfg, &p.pc, p.th, p.f0hz),
        };
        let ln100 = 100f64.ln();
        let cand = g.map(&[r, a], |v| v[0] * ExprId::select(v[1] - lit(ln100), lit(ln100), v[1]).exp()).unwrap();
        let mut keep = vec![r, cand];
        let s = glottis::project_keep(g, &p.pc, cand, p.target, &mut keep);
        resid.extend(keep);
        let re = g.map(&[s, ph], |v| v[0] * v[1].cos()).unwrap();
        let im = g.map(&[s, ph], |v| v[0] * v[1].sin()).unwrap();
        resid.push(s);
        return (istft(g, cfg, c.ib_c, c.ib_s, env, re, im), resid, ph);
    }
    // Vocos head: mag = clip(exp(a), max=100), S = mag·(cos φ + i sin φ)
    let ln100 = 100f64.ln();
    let re = g
        .map(&[a, ph], |v| {
            ExprId::select(v[0] - lit(ln100), lit(ln100), v[0]).exp() * v[1].cos()
        })
        .unwrap();
    let im = g
        .map(&[a, ph], |v| {
            ExprId::select(v[0] - lit(ln100), lit(ln100), v[0]).exp() * v[1].sin()
        })
        .unwrap();
    (istft(g, cfg, c.ib_c, c.ib_s, env, re, im), resid, ph)
}

/// APNet anti-wrapping phase losses (Ai & Ling 2023, §III-C2; eqs. 21–24):
/// `[L_IP, L_GD, L_PTD]`, each `mean(1 − cos Δ)` (= APNet's `−cos` + 1).
/// `ph` is the head's raw phase `[B, bins, T]`; the target phase comes from
/// `(re, im)` `[B, T, bins]` of the real audio on the vocoder's own framing.
/// No atan2 (tang IR gap): with e = φ̂ − φ,
/// `cos e = (R cos φ̂ + I sin φ̂)/|S|`, `sin e = (R sin φ̂ − I cos φ̂)/|S|`, and
/// the GD/PTD terms `cos(e_a − e_b) = cos e_a cos e_b + sin e_a sin e_b`.
pub fn phase_losses(g: &mut TensorGraph, ph: TensorId, re: TensorId, im: TensorId, weighted: bool) -> [TensorId; 3] {
    let [b, k, t] = ir::dims(g, ph)[..] else { panic!() };
    let tp = pick(3, &[Some(0), Some(2), Some(1)]);
    let id = View::identity(3, 3);
    // unweighted (APNet): divide by sqrt(|S|² + 1e-8), so near-silent bins
    // count less by construction; weighted: unit phasors, |S|-weighted means
    let eps = if weighted { 1e-12 } else { 1e-8 };
    let ce_se = |g: &mut TensorGraph, sin: bool| {
        g.comp(&[b, k, t], &[], vec![arg(ph, id.clone()), arg(re, tp.clone()), arg(im, tp.clone())], None, move |v| {
            let n = (v[1] * v[1] + v[2] * v[2] + lit(eps)).sqrt();
            let (c, s) = (v[0].cos(), v[0].sin());
            vec![if sin { (v[1] * s - v[2] * c) / n } else { (v[1] * c + v[2] * s) / n }]
        })
        .unwrap()
    };
    let ce = ce_se(g, false);
    let se = ce_se(g, true);
    let mag = weighted.then(|| {
        g.comp(&[b, k, t], &[], vec![arg(re, tp.clone()), arg(im, tp.clone())], None, |v| vec![(v[0] * v[0] + v[1] * v[1] + lit(1e-12)).sqrt()]).unwrap()
    });
    // mean of (1 − c) over `sh`, or Σ w(1 − c) / Σ w with w from `w_of`
    let mean1m = |g: &mut TensorGraph, c: TensorId, w: Option<TensorId>| {
        let sh = ir::dims(g, c);
        match w {
            None => {
                let n = sh.iter().product::<usize>() as f64;
                ir::sum_all2(g, c, None, |v| (lit(1.0) - v[0]) * lit(1.0 / n))
            }
            Some(w) => {
                let num = ir::sum_all2(g, c, Some(w), |v| (lit(1.0) - v[0]) * v[1]);
                let den = ir::sum_all2(g, w, None, |v| v[0]);
                g.map(&[num, den], |v| v[0] / (v[1] + lit(1e-12))).unwrap()
            }
        }
    };
    let ip = mean1m(g, ce, mag);
    // shifted pair along axis `ax` (1 = freq: group delay, 2 = time: inst. freq.)
    let diff = |g: &mut TensorGraph, ax: usize| {
        let mut sh = vec![b, k, t];
        sh[ax] -= 1;
        // both views guarded: their adjoints read the shorter cos_d grad
        let at = |o: i64| view((0..3).map(|d| form(3, &[(d, 1)], if d == ax { o } else { 0 })).collect(), true);
        let (cur, nxt) = (at(0), at(1));
        let cos_d = g
            .comp(&sh, &[], vec![arg(ce, cur.clone()), arg(ce, nxt.clone()), arg(se, cur.clone()), arg(se, nxt.clone())], None, |v| vec![v[0] * v[1] + v[2] * v[3]])
            .unwrap();
        // pair weight: geometric mean of the two bins' magnitudes
        let w = mag.map(|m| g.comp(&sh, &[], vec![arg(m, cur), arg(m, nxt)], None, |v| vec![(v[0] * v[1]).sqrt()]).unwrap());
        mean1m(g, cos_d, w)
    };
    let gd = diff(g, 1);
    let ptd = diff(g, 2);
    [ip, gd, ptd]
}

/// F0 below this is treated as unvoiced (the estimator writes 0 there).
pub const F0_MIN: f64 = 40.0;

/// Source features `[B, n_mels + 2, T]` from per-frame f0 (Hz, 0 = unvoiced),
/// computed on the host (`dsp::source_feats`) and fed as data: the same
/// features built in the graph gave inf `src` gradients on CUDA at V1 size
/// (nondeterministic; finite without them; docs §10.2).
/// Vocos "same" iSTFT of `re + i·im` (`[B, bins, T]`) → `[B, T·hop]`:
/// irfft·window as one contraction over bins, then polyphase overlap-add
/// divided by the window envelope.
pub fn istft(
    g: &mut TensorGraph,
    cfg: &Config,
    ib_c: TensorId,
    ib_s: TensorId,
    env: TensorId,
    re: TensorId,
    im: TensorId,
) -> TensorId {
    let [b, _, t] = ir::dims(g, re)[..] else { panic!() };
    // irfft·window as one contraction over bins: frame[b, t, n]
    let (n_fft, hop, bins) = (cfg.n_fft, cfg.hop, cfg.bins());
    let frame = g
        .comp(
            &[b, t, n_fft],
            &[bins],
            vec![
                arg(re, pick(4, &[Some(0), Some(3), Some(1)])),
                arg(ib_c, pick(4, &[Some(3), Some(2)])),
                arg(im, pick(4, &[Some(0), Some(3), Some(1)])),
                arg(ib_s, pick(4, &[Some(3), Some(2)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1] - v[2] * v[3]],
        )
        .unwrap();
    // overlap-add, polyphase: y[b, q, r] = Σ_j frame[b, q−j+c0, r+base+hop·j] / env
    let o = dsp::ola(n_fft, hop);
    let fv = view(
        vec![
            form(4, &[(0, 1)], 0),
            form(4, &[(1, 1), (3, -1)], o.c0),
            form(4, &[(2, 1), (3, hop as i64)], o.base),
        ],
        true,
    );
    let y = g
        .comp(
            &[b, t, hop],
            &[o.taps],
            vec![arg(frame, fv), arg(env, pick(4, &[Some(1), Some(2)]))],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    g.reshape(y, &[b, t * hop]).unwrap()
}

/// Inference: mel `[1, n_mels, frames]` → audio. Inputs: params, consts, mel.
pub struct Infer {
    pub b: Built,
    pub mel: TensorId,
    /// `[1, 1, frames]` Hz (only when `cfg.f0`)
    pub f0: Option<TensorId>,
    pub audio: TensorId,
    /// range > 0: `[max |mel(s) − target|]` (log-mel units) and the
    /// edited target mel `[1, n_mels, frames]`
    pub mel_err: Option<TensorId>,
    pub target: Option<TensorId>,
}

/// iSTFT only (for the round-trip test): inputs re, im `[B, bins, T]`.
pub struct IstftGraph {
    pub b: Built,
    pub re: TensorId,
    pub im: TensorId,
    pub audio: TensorId,
}

pub fn istft_graph(cfg: &Config, batch: usize, frames: usize) -> IstftGraph {
    let mut b = Built::new(cfg);
    let (c, env) = consts(&mut b, &[], frames);
    let re = b.input(&[batch, cfg.bins(), frames], Role::Data("re"));
    let im = b.input(&[batch, cfg.bins(), frames], Role::Data("im"));
    let audio = istft(&mut b.g, cfg, c.ib_c, c.ib_s, env, re, im);
    IstftGraph { b, re, im, audio }
}

pub fn infer_graph(cfg: &Config, batch: usize, frames: usize) -> Infer {
    let mut b = Built::new(cfg);
    let p = param_inputs(&mut b);
    let (c, env) = consts(&mut b, &[], frames);
    let mel = b.input(&[batch, cfg.in_ch(), frames], Role::Data("mel"));
    let f0 = cfg.f0.then(|| b.input(&[batch, cfg.src_ch(), frames], Role::Data("src")));
    if cfg.range == 0 {
        let audio = vocoder(&mut b.g, cfg, &c, env, &p, mel, f0, None).0;
        return Infer { b, mel, f0, audio, mel_err: None, target: None };
    }
    let mut pi = phys_inputs(&mut b, batch, frames, mel).unwrap();
    let mut target = mel;
    if cfg.range == 2 {
        // knob edit: mel' = mel + logmel(phys(θ)) − logmel(phys(θ₀))
        let np = glottis::n_params(cfg);
        let th0 = b.input(&[batch, np, frames], Role::Data("phys0"));
        let g = &mut b.g;
        let x1 = glottis::phys_mag(g, cfg, &pi.pc, pi.th, pi.f0hz);
        let x0 = glottis::phys_mag(g, cfg, &pi.pc, th0, pi.f0hz);
        let l1 = glottis::log_mel_of(g, &pi.pc, x1);
        let l0 = glottis::log_mel_of(g, &pi.pc, x0);
        target = g.map(&[mel, l1, l0], |v| {
            let m = v[0] + v[1] - v[2];
            let fl = lit(1e-5f64.ln());
            ExprId::select(m - fl, m, fl)
        }).unwrap();
        pi.target = target;
    }
    let (audio, resid, _) = vocoder(&mut b.g, cfg, &c, env, &p, target, f0, Some(&pi));
    let s = *resid.last().unwrap();
    let lm = glottis::log_mel_of(&mut b.g, &pi.pc, s);
    let err = glottis::max_abs_diff(&mut b.g, lm, target);
    Infer { b, mel, f0, audio, mel_err: Some(err), target: Some(target) }
}

/// Frontend only: padded audio `[B, frames·hop + 2·pad]` → log-mel.
pub struct Frontend {
    pub b: Built,
    pub audio: TensorId,
    pub mel: TensorId,
}

pub fn frontend_graph(cfg: &Config, batch: usize, frames: usize) -> Frontend {
    let mut b = Built::new(cfg);
    let (c, _) = consts(&mut b, &[cfg.n_fft], frames);
    let audio = b.input(&[batch, frames * cfg.hop + 2 * cfg.mel_pad()], Role::Data("audio_pad"));
    let mel = log_mel(&mut b.g, cfg, &c, audio, 0.0);
    Frontend { b, audio, mel }
}

/// Named scalar outputs of the training graph.
pub struct Losses {
    pub total: TensorId,
    pub mrstft: TensorId,
    pub mel_l1: TensorId,
    pub sc: Vec<TensorId>,
    pub logmag: Vec<TensorId>,
    /// `[L_IP, L_GD, L_PTD]` (`loss_graph_phase` only)
    pub phase: Option<[TensorId; 3]>,
}

/// Forward + loss (no optimizer): used by the gradient check and the eval.
pub struct LossGraph {
    pub b: Built,
    pub params: Vec<TensorId>,
    pub audio_pad: TensorId,
    pub audio: TensorId,
    pub f0: Option<TensorId>,
    /// `loss_graph_ext`: (mel_ext, mel_sel) data inputs `[B, n_mels, T]`; G reads
    /// `mel + sel·(mel_ext − mel)` (vocoder fine-tune on predicted mels, §16)
    pub mel_ext: Option<(TensorId, TensorId)>,
    pub y: TensorId,
    pub consts: Consts,
    /// The residual stream entering each block (and the final norm).
    pub resid: Vec<TensorId>,
    pub losses: Losses,
}

pub fn loss_graph(cfg: &Config, batch: usize, frames: usize) -> LossGraph {
    loss_graph_ext(cfg, batch, frames, false)
}

/// `loss_graph`, optionally with an external input mel per item: with `ext`,
/// G vocodes `mel + sel·(mel_ext − mel)` (sel = 1: an acoustic model's
/// predicted mel; sel = 0: the frontend mel of the crop), while every loss
/// still targets the real audio.
pub fn loss_graph_ext(cfg: &Config, batch: usize, frames: usize, ext: bool) -> LossGraph {
    loss_graph_phase(cfg, batch, frames, ext, false, false)
}

/// `loss_graph_ext`, plus the APNet phase losses when `phase` (§29.3).
pub fn loss_graph_phase(cfg: &Config, batch: usize, frames: usize, ext: bool, phase: bool, phase_w: bool) -> LossGraph {
    let mut b = Built::new(cfg);
    let params = param_inputs(&mut b);
    let mut analysis: Vec<usize> = cfg.loss_res.iter().map(|r| r.0).collect();
    analysis.push(cfg.n_fft);
    let (c, env) = consts(&mut b, &analysis, frames);
    let n = frames * cfg.hop;
    let audio_pad = b.input(&[batch, n + 2 * cfg.mel_pad()], Role::Data("audio_pad"));
    let audio = b.input(&[batch, n], Role::Data("audio"));
    let f0 = cfg.f0.then(|| b.input(&[batch, cfg.src_ch(), frames], Role::Data("src")));
    assert!(cfg.hid == 0 || ext, "a hidden-input G needs the ext graph");
    let mel_ext = ext.then(|| (b.input(&[batch, cfg.in_ch(), frames], Role::Data("mel_ext")), b.input(&[batch, cfg.in_ch(), frames], Role::Data("mel_sel"))));
    let g = &mut b.g;
    let mel_in = match mel_ext {
        // hidden input: sel is a 0/1 mask; there is no frontend fallback
        Some((me, ms)) if cfg.hid > 0 => g.map(&[me, ms], |v| v[0] * v[1]).unwrap(),
        Some((me, ms)) => {
            let fm = log_mel(g, cfg, &c, audio_pad, 0.0);
            g.map(&[fm, me, ms], |v| v[0] + v[2] * (v[1] - v[0])).unwrap()
        }
        None => log_mel(g, cfg, &c, audio_pad, 0.0),
    };
    let pi = phys_inputs(&mut b, batch, frames, mel_in);
    let g = &mut b.g;
    let (y, resid, ph) = vocoder(g, cfg, &c, env, &params, mel_in, f0, pi.as_ref());

    let mut sc = Vec::new();
    let mut logmag = Vec::new();
    for &(nf, hop) in &cfg.loss_res {
        let ym = stft_mag(g, y, nf, hop, c.bases(nf), 1e-10);
        let xm = stft_mag(g, audio, nf, hop, c.bases(nf), 1e-10);
        let count = ir::dims(g, ym).iter().product::<usize>() as f64;
        let num = ir::sum_all2(g, ym, Some(xm), |v| (v[0] - v[1]) * (v[0] - v[1]));
        let den = ir::sum_all2(g, xm, None, |v| v[0] * v[0]);
        let s = g.map(&[num, den], |v| (v[0] / (v[1] + lit(1e-12))).sqrt()).unwrap();
        let l = ir::sum_all2(g, ym, Some(xm), |v| ir::abs(v[0].ln() - v[1].ln()) * lit(1.0 / count));
        sc.push(s);
        logmag.push(l);
    }
    let r = cfg.loss_res.len() as f64;
    let mut parts = sc.clone();
    parts.extend(logmag.iter().copied());
    let mrstft = g
        .map(&parts, |v| v.iter().fold(ExprId::ZERO, |a, &x| a + x) * lit(1.0 / r))
        .unwrap();
    let ym = log_mel(g, cfg, &c, y, 1e-10);
    let xm = log_mel(g, cfg, &c, audio, 1e-10);
    let count = ir::dims(g, ym).iter().product::<usize>() as f64;
    let mel_l1 = ir::sum_all2(g, ym, Some(xm), |v| ir::abs(v[0] - v[1]) * lit(1.0 / count));
    let w = cfg.mel_weight;
    let total = g.map(&[mrstft, mel_l1], |v| v[0] + lit(w) * v[1]).unwrap();
    let phase = phase.then(|| {
        let (re, im) = stft_ri(g, audio_pad, cfg.n_fft, cfg.hop, c.bases(cfg.n_fft));
        phase_losses(g, ph, re, im, phase_w)
    });
    LossGraph {
        b,
        params,
        audio_pad,
        audio,
        f0,
        mel_ext,
        y,
        consts: c,
        resid,
        losses: Losses { total, mrstft, mel_l1, sc, logmag, phase },
    }
}

/// One training step as one program: forward, losses, backward for every
/// parameter, global-norm clip and AdamW. Outputs `p'`, `m'`, `v'` per
/// parameter, which the runtime ping-pongs into the inputs.
pub struct TrainGraph {
    pub lg: LossGraph,
    pub hyper: TensorId,
    pub m: Vec<TensorId>,
    pub v: Vec<TensorId>,
    pub new_p: Vec<TensorId>,
    pub new_m: Vec<TensorId>,
    pub new_v: Vec<TensorId>,
    pub grad_norm: TensorId,
    pub outputs: Vec<TensorId>,
}

pub const BETA1: f64 = 0.8;
pub const BETA2: f64 = 0.99;
pub const ADAM_EPS: f64 = 1e-8;

pub fn train_graph(cfg: &Config, batch: usize, frames: usize) -> TrainGraph {
    // graph inputs are numbered by creation order, not node position, so
    // the optimizer state can be appended after the loss graph's comps
    let mut lg = loss_graph(cfg, batch, frames);
    let specs = cfg.params();
    let np = specs.len();
    let mut m = Vec::new();
    let mut v = Vec::new();
    for (i, s) in specs.iter().enumerate() {
        m.push(lg.b.input(&s.shape, Role::AdamM(i)));
    }
    for (i, s) in specs.iter().enumerate() {
        v.push(lg.b.input(&s.shape, Role::AdamV(i)));
    }
    let hyper = lg.b.input(&[6], Role::Hyper);
    let g = &mut lg.b.g;
    // Gradients w.r.t. the residual stream too, kept as outputs: otherwise
    // the backward residual chain (dx += …, eight blocks deep) fuses into
    // one kernel that binds more than Metal's 31 buffers.
    let mut wrt = lg.params.clone();
    wrt.extend(lg.resid.iter().copied());
    let mut grads = g.backward(lg.losses.total, &wrt).unwrap();
    let resid_grads = grads.split_off(np);
    let ad = adamw(g, &specs, &lg.params, &grads, &m, &v, hyper);
    let (new_p, new_m, new_v, grad_norm, extra_outputs) = (ad.new_p, ad.new_m, ad.new_v, ad.grad_norm, ad.partials);
    let l = &lg.losses;
    let mut outputs = vec![l.total, l.mrstft, l.mel_l1, grad_norm];
    outputs.extend(l.sc.iter().copied());
    outputs.extend(l.logmag.iter().copied());
    outputs.extend(new_p.iter().copied());
    outputs.extend(new_m.iter().copied());
    outputs.extend(new_v.iter().copied());
    outputs.extend(extra_outputs);
    outputs.extend(resid_grads);
    TrainGraph { lg, hyper, m, v, new_p, new_m, new_v, grad_norm, outputs }
}

/// `Σ x²` as `[1]`. Rank ≥ 2 goes in two stages (last dim first, many
/// points): a one-stage reduction to one point inlines the producing
/// gradient reduction as a serial loop (docs §10.2 gap 10), and at V1.1
/// size on CUDA it returned inf/garbage (|g| ~1e9 while the per-term norms
/// were ~1e2), which made clipping zero G's updates.
pub fn sq_norm(g: &mut TensorGraph, x: TensorId) -> TensorId {
    let sh = ir::dims(g, x);
    let r = sh.len();
    if r < 2 {
        return ir::sum_all2(g, x, None, |v| v[0] * v[0]);
    }
    let (outer, inner) = sh.split_at(r - 1);
    let picks: Vec<Option<usize>> = (0..r).map(Some).collect();
    let s1 = g.comp(outer, inner, vec![arg(x, pick(r, &picks))], Some(Monoid::sum()), |v| vec![v[0] * v[0]]).unwrap();
    ir::sum_all2(g, s1, None, |v| v[0])
}

/// AdamW with global-norm clipping, as graph nodes.
pub struct Adam {
    pub new_p: Vec<TensorId>,
    pub new_m: Vec<TensorId>,
    pub new_v: Vec<TensorId>,
    pub grad_norm: TensorId,
    /// grad-norm partial sums: keep these as program outputs (see below)
    pub partials: Vec<TensorId>,
}

/// `hyper = [lr, 1−β₁ᵗ, 1−β₂ᵗ, wd, clip, 0]`.
pub fn adamw(
    g: &mut TensorGraph,
    specs: &[ParamSpec],
    params: &[TensorId],
    grads: &[TensorId],
    m: &[TensorId],
    v: &[TensorId],
    hyper: TensorId,
) -> Adam {
    adamw_betas(g, specs, params, grads, m, v, hyper, BETA1, BETA2)
}

/// [`adamw`] with explicit betas (the GAN steps use Vocos' 0.8 / 0.9).
#[allow(clippy::too_many_arguments)]
pub fn adamw_betas(
    g: &mut TensorGraph,
    specs: &[ParamSpec],
    params: &[TensorId],
    grads: &[TensorId],
    m: &[TensorId],
    v: &[TensorId],
    hyper: TensorId,
    beta1: f64,
    beta2: f64,
) -> Adam {
    let np = specs.len();
    let lg_params = params;
    // global grad norm
    let sq: Vec<TensorId> = grads
        .iter()
        .map(|&gr| sq_norm(g, gr))
        .collect();
    // Partial sums in groups of 8, each kept as a graph output so it is
    // materialized: one map over all ~80 squares would fuse into a single
    // kernel reading >31 buffers, past Metal's argument-table limit (an MSL
    // emitter gap, see the ribt paper "V1 status").
    let mut partials: Vec<TensorId> = sq
        .chunks(8)
        .map(|c| g.map(c, |v| v.iter().fold(ExprId::ZERO, |a, &x| a + x)).unwrap())
        .collect();
    let extra_outputs = partials.clone();
    while partials.len() > 8 {
        partials = partials
            .chunks(8)
            .map(|c| g.map(c, |v| v.iter().fold(ExprId::ZERO, |a, &x| a + x)).unwrap())
            .collect();
    }
    let gn2 = g.map(&partials, |v| v.iter().fold(ExprId::ZERO, |a, &x| a + x)).unwrap();
    let grad_norm = g.map(&[gn2], |v| v[0].sqrt()).unwrap();
    // scale = min(1, clip / ‖g‖)
    let scale = g
        .comp(
            &[1],
            &[],
            vec![
                arg(gn2, View::identity(1, 1)),
                arg(hyper, view(vec![form(1, &[], 4)], false)),
            ],
            None,
            |v| {
                let r = v[1] / (v[0].sqrt() + lit(1e-12));
                vec![ExprId::select(r - lit(1.0), lit(1.0), r)]
            },
        )
        .unwrap();
    let (mut new_p, mut new_m, mut new_v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..np {
        let r = specs[i].shape.len();
        let id = View::identity(r, r);
        let hy = |k: i64| view(vec![form(r, &[], k)], false);
        let sc = view(vec![form(r, &[], 0)], false);
        let m2 = g
            .comp(
                &specs[i].shape,
                &[],
                vec![arg(m[i], id.clone()), arg(grads[i], id.clone()), arg(scale, sc.clone())],
                None,
                |x| vec![x[0] * lit(beta1) + x[1] * x[2] * lit(1.0 - beta1)],
            )
            .unwrap();
        let v2 = g
            .comp(
                &specs[i].shape,
                &[],
                vec![arg(v[i], id.clone()), arg(grads[i], id.clone()), arg(scale, sc)],
                None,
                |x| {
                    let gs = x[1] * x[2];
                    vec![x[0] * lit(beta2) + gs * gs * lit(1.0 - beta2)]
                },
            )
            .unwrap();
        let decay = specs[i].decay;
        let p2 = g
            .comp(
                &specs[i].shape,
                &[],
                vec![
                    arg(lg_params[i], id.clone()),
                    arg(m2, id.clone()),
                    arg(v2, id),
                    arg(hyper, hy(0)),
                    arg(hyper, hy(1)),
                    arg(hyper, hy(2)),
                    arg(hyper, hy(3)),
                ],
                None,
                |x| {
                    let (p, m, v, lr, bc1, bc2, wd) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
                    let upd = (m / bc1) / ((v / bc2).sqrt() + lit(ADAM_EPS));
                    let upd = if decay { upd + wd * p } else { upd };
                    vec![p - lr * upd]
                },
            )
            .unwrap();
        new_m.push(m2);
        new_v.push(v2);
        new_p.push(p2);
    }
    Adam { new_p, new_m, new_v, grad_norm, partials: extra_outputs }
}

/// Fresh parameters for `cfg` (deterministic in `seed`).
pub fn init_params(cfg: &Config, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    cfg.params().iter().map(|s| s.init_data(&mut rng)).collect()
}
