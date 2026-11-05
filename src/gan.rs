//! V1.1 adversarial training: a multi-period discriminator (HiFi-GAN MPD)
//! and a multi-resolution STFT discriminator (UnivNet / Vocos MRD), the
//! hinge + feature-matching losses, and the two step programs
//! (generator step, discriminator step), all in the tensor IR with the
//! derived backward.
//!
//! Strided convs are written polyphase so every view stays a unit-stride
//! guarded offset (tang's backward inverts those by elimination; a strided
//! view would differentiate into a gated scatter, `ir_gaps scatter`):
//! a stride-3 conv with 6 taps over rows reads `x[3t + 3q + r − 3]`
//! = `x3[t + q − 1, r]` with `x3 = x` reshaped `[R/3, 3]`. Taps are 6 (not
//! HiFi-GAN's 5) and freq taps 10 (not 9) so they split evenly.

use tang::Scalar;
use tang_compile::tensor::{Monoid, TensorGraph, TensorId};
use tang_expr::ExprId;

use crate::ir::{self, arg, form, lit, pick, view};
use crate::model::{self, Built, Config, LossGraph, ParamSpec, Role};

#[derive(Clone, Debug)]
pub struct DConfig {
    pub periods: Vec<usize>,
    /// output channels of the 4 stride-3 layers and the stride-1 layer
    pub mpd_ch: Vec<usize>,
    /// `(n_fft, hop)` per MRD resolution
    pub mrd_res: Vec<(usize, usize)>,
    pub mrd_ch: usize,
    /// weight norm on every conv (`w = g·v/‖v‖` per output channel)
    pub wn: bool,
    pub bias: bool,
    /// §29.5: the MRD's first conv also sees the phase differences of the
    /// spectrum (instantaneous frequency along time, group delay along
    /// frequency) as two more (re, im) channel pairs (`phase_feats`)
    pub mrd_phase: bool,
}

const MPD_STRIDED: usize = 4;

impl DConfig {
    /// V1.2 at HiFi-GAN / Vocos widths: MPD 32/128/512/1024/1024, MRD 32
    /// channels on the complex spectrum (re, im) at hop n_fft/4, weight
    /// norm, biases. Vocos' MRD splits the spectrum into 5 bands with their
    /// own conv stacks; this one runs a single stack over all bins (UnivNet).
    pub fn v12() -> Self {
        Self { periods: vec![2, 3, 5, 7, 11], mpd_ch: vec![32, 128, 512, 1024, 1024], mrd_res: vec![(512, 128), (1024, 256), (2048, 512)], mrd_ch: 32, wn: true, bias: true, mrd_phase: false }
    }
    /// half the MPD widths of `v12`
    pub fn v12h() -> Self {
        Self { mpd_ch: vec![32, 64, 256, 512, 512], ..Self::v12() }
    }
    /// V1.1's widths (MPD ¼ HiFi-GAN, MRD 16) with the V1.2 layers
    pub fn v1() -> Self {
        Self { mpd_ch: vec![32, 64, 128, 256, 256], mrd_ch: 16, ..Self::v12() }
    }
    pub fn tiny() -> Self {
        Self { periods: vec![2, 3], mpd_ch: vec![2, 3, 2, 3, 2], mrd_res: vec![(16, 4)], mrd_ch: 2, wn: true, bias: true, mrd_phase: false }
    }
    pub fn check() -> Self {
        Self { periods: vec![2, 3, 5], mpd_ch: vec![8, 16, 16, 16, 16], mrd_res: vec![(64, 16), (128, 32)], mrd_ch: 8, wn: true, bias: true, mrd_phase: false }
    }
    pub fn by_name(n: &str) -> Option<Self> {
        let (base, rest) = n.split_once('-').unwrap_or((n, ""));
        let mut d = match base {
            "v12" => Self::v12(),
            "v12h" => Self::v12h(),
            "v1" | "v1f0" => Self::v1(),
            "tiny" | "tinyf0" => Self::tiny(),
            "check" | "checkf0" => Self::check(),
            _ => return None,
        };
        // suffixes: "-nown" (no weight norm), "-nobias"
        for s in rest.split('-').filter(|s| !s.is_empty()) {
            match s {
                "nown" => d.wn = false,
                "nobias" => d.bias = false,
                "nompd" => d.periods.clear(),
                "nomrd" => d.mrd_res.clear(),
                "lastmrd" => d.mrd_res = d.mrd_res.split_off(d.mrd_res.len() - 1),
                // §29.5: phase-difference channels on every MRD resolution
                "ph" => d.mrd_phase = true,
                // MRD width override, e.g. "c32"
                c if c.starts_with('c') && c[1..].parse::<usize>().is_ok() => d.mrd_ch = c[1..].parse().unwrap(),
                _ => return None,
            }
        }
        Some(d)
    }

    fn conv(&self, p: &mut Vec<ParamSpec>, name: String, sh: &[usize]) {
        // torch's default conv init has std 1/√(3·fan_in)
        // weights are stored taps-first, output channel last (`[taps…, C, O]`)
        let o = *sh.last().unwrap();
        let fan: usize = sh[..sh.len() - 1].iter().product();
        p.push(ParamSpec::normal(&format!("{name}.w"), sh, (1.0 / (3.0 * fan as f64)).sqrt()));
        if self.wn {
            p.push(ParamSpec::ones(&format!("{name}.g"), &[o]));
        }
        if self.bias {
            p.push(ParamSpec::zeros(&format!("{name}.b"), &[o]));
        }
    }

    /// Tensors per conv: `w` [, `g`] [, `b`].
    fn per_conv(&self) -> usize {
        1 + self.wn as usize + self.bias as usize
    }

    pub fn params(&self) -> Vec<ParamSpec> {
        let mut p = Vec::new();
        for &per in &self.periods {
            let mut c = 1;
            for (l, &o) in self.mpd_ch.iter().enumerate() {
                let sh: Vec<usize> = if l < MPD_STRIDED { vec![2, 3, c, o] } else { vec![5, c, o] };
                self.conv(&mut p, format!("mpd{per}.{l}"), &sh);
                c = o;
            }
            self.conv(&mut p, format!("mpd{per}.post"), &[3, c, 1]);
        }
        for &(nf, _) in &self.mrd_res {
            let c = self.mrd_ch;
            self.conv(&mut p, format!("mrd{nf}.0"), &[3, 9, self.mrd_in(), c]);
            for l in 1..=3 {
                self.conv(&mut p, format!("mrd{nf}.{l}"), &[3, 5, 2, c, c]);
            }
            self.conv(&mut p, format!("mrd{nf}.4"), &[3, 3, c, c]);
            self.conv(&mut p, format!("mrd{nf}.post"), &[3, 3, c, 1]);
        }
        p
    }

    /// MRD input channels: (re, im), plus the IF and GD pairs with `mrd_phase`
    pub fn mrd_in(&self) -> usize {
        if self.mrd_phase { 6 } else { 2 }
    }

    pub fn n_params(&self) -> usize {
        self.params().iter().map(|s| s.numel()).sum()
    }

    /// MPD rows for a segment of `n` samples at period `p`: `⌈n/p⌉` rounded
    /// up to a multiple of `3^4` so every stride-3 layer divides evenly
    /// (the tail is zero padding).
    fn rows(n: usize, p: usize) -> usize {
        let m = 3usize.pow(MPD_STRIDED as u32);
        n.div_ceil(p).div_ceil(m) * m
    }
}

fn lrelu(x: ExprId) -> ExprId {
    ExprId::select(x, x, x * lit(0.1))
}

/// `mask[n] · [lrelu](y[n, o] + b[o])` on a flat canvas `y [N, O]`, always
/// materialized (a reduction over a size-1 dim): the next layer reads it
/// guarded through a reshape, and tang 1e076822's `substitute_maps` inlines
/// a map read that way and then loses the reshape's source (gap 8,
/// `ir_gaps reshape-guard`). The mask zeroes the canvas padding.
fn bias_act(g: &mut TensorGraph, y: TensorId, b: Option<TensorId>, mask: TensorId, act: bool) -> TensorId {
    let sh = ir::dims(g, y);
    assert_eq!(sh.len(), 2);
    let mut args = vec![arg(y, pick(3, &[Some(0), Some(1)])), arg(mask, pick(3, &[Some(0)]))];
    if let Some(b) = b {
        args.push(arg(b, pick(3, &[Some(1)])));
    }
    g.comp(&sh, &[1], args, Some(Monoid::sum()), move |v| {
        let x = if v.len() > 2 { v[0] + v[2] } else { v[0] };
        vec![(if act { lrelu(x) } else { x }) * v[1]]
    })
    .unwrap()
}

/// Canvas validity mask `[S·L]`: 1 where `l < valid`, else 0. A graph
/// constant: built in-graph (ones through a guarded view) the masks are
/// zero-input comps, and the GPU codegen got some of them wrong once the
/// MPD's and MRD's were in one program (deterministic 1–8% errors in the
/// last MRD's first-layer grads on both CUDA and Metal; every pass is exact
/// in the interpreter, `disc_passes_preserve_semantics`).
fn canvas_mask(bt: &mut Built, segs: usize, len: usize, valid: usize) -> TensorId {
    let data = (0..segs * len).map(|i| if i % len < valid { 1.0 } else { 0.0 }).collect();
    bt.constant(&[segs * len], data)
}

/// 2-D canvas mask `[B·Tp·W]`: 1 on rows `< t` and columns `< f`.
fn canvas_mask2(bt: &mut Built, b: usize, tp: usize, w: usize, t: usize, f: usize) -> TensorId {
    let data = (0..b * tp * w).map(|i| if (i / w) % tp < t && i % w < f { 1.0 } else { 0.0 }).collect();
    bt.constant(&[b * tp * w], data)
}

/// `x` materialized (identity as a reduction over a size-1 dim; see `bias_act`).
fn materialize(g: &mut TensorGraph, x: TensorId) -> TensorId {
    ir::materialize(g, x)
}

/// One conv's parameters as graph tensors: the effective weight
/// (weight-normed when `g` is present) and the bias.
struct ConvP {
    w: TensorId,
    b: Option<TensorId>,
}

/// `w = g[o] · v / ‖v[o]‖`: the reparametrization of Salimans & Kingma
/// (2016) that HiFi-GAN and Vocos put on every D conv; its backward is
/// derived like any other comp.
fn weight_norm(g: &mut TensorGraph, v: TensorId, gain: TensorId) -> TensorId {
    let sh = ir::dims(g, v);
    let r = sh.len();
    // `v [taps…, C, O]`: the norm is over everything but the last dim
    let picks: Vec<Option<usize>> = (0..r).map(Some).collect();
    let mut np: Vec<Option<usize>> = (1..r).map(Some).collect();
    np.push(Some(0));
    let n2 = g.comp(&sh[r - 1..], &sh[..r - 1], vec![arg(v, pick(r, &np))], Some(Monoid::sum()), |x| vec![x[0] * x[0]]).unwrap();
    let w = g
        .comp(&sh, &[], vec![arg(v, pick(r, &picks)), arg(gain, pick(r, &[Some(r - 1)])), arg(n2, pick(r, &[Some(r - 1)]))], None, |x| vec![x[0] * x[1] / (x[2] + lit(1e-12)).sqrt()])
        .unwrap();
    // left a map: the identity-reduction `materialize` would put dL/dw (a
    // sum over the whole batch per weight) serially into its size-1 backward
    w
}

// Every D conv runs on a flat "canvas" `[N, C]`: segments (MPD: one per
// (batch, phase) column; MRD: one per spectrogram row) laid end to end, each
// with a zero tail at least as long as the conv's reach, so a tap is a
// constant offset `n + Δ` (a guarded unit view) and the output is `[N', O]`.
// Everything is then matmul-shaped: the tiled schedule takes (N', O) as its
// tile axes (input read varies along N' only, weight along O only: an
// implicit GEMM), and dL/dw reduces over the single dim N'.
// The V1.1 layouts reduced weight grads over (b, t, f) and ran split, at
// ~0.1–0.2 TFLOP/s. Strided convs are polyphase: the canvas is reshaped
// `[N/s, s, C]` and the phase joins the channels.

/// `y[n, o] = Σ_{taps, ph, c} h[n + Σ aₖ·tapₖ + off, ph, c] · w[taps…, ph, c, o]`
/// with `h [N, PH, C]` (stride 1: `[N, C]`), `w [taps…, (PH,) C, O]`.
fn flat_conv(g: &mut TensorGraph, h: TensorId, w: TensorId, coefs: &[i64], off: i64) -> TensorId {
    let hd = ir::dims(g, h);
    let wd = ir::dims(g, w);
    let nt = coefs.len();
    let strided = hd.len() == 3;
    let (n, o) = (hd[0], *wd.last().unwrap());
    // space (n, o, taps…, [ph,] c)
    let red: Vec<usize> = wd[..wd.len() - 1].to_vec();
    let rank = 2 + red.len();
    let mut terms: Vec<(usize, i64)> = vec![(0, 1)];
    terms.extend(coefs.iter().enumerate().map(|(k, &a)| (2 + k, a)));
    let mut forms = vec![form(rank, &terms, off)];
    if strided {
        forms.push(form(rank, &[(2 + nt, 1)], 0));
    }
    forms.push(form(rank, &[(rank - 1, 1)], 0));
    let mut wp: Vec<Option<usize>> = (2..rank).map(Some).collect();
    wp.push(Some(1));
    g.comp(&[n, o], &red, vec![arg(h, view(forms, true)), arg(w, pick(rank, &wp))], Some(Monoid::sum()), |v| vec![v[0] * v[1]]).unwrap()
}

/// `(re, im)` of the Hann STFT (no centering) as `[B, frames, bins, 2]`,
/// scaled by `1/√n_fft` (Vocos peak-normalizes the waveform to 0.8 first;
/// a fixed scale keeps D off a max-reduction backward). One contraction
/// against a stacked `[n_fft, bins, 2]` basis, so the first conv reads one
/// 2-channel canvas: reading `re` and `im` as two guarded args of one conv
/// gave wrong weight grads on CUDA (transpiled tiled kernel; Metal exact).
fn stft_ri(bt: &mut Built, x: TensorId, n_fft: usize, hop: usize) -> TensorId {
    let (c, sn) = crate::dsp::stft_bases(n_fft);
    let bins = n_fft / 2 + 1;
    let basis = bt.constant(&[n_fft, bins, 2], c.iter().zip(&sn).flat_map(|(&a, &b)| [a, b]).collect());
    let g = &mut bt.g;
    let [b, l] = ir::dims(g, x)[..] else { panic!() };
    let (q, j) = (l / hop, n_fft / hop);
    let frames = q - j + 1;
    let x3 = g.reshape(x, &[b, q, hop]).unwrap();
    let uv = view(vec![form(4, &[(0, 1)], 0), form(4, &[(1, 1), (2, 1)], 0), form(4, &[(3, 1)], 0)], true);
    let u = g.comp(&[b, frames, j, hop], &[], vec![arg(x3, uv)], None, |v| vec![v[0]]).unwrap();
    let u = g.reshape(u, &[b, frames, n_fft]).unwrap();
    let s = 1.0 / (n_fft as f64).sqrt();
    // space (b, t, k, c, n)
    let y = g
        .comp(&[b, frames, bins, 2], &[n_fft], vec![arg(u, pick(5, &[Some(0), Some(1), Some(4)])), arg(basis, pick(5, &[Some(4), Some(2), Some(3)]))], Some(Monoid::sum()), move |v| vec![v[0] * v[1] * lit(s)])
        .unwrap();
    materialize(g, y)
}

/// §29.5 phase-aware MRD input: `[B, T, bins, 6]` = (re, im) of `spec`,
/// then `u_t·conj(u_{t−1})` (instantaneous frequency: the frame-to-frame
/// phase advance) and `u_k·conj(u_{k−1})` (group delay: the bin-to-bin phase
/// step), with `u = S·(|S|² + ε)^{−1/4}` so each product has magnitude
/// ≈ √(|S_a||S_b|), the scale of `S` itself. tang has no atan2, and no angle
/// is needed: the phase difference rides on a complex product, so a silent
/// bin contributes ≈ 0 instead of a random angle. Built as one map with
/// constant channel selectors (tang has no concat, gap 16), so the first
/// conv reads one canvas (two guarded args into one CUDA conv: gap 12).
fn phase_feats(bt: &mut Built, spec: TensorId) -> TensorId {
    let [b, t, k, _] = ir::dims(&bt.g, spec)[..] else { panic!() };
    // eps on |S|²; spec is scaled by 1/√n_fft, speech bins sit ~1e-3…1
    let eps = 1e-8;
    let g = &mut bt.g;
    let ch = |c: i64| view(vec![form(4, &[(0, 1)], 0), form(4, &[(1, 1)], 0), form(4, &[(2, 1)], 0), form(4, &[], c)], false);
    let all = pick(4, &[Some(0), Some(1), Some(2), Some(3)]);
    let u = g
        .comp(&[b, t, k, 2], &[], vec![arg(spec, all), arg(spec, ch(0)), arg(spec, ch(1))], None, move |v| {
            vec![v[0] * (lit(-0.25) * (v[1] * v[1] + v[2] * v[2] + lit(eps)).ln()).exp()]
        })
        .unwrap();
    let u = materialize(g, u);
    // reads of u at channel c, shifted by (dt, dk) (guarded: 0 off the edge)
    let at = |c: i64, dt: i64, dk: i64| view(vec![form(4, &[(0, 1)], 0), form(4, &[(1, 1)], dt), form(4, &[(2, 1)], dk), form(4, &[], c)], true);
    let sel: Vec<f64> = (0..6).flat_map(|c| (0..6).map(move |j| if c == j { 1.0 } else { 0.0 })).collect();
    let sel = bt.constant(&[6, 6], sel);
    let g = &mut bt.g;
    let sv = |j: i64| view(vec![form(4, &[(3, 1)], 0), form(4, &[], j)], false);
    let mut args = vec![arg(spec, ch(0)), arg(spec, ch(1)), arg(u, at(0, 0, 0)), arg(u, at(1, 0, 0)), arg(u, at(0, -1, 0)), arg(u, at(1, -1, 0)), arg(u, at(0, 0, -1)), arg(u, at(1, 0, -1))];
    for j in 0..6 {
        args.push(arg(sel, sv(j)));
    }
    let y = g
        .comp(&[b, t, k, 6], &[], args, None, |v| {
            let (re, im, ar, ai, tr, ti, fr, fi) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            let f = [re, im, ar * tr + ai * ti, ai * tr - ar * ti, ar * fr + ai * fi, ai * fr - ar * fi];
            let mut o = f[0] * v[8];
            for j in 1..6 {
                o = o + f[j] * v[8 + j];
            }
            vec![o]
        })
        .unwrap();
    materialize(g, y)
}

/// One discriminator's output on its canvas.
pub struct DOut {
    /// feature maps used for feature matching (every layer after the first,
    /// as in HiFi-GAN/Vocos), `[S, L, C]`, zero on the canvas padding, each
    /// with `canvas size / valid size` (mean over the canvas × this = mean
    /// over valid positions); the logits come last
    pub feats: Vec<(TensorId, f64)>,
    /// logits `[S, L, 1]` (0 on padding), the validity mask (same shape), its scale
    pub logits: TensorId,
    pub mask: TensorId,
    pub scale: f64,
    /// this sub-discriminator's own materialized copy of the input
    pub input: TensorId,
}

/// MPD canvas tail (rows): ≥ every layer's backward reach at its depth
/// (3 rows per stride-3 layer at depth < 4, 2 rows for the k5 conv at
/// depth 4: 2·3⁴).
const MPD_TAIL: usize = 162;
/// MRD canvas tail (columns): ≥ 2 pairs at stride-2 depth 2 → 16 at depth 0.
const MRD_TAIL: usize = 16;

/// All sub-discriminators on audio `x [B, L]`: MPD (one per period) then MRD
/// (one per resolution). `p` are the D params in `DConfig::params` order.
pub fn discriminate(bt: &mut Built, d: &DConfig, _c: &model::Consts, p: &[TensorId], x: TensorId) -> Vec<DOut> {
    let [b, l] = ir::dims(&bt.g, x)[..] else { panic!() };
    assert_eq!(p.len(), d.params().len());
    let convs: Vec<ConvP> = {
        let k = d.per_conv();
        p.chunks(k)
            .map(|ch| {
                let w = if d.wn { weight_norm(&mut bt.g, ch[0], ch[1]) } else { ch[0] };
                ConvP { w, b: if d.bias { Some(ch[k - 1]) } else { None } }
            })
            .collect()
    };
    let mut it = convs.iter();
    let mut next = || it.next().unwrap();
    let mut outs = Vec::new();
    let seg3 = |g: &mut TensorGraph, h: TensorId, segs: usize, len: usize| {
        let ch = ir::dims(g, h)[1];
        g.reshape(h, &[segs, len, ch]).unwrap()
    };
    for &per in &d.periods {
        let x = materialize(&mut bt.g, x);
        let rows = DConfig::rows(l, per);
        let seg = rows + MPD_TAIL;
        let segs = b * per;
        // zero-pad to rows·per (guarded unit view), fold into [B, rows, per],
        // then lay each (b, phase) column out as a segment of `seg` rows
        let xp = bt.g.comp(&[b, rows * per], &[], vec![arg(x, view(vec![form(2, &[(0, 1)], 0), form(2, &[(1, 1)], 0)], true))], None, |v| vec![v[0]]).unwrap();
        let xp = materialize(&mut bt.g, xp);
        let x3 = bt.g.reshape(xp, &[b, rows, per]).unwrap();
        // canvas[b, p, r] = x3[b, r, p]  (r ≥ rows is past x3: zero)
        let cv = bt.g.comp(&[b, per, seg], &[], vec![arg(x3, view(vec![form(3, &[(0, 1)], 0), form(3, &[(2, 1)], 0), form(3, &[(1, 1)], 0)], true))], None, |v| vec![v[0]]).unwrap();
        let cv = materialize(&mut bt.g, cv);
        let mut h = bt.g.reshape(cv, &[segs * seg, 1]).unwrap();
        let (mut len, mut valid) = (seg, rows);
        let mut feats = Vec::new();
        let mut mask = None;
        for li in 0..d.mpd_ch.len() {
            let cp = next();
            let y = if li < MPD_STRIDED {
                let [n, ch] = ir::dims(&bt.g, h)[..] else { panic!() };
                let h3 = bt.g.reshape(h, &[n / 3, 3, ch]).unwrap();
                len /= 3;
                valid /= 3;
                // output row t reads rows 3(t + q − 1) + r
                flat_conv(&mut bt.g, h3, cp.w, &[1], -1)
            } else {
                flat_conv(&mut bt.g, h, cp.w, &[1], -2)
            };
            let m = canvas_mask(bt, segs, len, valid);
            mask = Some(m);
            h = bias_act(&mut bt.g, y, cp.b, m, true);
            if li > 0 {
                let f = seg3(&mut bt.g, h, segs, len);
                feats.push((f, len as f64 / valid as f64));
            }
        }
        let mask = mask.unwrap();
        let cp = next();
        let y = flat_conv(&mut bt.g, h, cp.w, &[1], -1);
        let lo = bias_act(&mut bt.g, y, cp.b, mask, false);
        let logits = seg3(&mut bt.g, lo, segs, len);
        let scale = len as f64 / valid as f64;
        feats.push((logits, scale));
        let mask = bt.g.reshape(mask, &[segs, len, 1]).unwrap();
        outs.push(DOut { feats, logits, mask, scale, input: x });
    }
    for &(nf, hop) in &d.mrd_res {
        let x = materialize(&mut bt.g, x);
        let spec = stft_ri(bt, x, nf, hop);
        let spec = if d.mrd_phase { phase_feats(bt, spec) } else { spec };
        let cin = d.mrd_in();
        let t = ir::dims(&bt.g, spec)[1];
        let (tp, f0) = (t + 1, nf / 2);
        let mut wd = f0 + MRD_TAIL;
        let segs = b * tp;
        // canvas [B, T+1, W, 2]: spectrum at (t < T, f ≤ F0), zero elsewhere
        let g = &mut bt.g;
        let cv = g.comp(&[b, tp, wd, cin], &[], vec![arg(spec, view(vec![form(4, &[(0, 1)], 0), form(4, &[(1, 1)], 0), form(4, &[(2, 1)], 0), form(4, &[(3, 1)], 0)], true))], None, |v| vec![v[0]]).unwrap();
        let cv = materialize(g, cv);
        let cv = g.reshape(cv, &[b * tp * wd, cin]).unwrap();
        let mut fv = f0;
        let mut feats = Vec::new();
        let cp = next();
        let y = flat_conv(&mut bt.g, cv, cp.w, &[wd as i64, 1], -(wd as i64) - 4);
        let mask = canvas_mask2(bt, b, tp, wd, t, fv);
        let mut h = bias_act(&mut bt.g, y, cp.b, mask, true);
        for _ in 1..=3 {
            let cp = next();
            let [n, ch] = ir::dims(&bt.g, h)[..] else { panic!() };
            let h2 = bt.g.reshape(h, &[n / 2, 2, ch]).unwrap();
            wd /= 2;
            fv /= 2;
            // tap (i, q, r) reads [t + i − 1, 2(f + q − 2) + r]
            let y = flat_conv(&mut bt.g, h2, cp.w, &[wd as i64, 1], -(wd as i64) - 2);
            let mask = canvas_mask2(bt, b, tp, wd, t, fv);
            h = bias_act(&mut bt.g, y, cp.b, mask, true);
            feats.push(seg3(&mut bt.g, h, segs, wd));
        }
        let mask = canvas_mask2(bt, b, tp, wd, t, fv);
        let cp = next();
        let y = flat_conv(&mut bt.g, h, cp.w, &[wd as i64, 1], -(wd as i64) - 1);
        h = bias_act(&mut bt.g, y, cp.b, mask, true);
        feats.push(seg3(&mut bt.g, h, segs, wd));
        let cp = next();
        let y = flat_conv(&mut bt.g, h, cp.w, &[wd as i64, 1], -(wd as i64) - 1);
        let lo = bias_act(&mut bt.g, y, cp.b, mask, false);
        let logits = seg3(&mut bt.g, lo, segs, wd);
        feats.push(logits);
        // the valid fraction of the last 2-D canvas (the stride-2 layers'
        // canvases have the same fraction up to rounding of fv)
        let scale = (tp * wd) as f64 / (t * fv) as f64;
        let feats = feats.into_iter().map(|f| (f, scale)).collect();
        let mask = bt.g.reshape(mask, &[segs, wd, 1]).unwrap();
        outs.push(DOut { feats, logits, mask, scale, input: x });
    }
    assert!(it.next().is_none());
    outs
}

/// `mean(f(a[, b]))` over all elements as `[1]`, in two stages: the last
/// two dims first (many output points), then the rest. A one-stage reduction
/// to a single point runs as one threadgroup in tang 1e076822's schedule,
/// and inlines its producer: ~0.3–0.4 s each at V1 size.
fn mean2(g: &mut TensorGraph, a: TensorId, b: Option<TensorId>, f: impl Fn(&[ExprId]) -> ExprId) -> TensorId {
    let sh = ir::dims(g, a);
    let r = sh.len();
    let n = sh.iter().product::<usize>() as f64;
    let (outer, inner) = sh.split_at(r - 2);
    // space (outer…, inner…)
    let picks: Vec<Option<usize>> = (0..r).map(Some).collect();
    let mut args = vec![arg(a, pick(r, &picks))];
    if let Some(b) = b {
        args.push(arg(b, pick(r, &picks)));
    }
    let s1 = g.comp(outer, inner, args, Some(Monoid::sum()), |v| vec![f(v)]).unwrap();
    ir::sum_all2(g, s1, None, |v| v[0] * lit(1.0 / n))
}

/// `mean over valid positions of relu(1 + s·logit)` (s = −1: real / G, +1: fake).
fn hinge(g: &mut TensorGraph, d: &DOut, s: f64) -> TensorId {
    let sc = d.scale;
    mean2(g, d.logits, Some(d.mask), move |v| v[1] * relu(lit(1.0) + v[0] * lit(s)) * lit(sc))
}

fn relu(x: ExprId) -> ExprId {
    ExprId::select(x, x, lit(0.0))
}

/// `Σ wᵢ·xᵢ` of `[1]` scalars, in groups of 8, each group sum materialized
/// (a size-1 reduction): maps of scalars otherwise inline back into one
/// kernel binding more than Metal's 31 buffers. `keep` is unused now but
/// kept for callers that want the partials as outputs.
pub fn weighted_sum(g: &mut TensorGraph, xs: &[(TensorId, f64)], keep: &mut Vec<TensorId>) -> TensorId {
    let _ = keep;
    if xs.is_empty() {
        let z = g.comp(&[1], &[], vec![], None, |_| vec![lit(0.0)]).unwrap();
        return materialize(g, z);
    }
    let mut parts: Vec<TensorId> = xs
        .chunks(8)
        .map(|c| {
            let ws: Vec<f64> = c.iter().map(|x| x.1).collect();
            // a private reader per scalar: `merge_siblings` would otherwise
            // tuple up to 4 sibling reductions (each reading 2+ feature maps)
            // into one kernel, again past 31 buffers
            let ids: Vec<TensorId> = c.iter().map(|x| materialize(g, x.0)).collect();
            let m = g.map(&ids, move |v| v.iter().zip(&ws).fold(ExprId::ZERO, |a, (&x, &w)| a + x * lit(w))).unwrap();
            materialize(g, m)
        })
        .collect();
    while parts.len() > 1 {
        parts = parts
            .chunks(8)
            .map(|c| {
                let m = g.map(c, |v| v.iter().fold(ExprId::ZERO, |a, &x| a + x)).unwrap();
                materialize(g, m)
            })
            .collect();
    }
    parts[0]
}

/// Loss weights and optimizer betas. Defaults are Vocos' (configs/vocos.yaml
/// + experiment.py, gemelo-ai/vocos v0.1.0): mel 45, MRD adv and FM × 0.1,
/// MPD adv and FM × 1, each averaged over its sub-discriminators, hinge
/// losses, AdamW β (0.8, 0.9). HiFi-GAN instead uses LSGAN, FM × 2, β2 0.99.
#[derive(Clone, Debug)]
pub struct Weights {
    pub mel: f64,
    pub mrstft: f64,
    pub adv: f64,
    pub fm: f64,
    pub mrd: f64,
    pub beta1: f64,
    pub beta2: f64,
    /// APNet phase losses (IP + GD + PTD, §29.3); 0 = off (graph unchanged)
    pub phase: f64,
    /// per-term multipliers on `phase` for [IP, GD, PTD]
    pub phase_terms: [f64; 3],
    /// |S|-weighted phase means (instead of APNet's plain means)
    pub phase_mag: bool,
}

impl Default for Weights {
    fn default() -> Self {
        Self { mel: 45.0, mrstft: 1.0, adv: 1.0, fm: 1.0, mrd: 0.1, beta1: 0.8, beta2: 0.9, phase: 0.0, phase_terms: [1.0; 3], phase_mag: false }
    }
}

fn d_inputs(b: &mut Built, d: &DConfig) -> Vec<TensorId> {
    b.dspecs = d.params();
    (0..b.dspecs.len()).map(|i| b.input(&b.dspecs[i].shape.clone(), Role::DParam(i))).collect()
}

/// Generator step: V1's losses plus adversarial + feature matching against
/// a fixed D (its params are inputs), backward w.r.t. G only, AdamW.
/// `hyper[5]` scales the adversarial + FM terms (the ramp; 0 = V1's loss).
pub struct GenGraph {
    pub lg: LossGraph,
    pub dparams: Vec<TensorId>,
    pub hyper: TensorId,
    pub m: Vec<TensorId>,
    pub v: Vec<TensorId>,
    /// `[g_total, mrstft, mel_l1, |g|, adv_mpd, adv_mrd, fm_mpd, fm_mrd]`,
    /// then p′, m′, v′ per G param, then `y`, then partials / residual grads.
    pub outputs: Vec<TensorId>,
    pub n_head: usize,
    /// index of `[L_IP, L_GD, L_PTD]` in `outputs` (when `Weights::phase > 0`)
    pub phase_out: Option<usize>,
}

pub const GEN_HEAD: usize = 8;

/// The G-step forward: loss graph, D inputs, and the weighted loss terms.
pub struct GenTerms {
    pub lg: LossGraph,
    pub dparams: Vec<TensorId>,
    pub m: Vec<TensorId>,
    pub v: Vec<TensorId>,
    pub hyper: TensorId,
    pub fake: Vec<DOut>,
    /// (name, scalar, weight); the first two are reconstruction, the rest adversarial
    pub terms: Vec<(&'static str, TensorId, f64)>,
    pub keep: Vec<TensorId>,
}

pub fn gen_terms(cfg: &Config, d: &DConfig, w: &Weights, batch: usize, frames: usize) -> GenTerms {
    gen_terms_ext(cfg, d, w, batch, frames, false)
}

/// `gen_terms` on `model::loss_graph_ext` (external input mels, §16).
pub fn gen_terms_ext(cfg: &Config, d: &DConfig, w: &Weights, batch: usize, frames: usize, ext: bool) -> GenTerms {
    // the MRD shares the loss graph's analysis bases
    for &r in &d.mrd_res {
        assert!(cfg.loss_res.iter().any(|x| x.0 == r.0), "MRD n_fft {} must be a loss resolution", r.0);
    }
    let mut lg = model::loss_graph_phase(cfg, batch, frames, ext, w.phase > 0.0, w.phase_mag);
    let dparams = d_inputs(&mut lg.b, d);
    let specs = cfg.params();
    let m: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| lg.b.input(&s.shape, Role::AdamM(i))).collect();
    let v: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| lg.b.input(&s.shape, Role::AdamV(i))).collect();
    let hyper = lg.b.input(&[6], Role::Hyper);
    let real = discriminate(&mut lg.b, d, &lg.consts, &dparams, lg.audio);
    let fake = discriminate(&mut lg.b, d, &lg.consts, &dparams, lg.y);
    let g = &mut lg.b.g;
    let (np_, nr) = (d.periods.len() as f64, d.mrd_res.len() as f64);
    let mut keep = Vec::new();
    let (mut adv_mpd, mut adv_mrd, mut fm_mpd, mut fm_mrd) = (vec![], vec![], vec![], vec![]);
    for (i, (r, f)) in real.iter().zip(&fake).enumerate() {
        let adv = hinge(g, f, -1.0);
        let fms: Vec<(TensorId, f64)> = r
            .feats
            .iter()
            .zip(&f.feats)
            .map(|(&(a, sc), &(b, _))| (mean2(g, a, Some(b), |v| ir::abs(v[0] - v[1])), sc))
            .collect();
        let fm = weighted_sum(g, &fms, &mut keep);
        // Vocos divides each group by its number of sub-discriminators
        if i < d.periods.len() {
            adv_mpd.push((adv, 1.0 / np_));
            fm_mpd.push((fm, 1.0 / np_));
        } else {
            adv_mrd.push((adv, 1.0 / nr));
            fm_mrd.push((fm, 1.0 / nr));
        }
    }
    let adv_mpd = weighted_sum(g, &adv_mpd, &mut keep);
    let adv_mrd = weighted_sum(g, &adv_mrd, &mut keep);
    let fm_mpd = weighted_sum(g, &fm_mpd, &mut keep);
    let fm_mrd = weighted_sum(g, &fm_mrd, &mut keep);
    let l = &lg.losses;
    let terms = vec![("mrstft", l.mrstft, w.mrstft), ("mel", l.mel_l1, w.mel), ("adv_mpd", adv_mpd, w.adv), ("adv_mrd", adv_mrd, w.adv * w.mrd), ("fm_mpd", fm_mpd, w.fm), ("fm_mrd", fm_mrd, w.fm * w.mrd)];
    GenTerms { lg, dparams, m, v, hyper, fake, terms, keep }
}

pub fn gen_graph(cfg: &Config, d: &DConfig, w: &Weights, batch: usize, frames: usize) -> GenGraph {
    gen_graph_ext(cfg, d, w, batch, frames, false)
}

/// `gen_graph` with external input mels (`model::loss_graph_ext`).
pub fn gen_graph_ext(cfg: &Config, d: &DConfig, w: &Weights, batch: usize, frames: usize, ext: bool) -> GenGraph {
    let GenTerms { mut lg, dparams, m, v, hyper, fake, terms, mut keep } = gen_terms_ext(cfg, d, w, batch, frames, ext);
    let specs = cfg.params();
    let g = &mut lg.b.g;
    let tw: Vec<(TensorId, f64)> = terms.iter().map(|t| (t.1, t.2)).collect();
    let mut rw = tw[..2].to_vec();
    if let Some(ph) = lg.losses.phase {
        rw.extend(ph.iter().zip(w.phase_terms).filter(|(_, m)| *m != 0.0).map(|(&t, m)| (t, w.phase * m)));
    }
    let recon = weighted_sum(g, &rw, &mut keep);
    let adv = weighted_sum(g, &tw[2..], &mut keep);
    // total = recon + hyper[5]·adv
    let total = g
        .comp(&[1], &[], vec![arg(recon, pick(1, &[Some(0)])), arg(adv, pick(1, &[Some(0)])), arg(hyper, view(vec![form(1, &[], 5)], false))], None, |v| vec![v[0] + v[1] * v[2]])
        .unwrap();
    let total = materialize(g, total);
    let [adv_mpd, adv_mrd, fm_mpd, fm_mrd] = [terms[2].1, terms[3].1, terms[4].1, terms[5].1];
    let l = &lg.losses;
    let np = specs.len();
    // Gradients w.r.t. each sub-discriminator's input copy are kept as
    // outputs: otherwise dL/dy (8 discriminators + the losses) fuses into
    // one kernel reading more than Metal's 31 buffers.
    let mut wrt = lg.params.clone();
    wrt.extend(lg.resid.iter().copied());
    wrt.extend(fake.iter().map(|f| f.input));
    let mut grads = g.backward(total, &wrt).unwrap();
    let resid_grads = grads.split_off(np);
    let ad = model::adamw_betas(g, &specs, &lg.params, &grads, &m, &v, hyper, w.beta1, w.beta2);
    let mut outputs = vec![total, l.mrstft, l.mel_l1, ad.grad_norm, adv_mpd, adv_mrd, fm_mpd, fm_mrd];
    outputs.extend(&ad.new_p);
    outputs.extend(&ad.new_m);
    outputs.extend(&ad.new_v);
    outputs.push(lg.y);
    outputs.extend(ad.partials);
    outputs.extend(keep);
    outputs.extend(resid_grads);
    // phase losses last, so every index above is unchanged
    let phase_out = lg.losses.phase.map(|ph| {
        let at = outputs.len();
        outputs.extend(ph);
        at
    });
    GenGraph { lg, dparams, hyper, m, v, outputs, n_head: GEN_HEAD, phase_out }
}

/// Discriminator step: hinge loss on real `audio` vs generated `fake`
/// (data: the G step's `y`), backward w.r.t. D, AdamW.
pub struct DiscGraph {
    pub b: Built,
    pub dparams: Vec<TensorId>,
    pub audio: TensorId,
    pub fake: TensorId,
    pub hyper: TensorId,
    pub dm: Vec<TensorId>,
    pub dv: Vec<TensorId>,
    /// `[d_total, d_mpd, d_mrd, |g_D|]`, then p′, m′, v′ per D param, then partials
    pub outputs: Vec<TensorId>,
}

pub const DISC_HEAD: usize = 4;

pub fn disc_graph(cfg: &Config, d: &DConfig, w: &Weights, batch: usize, frames: usize) -> DiscGraph {
    let mut b = Built::new_pub(cfg);
    let analysis: Vec<usize> = d.mrd_res.iter().map(|r| r.0).collect();
    let (c, _) = model::consts(&mut b, &analysis, frames);
    let n = frames * cfg.hop;
    let dparams = d_inputs(&mut b, d);
    let audio = b.input(&[batch, n], Role::Data("audio"));
    let fake = b.input(&[batch, n], Role::Data("fake"));
    let ds = b.dspecs.clone();
    let dm: Vec<TensorId> = ds.iter().enumerate().map(|(i, s)| b.input(&s.shape, Role::DAdamM(i))).collect();
    let dv: Vec<TensorId> = ds.iter().enumerate().map(|(i, s)| b.input(&s.shape, Role::DAdamV(i))).collect();
    let hyper = b.input(&[6], Role::Hyper);
    let real = discriminate(&mut b, d, &c, &dparams, audio);
    let gen = discriminate(&mut b, d, &c, &dparams, fake);
    let g = &mut b.g;
    let (np_, nr) = (d.periods.len() as f64, d.mrd_res.len() as f64);
    let mut keep = Vec::new();
    let (mut lmpd, mut lmrd) = (vec![], vec![]);
    for (i, (r, f)) in real.iter().zip(&gen).enumerate() {
        let lr = hinge(g, r, -1.0);
        let lf = hinge(g, f, 1.0);
        let (dst, k) = if i < d.periods.len() { (&mut lmpd, np_) } else { (&mut lmrd, nr) };
        dst.push((lr, 1.0 / k));
        dst.push((lf, 1.0 / k));
    }
    let lmpd = weighted_sum(g, &lmpd, &mut keep);
    let lmrd = weighted_sum(g, &lmrd, &mut keep);
    let total = weighted_sum(g, &[(lmpd, 1.0), (lmrd, w.mrd)], &mut keep);
    let grads = g.backward(total, &dparams).unwrap();
    // Per-channel grads (biases, weight-norm gains) are kept as outputs so
    // they are materialized: otherwise tang 1e076822 inlines each one (a sum
    // over the whole feature map per channel) into the 1-point grad-norm
    // kernel as a serial loop, 0.1–0.4 s each (gap 10, `ir_gaps point-reduce`).
    keep.extend(grads.iter().zip(&ds).filter(|(_, s)| s.shape.len() == 1).map(|(&t, _)| t));
    let ad = model::adamw_betas(g, &ds, &dparams, &grads, &dm, &dv, hyper, w.beta1, w.beta2);
    let mut outputs = vec![total, lmpd, lmrd, ad.grad_norm];
    outputs.extend(&ad.new_p);
    outputs.extend(&ad.new_m);
    outputs.extend(&ad.new_v);
    outputs.extend(ad.partials);
    outputs.extend(keep);
    DiscGraph { b, dparams, audio, fake, hyper, dm, dv, outputs }
}

/// Fresh D params (deterministic in `seed`).
/// Weight-norm gains start at `‖v[o]‖` (torch's `weight_norm` init), so
/// the effective weight starts equal to `v`.
pub fn init_dparams(d: &DConfig, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = model::Rng::new(seed);
    let specs = d.params();
    let mut p: Vec<Vec<f32>> = specs.iter().map(|s| s.init_data(&mut rng)).collect();
    if d.wn {
        for i in 0..specs.len() {
            if specs[i].name.ends_with(".g") {
                // v is [taps…, C, O]: output channel fastest
                let v = &p[i - 1];
                let o = specs[i].shape[0];
                p[i] = (0..o).map(|k| v.iter().skip(k).step_by(o).map(|x| x * x).sum::<f32>().sqrt()).collect();
            }
        }
    }
    p
}

/// Per-term gradient norms of the G loss (diagnostics): for each weighted
/// term, ‖w·∂term/∂θ_G‖ and the largest |entry| with the param it sits in.
pub struct Diag {
    pub lg: LossGraph,
    pub dparams: Vec<TensorId>,
    pub names: Vec<&'static str>,
    /// per term: [norm] then per-param max |g|
    pub outputs: Vec<TensorId>,
}

pub fn diag_graph(cfg: &Config, d: &DConfig, w: &Weights, batch: usize, frames: usize) -> Diag {
    let GenTerms { mut lg, dparams, terms, .. } = gen_terms(cfg, d, w, batch, frames);
    let g = &mut lg.b.g;
    let mut outputs = Vec::new();
    let mut names = Vec::new();
    let params = lg.params.clone();
    for (name, t, wt) in &terms {
        let mut wrt = params.clone();
        wrt.extend(lg.resid.iter().copied());
        let grads = g.backward(*t, &wrt).unwrap();
        for gr in &grads[..params.len()] {
            let wt = *wt;
            let mx = ir::sum_all2(g, *gr, None, move |v| v[0] * v[0] * lit(wt * wt));
            outputs.push(mx);
        }
        names.push(*name);
    }
    Diag { lg, dparams, names, outputs }
}
