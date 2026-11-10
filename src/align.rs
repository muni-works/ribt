//! V2 aligner (the ribt paper): phoneme tokens ↔ mel frames →
//! per-phoneme durations. RAD-TTS / "One TTS Alignment to Rule Them All"
//! (arXiv 2108.10447): a conv text encoder and a conv mel encoder, soft
//! alignment `softmax_n(−temp·‖q_t − k_n‖² + log prior)`, a forward-sum loss
//! over monotonic paths, and later a binarization term that pulls the soft
//! alignment onto its Viterbi (MAS) path. Durations are MAS run lengths.
//!
//! **The scan cut.** The IR has no scan op (IR gap 16), and the forward-sum
//! is a recursion over frames. So the graph stops at the distance logits
//! `z[b,n,t]`; the host (f64, Rust) does the log-softmax with the prior,
//! forward–backward, MAS, both losses, and `G = ∂L/∂z`. `G` goes back into
//! the train program as data, whose objective is the surrogate `Σ z ⊙ G`:
//! its derived backward is exactly `∂L/∂θ` (G is constant w.r.t. θ).
//! Two programs per step: `fwd` (z only) and `train` (z, surrogate,
//! backward, AdamW); after each step the new params are copied into `fwd`.
//!
//! **The scan path.** tang now has semiring scans, so [`scan_loss`] writes
//! the same losses in the graph: the log-softmax, the forward-sum as a
//! banded log-semiring scan over frames (bands stay / advance), MAS as the
//! same scan in max-plus (its gradient w.r.t. the log attention is the
//! one-hot path), and both losses, with the backward derived.
//! [`scan_train_graph`] is then one program per step. The host path stays
//! the reference and the training default.

use tang::Scalar;
use tang_compile::tensor::{Monoid, Semiring, TensorGraph, TensorId, View};
use tang_expr::ExprId;

use crate::ir::{self, arg, form, lit, pick, view};
use crate::model::{self, Built, Config, ParamSpec, Role};

/// Kokoro's token count (misaki phoneme chars + punctuation, pad id 0).
pub const KOKORO_VOCAB: usize = 178;

#[derive(Clone, Debug)]
pub struct AConfig {
    pub vocab: usize,
    pub n_mels: usize,
    /// text embedding width, text conv hidden width
    pub emb: usize,
    pub text_h: usize,
    /// mel encoder widths (k3 conv → 1×1 → 1×1)
    pub mel_h1: usize,
    pub mel_h2: usize,
    /// key/query width
    pub att: usize,
    /// RAD-TTS temperature on the squared distance
    pub temp: f64,
    /// fixed input normalization of the log-mel
    pub mel_mu: f64,
    pub mel_sd: f64,
}

impl AConfig {
    /// RAD-TTS widths (n_mel→2n_mel→n_mel→80, text →2·text→80), our 256-d
    /// embedding and 100 mels at the vocoder's frame config.
    pub fn base() -> Self {
        Self {
            vocab: KOKORO_VOCAB,
            n_mels: 100,
            emb: 256,
            text_h: 512,
            mel_h1: 200,
            mel_h2: 100,
            att: 80,
            temp: 0.0005,
            mel_mu: -5.0,
            mel_sd: 2.5,
        }
    }

    /// For the f64 interpreter.
    pub fn tiny() -> Self {
        Self { vocab: 7, n_mels: 5, emb: 4, text_h: 6, mel_h1: 6, mel_h2: 5, att: 3, temp: 0.05, mel_mu: 0.0, mel_sd: 1.0 }
    }

    /// Big enough for tiled GPU paths, small enough for the interpreter.
    pub fn check() -> Self {
        Self { vocab: 40, n_mels: 20, emb: 32, text_h: 48, mel_h1: 40, mel_h2: 24, att: 16, temp: 0.01, mel_mu: -5.0, mel_sd: 2.5 }
    }

    pub fn by_name(s: &str) -> Option<Self> {
        match s {
            "base" => Some(Self::base()),
            "tiny" => Some(Self::tiny()),
            "check" => Some(Self::check()),
            _ => None,
        }
    }

    pub fn params(&self) -> Vec<ParamSpec> {
        let he = |fan: usize| (2.0 / fan as f64).sqrt();
        let lin = |fan: usize| (1.0 / fan as f64).sqrt();
        vec![
            ParamSpec::normal("text.emb", &[self.emb, self.vocab], 0.3),
            ParamSpec::normal("text.c1.w", &[self.text_h, self.emb, 3], he(3 * self.emb)),
            ParamSpec::zeros("text.c1.b", &[self.text_h]),
            ParamSpec::normal("text.c2.w", &[self.att, self.text_h], lin(self.text_h)),
            ParamSpec::zeros("text.c2.b", &[self.att]),
            ParamSpec::normal("mel.c1.w", &[self.mel_h1, self.n_mels, 3], he(3 * self.n_mels)),
            ParamSpec::zeros("mel.c1.b", &[self.mel_h1]),
            ParamSpec::normal("mel.c2.w", &[self.mel_h2, self.mel_h1], he(self.mel_h1)),
            ParamSpec::zeros("mel.c2.b", &[self.mel_h2]),
            ParamSpec::normal("mel.c3.w", &[self.att, self.mel_h2], lin(self.mel_h2)),
            ParamSpec::zeros("mel.c3.b", &[self.att]),
        ]
    }

    pub fn n_params(&self) -> usize {
        self.params().iter().map(|p| p.numel()).sum()
    }
}

fn relu(x: ExprId) -> ExprId {
    ExprId::select(x, x, ExprId::ZERO)
}

/// Forward graph: one-hot text `[B, V, N]` + log-mel `[B, n_mels, T]` →
/// distance logits `z[B, N, T] = −temp·Σ_c (q[b,c,t] − k[b,c,n])²`.
pub struct AlignGraph {
    pub b: Built,
    pub ac: AConfig,
    pub params: Vec<TensorId>,
    pub onehot: TensorId,
    pub mel: TensorId,
    pub z: TensorId,
}

pub fn align_graph(ac: &AConfig, batch: usize, n: usize, t: usize) -> AlignGraph {
    // `Built` carries a vocoder config only for its input bookkeeping
    let mut b = Built::new_pub(&Config::tiny());
    let specs = ac.params();
    let params: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| b.input(&s.shape, Role::Param(i))).collect();
    let onehot = b.input(&[batch, ac.vocab, n], Role::Data("onehot"));
    let mel = b.input(&[batch, ac.n_mels, t], Role::Data("mel"));
    let g = &mut b.g;
    let p = |i: usize| params[i];
    // text keys
    let h = ir::linear(g, onehot, p(0));
    let h = ir::conv1d(g, h, p(1), p(2));
    let h = g.map(&[h], |v| relu(v[0])).unwrap();
    let h = ir::linear(g, h, p(3));
    let k = ir::add_channel(g, h, p(4));
    // mel queries
    let (mu, sd) = (ac.mel_mu, ac.mel_sd);
    let x = g.map(&[mel], |v| (v[0] - lit(mu)) * lit(1.0 / sd)).unwrap();
    let x = ir::conv1d(g, x, p(5), p(6));
    let x = g.map(&[x], |v| relu(v[0])).unwrap();
    let x = ir::linear(g, x, p(7));
    let x = ir::channel_map(g, x, &[p(8)], |x, b| relu(x + b[0]));
    let x = ir::linear(g, x, p(9));
    let q = ir::add_channel(g, x, p(10));
    // space (b, n, t, c)
    let d2 = g
        .comp(
            &[batch, n, t],
            &[ac.att],
            vec![arg(k, pick(4, &[Some(0), Some(3), Some(1)])), arg(q, pick(4, &[Some(0), Some(3), Some(2)]))],
            Some(Monoid::sum()),
            |v| {
                let d = v[1] - v[0];
                vec![d * d]
            },
        )
        .unwrap();
    let temp = ac.temp;
    let z = g.map(&[d2], |v| -(v[0] * lit(temp))).unwrap();
    AlignGraph { b, ac: ac.clone(), params, onehot, mel, z }
}

/// One training step: `fwd` + surrogate `Σ z ⊙ G` (G = host ∂L/∂z, data
/// "gz") + derived backward + global-norm clip + AdamW.
pub struct AlignTrain {
    pub ag: AlignGraph,
    pub gz: TensorId,
    pub m: Vec<TensorId>,
    pub v: Vec<TensorId>,
    pub hyper: TensorId,
    pub surrogate: TensorId,
    pub grads: Vec<TensorId>,
    /// `[grad_norm, p'…, m'…, v'…, partials…]`
    pub outputs: Vec<TensorId>,
}

pub fn train_graph(ac: &AConfig, batch: usize, n: usize, t: usize) -> AlignTrain {
    let mut ag = align_graph(ac, batch, n, t);
    let specs = ac.params();
    let gz = ag.b.input(&[batch, n, t], Role::Data("gz"));
    let m: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| ag.b.input(&s.shape, Role::AdamM(i))).collect();
    let v: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| ag.b.input(&s.shape, Role::AdamV(i))).collect();
    let hyper = ag.b.input(&[6], Role::Hyper);
    let g = &mut ag.b.g;
    let surrogate = ir::sum_all2(g, ag.z, Some(gz), |v| v[0] * v[1]);
    let grads = g.backward(surrogate, &ag.params).unwrap();
    let ad = model::adamw_betas(g, &specs, &ag.params, &grads, &m, &v, hyper, 0.9, 0.999);
    let mut outputs = vec![ad.grad_norm];
    outputs.extend(&ad.new_p);
    outputs.extend(&ad.new_m);
    outputs.extend(&ad.new_v);
    outputs.extend(ad.partials);
    AlignTrain { ag, gz, m, v, hyper, surrogate, grads, outputs }
}

/// The losses of [`batch_host`] written in the graph over distance logits
/// `z [B, N, T]`. Per-item lengths come in as data, built by
/// [`scan_data`]: "prior" `[B, N, T]` (log prior, [`Semiring::NEG`] on
/// tokens `≥ n_b`), "end" `[B, N, T]` (1 at `(n_b − 1, t_b − 1)`), "fw"
/// `[B, 1, T]` (`1/(B·t_b)` on frames `< t_b`, else 0) and "w_bin" `[1]`.
pub struct ScanLoss {
    pub prior: TensorId,
    pub end: TensorId,
    pub fw: TensorId,
    pub w_bin: TensorId,
    /// batch means, `[1]` each, as [`batch_host`] returns them
    pub loss_fs: TensorId,
    pub loss_bin: TensorId,
    pub sharp: TensorId,
    /// MAS path as a one-hot `[B, N, T]` (detached) and its run lengths `[B, N]`
    pub hard: TensorId,
    pub dur: TensorId,
    /// `loss_fs + w_bin·loss_bin`
    pub total: TensorId,
}

/// `log_softmax` over the token axis of `x [B, N, T]`, with an optional
/// extra class of fixed logit `extra`: `(x − lse, Σ exp(x − max), lse)`,
/// the last two `[B, 1, T]`. The max is detached (the result does not
/// depend on it), so no max-reduction backward is derived.
fn log_softmax_n(g: &mut TensorGraph, x: TensorId, extra: Option<f64>) -> (TensorId, TensorId, TensorId) {
    let [bs, n, t] = ir::dims(g, x)[..] else { panic!() };
    let over_n = pick(4, &[Some(0), Some(3), Some(2)]);
    let per_frame = pick(3, &[Some(0), None, Some(2)]);
    let mut m = g.comp(&[bs, 1, t], &[n], vec![arg(x, over_n.clone())], Some(Monoid::max()), |v| vec![v[0]]).unwrap();
    if let Some(e) = extra {
        m = g.map(&[m], |v| ExprId::select(v[0] - lit(e), v[0], lit(e))).unwrap();
    }
    let m = g.detach(m).unwrap();
    let mut l = g
        .comp(&[bs, 1, t], &[n], vec![arg(x, over_n), arg(m, pick(4, &[Some(0), None, Some(2)]))], Some(Monoid::sum()), |v| vec![(v[0] - v[1]).exp()])
        .unwrap();
    if let Some(e) = extra {
        l = g.map(&[l, m], |v| v[0] + (lit(e) - v[1]).exp()).unwrap();
    }
    let lse = g.map(&[m, l], |v| v[0] + v[1].ln()).unwrap();
    let lp = g.comp(&[bs, n, t], &[], vec![arg(x, View::identity(3, 3)), arg(lse, per_frame)], None, |v| vec![v[0] - v[1]]).unwrap();
    (lp, l, lse)
}

/// Forward-sum and MAS as banded scans along frames with the token axis as
/// the state: `S_t[i] = lp[t,i] ⊗ (S_{t−1}[i−1] ⊕ S_{t−1}[i])`, started by
/// `U_0[0] = lp[0,0]`. In the log semiring `S[t_b−1, n_b−1]` is `log Z`; in
/// max-plus it is the best path's score, whose gradient w.r.t. `lp` is that
/// path (MAS). A padded item reads its own end cell: the lattice only moves
/// up in tokens and forward in frames, so cells past `(n_b, t_b)` never
/// reach it, and padded tokens are excluded from the softmax by the prior.
///
/// With `blank` the forward-sum is [`forward_sum_ctc`]'s instead: states
/// `∅ t₀ ∅ t₁ … ∅` (`e = 2i + 1` is token `i`), bands `[-2, -1, 0]` with
/// the skip only into a token, `log Z = S[2n_b − 1] ⊕ S[2n_b]` at the last
/// frame. MAS and the binarization term stay on the blank-free lattice, as
/// on the host.
///
/// MAS ties go to the advance (tang's max-plus rule: smallest predecessor
/// index), where the host's [`mas`] stays.
pub fn scan_loss(b: &mut Built, z: TensorId, blank: Option<f64>) -> ScanLoss {
    let [bs, n, t] = ir::dims(&b.g, z)[..] else { panic!() };
    let neg = Semiring::NEG;
    let prior = b.input(&[bs, n, t], Role::Data("prior"));
    let end = b.input(&[bs, n, t], Role::Data("end"));
    let fw = b.input(&[bs, 1, t], Role::Data("fw"));
    let w_bin = b.input(&[1], Role::Data("w_bin"));
    // 1 where a path starts: (frame 0, token 0); CTC: (frame 0, states 0 and 1)
    let mut first = vec![0.0; t * n];
    first[0] = 1.0;
    let first = b.constant(&[t, n], first);
    let ne = 2 * n + 2;
    let ctc = blank.map(|_| {
        let mut first = vec![0.0; t * ne];
        first[0] = 1.0;
        first[1] = 1.0;
        // the skip band (k = 0, from e − 2) only enters a token state e ≥ 3
        let skip: Vec<f64> = (0..ne * 3).map(|i| if i % 3 == 0 && ((i / 3) % 2 == 0 || i / 3 < 3) { neg } else { 0.0 }).collect();
        // state 2n + 1 has no token: it pads the states to an even count
        let mut phantom = vec![0.0; n + 1];
        phantom[n] = neg;
        (b.constant(&[t, ne], first), b.constant(&[ne, 3], skip), b.constant(&[n + 1], phantom), b.constant(&[2], vec![0.0, 1.0]))
    });
    let g = &mut b.g;
    // lp = log_softmax_n(z + prior)
    let x = g.map(&[z, prior], |v| v[0] + v[1]).unwrap();
    let (lp, l, _) = log_softmax_n(g, x, None);
    // lattice in scan layout [B, T, N, ·]: both bands carry the emission
    let lat = pick(4, &[Some(0), Some(2), Some(1)]);
    let a = g.comp(&[bs, t, n, 2], &[], vec![arg(lp, lat.clone())], None, |v| vec![v[0]]).unwrap();
    let u = g
        .comp(&[bs, t, n, 1], &[], vec![arg(lp, lat.clone()), arg(first, pick(4, &[Some(1), Some(2)]))], None, |v| vec![ExprId::select(v[1] - lit(0.5), v[0], lit(neg))])
        .unwrap();
    let bands = [-1, 0];
    let best = g.semiring_scan_banded(a, u, &bands, 1, false, Semiring::MaxPlus).unwrap();
    // Σ over the end cells of f(S[t, i], S′, fw); space (one, b, i, t)
    let end_v = pick(4, &[Some(1), Some(2), Some(3)]);
    let fw_v = pick(4, &[Some(1), None, Some(3)]);
    let loss_fs = match (blank, ctc) {
        (Some(bl), Some((first, skip, phantom, cls))) => {
            // q = softmax over [blank, log_softmax_n(z) + prior]; padded tokens masked out of both
            let zm = g.map(&[z, prior], |v| ExprId::select(v[1] - lit(neg / 10.0), v[0], lit(neg))).unwrap();
            let (lz, _, _) = log_softmax_n(g, zm, None);
            let x = g.map(&[lz, prior], |v| v[0] + v[1]).unwrap();
            let (lq, _, all) = log_softmax_n(g, x, Some(bl));
            // em[b, t, 2i + c] = c = 0: blank, c = 1: token i; via [B, T, N+1, 2]
            let tok = view(vec![form(4, &[(0, 1)], 0), form(4, &[(2, 1)], 0), form(4, &[(1, 1)], 0)], true);
            let em = g
                .comp(
                    &[bs, t, n + 1, 2],
                    &[],
                    vec![arg(lq, tok), arg(all, pick(4, &[Some(0), None, Some(1)])), arg(phantom, pick(4, &[Some(2)])), arg(cls, pick(4, &[Some(3)]))],
                    None,
                    move |v| vec![ExprId::select(v[3] - lit(0.5), v[0] + v[2], lit(bl) - v[1])],
                )
                .unwrap();
            let em = g.reshape(em, &[bs, t, ne]).unwrap();
            let id3 = pick(4, &[Some(0), Some(1), Some(2)]);
            let a = g.comp(&[bs, t, ne, 3], &[], vec![arg(em, id3.clone()), arg(skip, pick(4, &[Some(2), Some(3)]))], None, |v| vec![v[0] + v[1]]).unwrap();
            let u = g
                .comp(&[bs, t, ne, 1], &[], vec![arg(em, id3), arg(first, pick(4, &[Some(1), Some(2)]))], None, |v| vec![ExprId::select(v[1] - lit(0.5), v[0], lit(neg))])
                .unwrap();
            let fwd = g.semiring_scan_banded(a, u, &[-2, -1, 0], 1, false, Semiring::Log).unwrap();
            // log Z = S[2i + 1] ⊕ S[2i + 2] at the end cell (i = n_b − 1)
            let st = |o: i64| view(vec![form(4, &[(1, 1)], 0), form(4, &[(3, 1)], 0), form(4, &[(2, 2)], o), form(4, &[], 0)], false);
            g.comp(&[1], &[bs, n, t], vec![arg(fwd, st(1)), arg(fwd, st(2)), arg(end, end_v.clone()), arg(fw, fw_v.clone())], Some(Monoid::sum()), |v| {
                let m = ExprId::select(v[0] - v[1], v[0], v[1]);
                let lz = m + ((v[0] - m).exp() + (v[1] - m).exp()).ln();
                vec![ExprId::select(v[2] - lit(0.5), -(lz * v[3]), ExprId::ZERO)]
            })
            .unwrap()
        }
        _ => {
            let fwd = g.semiring_scan_banded(a, u, &bands, 1, false, Semiring::Log).unwrap();
            g.comp(&[1], &[bs, n, t], vec![arg(fwd, pick(4, &[Some(1), Some(3), Some(2), None])), arg(end, end_v.clone()), arg(fw, fw_v.clone())], Some(Monoid::sum()), |v| {
                vec![ExprId::select(v[1] - lit(0.5), -(v[0] * v[2]), ExprId::ZERO)]
            })
            .unwrap()
        }
    };
    let score = g
        .comp(&[1], &[bs, n, t], vec![arg(best, pick(4, &[Some(1), Some(3), Some(2), None])), arg(end, end_v)], Some(Monoid::sum()), |v| vec![ExprId::select(v[1] - lit(0.5), v[0], ExprId::ZERO)])
        .unwrap();
    let hard = g.backward(score, &[lp]).unwrap()[0];
    let hard = g.detach(hard).unwrap();
    let on_path = |g: &mut TensorGraph, out: &[usize], red: &[usize], f: fn(&[ExprId]) -> ExprId| {
        // space (out…, red…) = (one, b, i, t) or (b, i, t)
        let r = out.len() + red.len();
        let d = |k: usize| Some(r - 3 + k);
        g.comp(out, red, vec![arg(hard, pick(r, &[d(0), d(1), d(2)])), arg(lp, pick(r, &[d(0), d(1), d(2)])), arg(fw, pick(r, &[d(0), None, d(2)]))], Some(Monoid::sum()), move |v| vec![f(v)])
            .unwrap()
    };
    let loss_bin = on_path(g, &[1], &[bs, n, t], |v| ExprId::select(v[0] - lit(0.5), -(v[1] * v[2]), ExprId::ZERO));
    let dur = on_path(g, &[bs, n], &[t], |v| v[0]);
    // max_n A = exp(max − lse) = 1/l
    let sharp = g
        .comp(&[1], &[bs, t], vec![arg(l, pick(3, &[Some(1), None, Some(2)])), arg(fw, pick(3, &[Some(1), None, Some(2)]))], Some(Monoid::sum()), |v| vec![v[1] / v[0]])
        .unwrap();
    let total = g.map(&[loss_fs, loss_bin, w_bin], |v| v[0] + v[2] * v[1]).unwrap();
    ScanLoss { prior, end, fw, w_bin, loss_fs, loss_bin, sharp, hard, dur, total }
}

/// Host data for [`scan_loss`]: `(prior, end, fw)` for a padded batch.
pub fn scan_data(lens: &[(usize, usize)], nmax: usize, tmax: usize, prior_scaling: f64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let bsz = lens.len();
    let mut prior = vec![0.0f32; bsz * nmax * tmax];
    let mut end = vec![0.0f32; bsz * nmax * tmax];
    let mut fw = vec![0.0f32; bsz * tmax];
    for (b, &(n, t)) in lens.iter().enumerate() {
        assert!(t >= n && n > 0 && n <= nmax && t <= tmax, "need tokens ≤ frames within the padded shape ({n}, {t})");
        let lp = log_prior(n, t, prior_scaling);
        for ni in 0..nmax {
            for ti in 0..tmax {
                prior[(b * nmax + ni) * tmax + ti] = if ni >= n { Semiring::NEG as f32 } else if ti < t { lp[ti * n + ni] as f32 } else { 0.0 };
            }
        }
        end[(b * nmax + n - 1) * tmax + t - 1] = 1.0;
        fw[b * tmax..b * tmax + t].fill((1.0 / (bsz * t) as f64) as f32);
    }
    (prior, end, fw)
}

/// One training step in one program: [`align_graph`] + [`scan_loss`] +
/// derived backward + global-norm clip + AdamW. `blank` as in [`utt_host`].
pub struct AlignScanTrain {
    pub ag: AlignGraph,
    pub sl: ScanLoss,
    pub m: Vec<TensorId>,
    pub v: Vec<TensorId>,
    pub hyper: TensorId,
    pub grads: Vec<TensorId>,
    /// `[grad_norm, p'…, m'…, v'…, loss_fs, loss_bin, sharp, dur, partials…]`
    pub outputs: Vec<TensorId>,
}

pub fn scan_train_graph(ac: &AConfig, batch: usize, n: usize, t: usize, blank: Option<f64>) -> AlignScanTrain {
    let mut ag = align_graph(ac, batch, n, t);
    let specs = ac.params();
    let sl = scan_loss(&mut ag.b, ag.z, blank);
    let m: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| ag.b.input(&s.shape, Role::AdamM(i))).collect();
    let v: Vec<TensorId> = specs.iter().enumerate().map(|(i, s)| ag.b.input(&s.shape, Role::AdamV(i))).collect();
    let hyper = ag.b.input(&[6], Role::Hyper);
    let g = &mut ag.b.g;
    let grads = g.backward(sl.total, &ag.params).unwrap();
    let ad = model::adamw_betas(g, &specs, &ag.params, &grads, &m, &v, hyper, 0.9, 0.999);
    let mut outputs = vec![ad.grad_norm];
    outputs.extend(&ad.new_p);
    outputs.extend(&ad.new_m);
    outputs.extend(&ad.new_v);
    outputs.extend([sl.loss_fs, sl.loss_bin, sl.sharp, sl.dur]);
    outputs.extend(ad.partials);
    AlignScanTrain { ag, sl, m, v, hyper, grads, outputs }
}

/// One vector per graph input of `b`, in input order (aligner params).
pub fn assemble(
    b: &Built,
    specs: &[ParamSpec],
    params: &[Vec<f32>],
    adam: Option<(&[Vec<f32>], &[Vec<f32>])>,
    data: &[(&str, &[f32])],
) -> Vec<Vec<f32>> {
    b.inputs
        .iter()
        .map(|s| match &s.role {
            Role::Param(i) => params[*i].clone(),
            Role::AdamM(i) => adam.map_or_else(|| vec![0.0; specs[*i].numel()], |a| a.0[*i].clone()),
            Role::AdamV(i) => adam.map_or_else(|| vec![0.0; specs[*i].numel()], |a| a.1[*i].clone()),
            Role::Hyper => vec![0.0; 6],
            Role::Const(v) => v.clone(),
            Role::Data(name) => {
                let n: usize = b.g.shape(s.id).iter().map(|d| d.n()).product();
                data.iter().find(|(k, _)| k == name).map_or_else(|| vec![0.0; n], |(_, v)| v.to_vec())
            }
            r => panic!("aligner graph has no {r:?} input"),
        })
        .collect()
}

pub fn init_params(ac: &AConfig, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = model::Rng::new(seed);
    ac.params().iter().map(|s| s.init_data(&mut rng)).collect()
}

/// One-hot `[B, V, N]` for token rows (shorter rows zero-padded).
pub fn onehot(rows: &[&[u16]], vocab: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows.len() * vocab * n];
    for (b, r) in rows.iter().enumerate() {
        for (i, &id) in r.iter().enumerate().take(n) {
            out[(b * vocab + id as usize) * n + i] = 1.0;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// host side: prior, forward–backward, MAS, losses, ∂L/∂z

/// ln Γ(x), x > 0 (Lanczos, g = 7, n = 9; |rel err| < 1e-14).
pub fn lgamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// RAD-TTS's beta-binomial alignment prior, as log-probabilities
/// `[t·N + n]`: frame `t` (1-based `i`) gets `BetaBinomial(N−1, s·i,
/// s·(T+1−i))` over token positions. RAD-TTS uses `betabinom(N, …)` on
/// `0..N−1` (a slightly unnormalized row); we use `N−1` so each row is a
/// distribution. The row is normalized again by the softmax anyway.
pub fn log_prior(n: usize, t: usize, scaling: f64) -> Vec<f64> {
    let mut out = vec![0.0; n * t];
    let nn = (n - 1) as f64;
    for ti in 0..t {
        let i = (ti + 1) as f64;
        let (a, b) = (scaling * i, scaling * (t as f64 + 1.0 - i));
        // pmf(0) = B(a, n+b) / B(a, b)
        let mut lp = lgamma(nn + b) + lgamma(a + b) - lgamma(a + nn + b) - lgamma(b);
        let row = &mut out[ti * n..(ti + 1) * n];
        row[0] = lp;
        for k in 0..n - 1 {
            let kf = k as f64;
            lp += ((nn - kf) / (kf + 1.0)).ln() + ((kf + a) / (nn - kf - 1.0 + b)).ln();
            row[k + 1] = lp;
        }
    }
    out
}

fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Per-utterance host result. `grad` is `∂L/∂z` laid out `[n·T + t]`
/// (the caller scales it by 1/B and scatters it into the padded batch).
pub struct UttOut {
    pub loss_fs: f64,
    pub loss_bin: f64,
    pub grad: Vec<f64>,
    /// MAS run length per token (each ≥ 1)
    pub dur: Vec<u32>,
    /// mean over frames of max_n A[t, n] (1 = hard alignment)
    pub sharp: f64,
}

/// Log soft alignment `lp[t·N + n] = log_softmax_n(z[n,t] + log prior)`.
pub fn log_attn(z: &dyn Fn(usize, usize) -> f64, n: usize, t: usize, prior: &[f64]) -> Vec<f64> {
    let mut lp = vec![0.0; n * t];
    for ti in 0..t {
        let row = &mut lp[ti * n..(ti + 1) * n];
        let mut m = f64::NEG_INFINITY;
        for ni in 0..n {
            row[ni] = z(ni, ti) + prior[ti * n + ni];
            m = m.max(row[ni]);
        }
        let s: f64 = row.iter().map(|v| (v - m).exp()).sum();
        let l = m + s.ln();
        row.iter_mut().for_each(|v| *v -= l);
    }
    lp
}

/// Forward–backward over monotonic paths that start at token 0, end at
/// token N−1 and advance by 0 or 1 token per frame. Returns `(log Z,
/// posterior γ[t·N + n])`.
pub fn forward_sum(lp: &[f64], n: usize, t: usize) -> (f64, Vec<f64>) {
    let ninf = f64::NEG_INFINITY;
    let mut a = vec![ninf; n * t];
    a[0] = lp[0];
    for ti in 1..t {
        for ni in 0..n.min(ti + 1) {
            let prev = if ni > 0 { lse(a[(ti - 1) * n + ni], a[(ti - 1) * n + ni - 1]) } else { a[(ti - 1) * n] };
            a[ti * n + ni] = prev + lp[ti * n + ni];
        }
    }
    let mut bt = vec![ninf; n * t];
    bt[(t - 1) * n + n - 1] = 0.0;
    for ti in (0..t - 1).rev() {
        for ni in 0..n {
            let stay = bt[(ti + 1) * n + ni] + lp[(ti + 1) * n + ni];
            let adv = if ni + 1 < n { bt[(ti + 1) * n + ni + 1] + lp[(ti + 1) * n + ni + 1] } else { ninf };
            bt[ti * n + ni] = lse(stay, adv);
        }
    }
    let lz = a[(t - 1) * n + n - 1];
    let gamma = a.iter().zip(&bt).map(|(x, y)| if *x == ninf || *y == ninf { 0.0 } else { (x + y - lz).exp() }).collect();
    (lz, gamma)
}

/// Monotonic alignment search (Viterbi over the same paths): token per frame.
pub fn mas(lp: &[f64], n: usize, t: usize) -> Vec<u32> {
    let ninf = f64::NEG_INFINITY;
    let mut v = vec![ninf; n * t];
    let mut adv = vec![false; n * t];
    v[0] = lp[0];
    for ti in 1..t {
        for ni in 0..n.min(ti + 1) {
            let stay = v[(ti - 1) * n + ni];
            let from = if ni > 0 { v[(ti - 1) * n + ni - 1] } else { ninf };
            let (best, a) = if from > stay { (from, true) } else { (stay, false) };
            v[ti * n + ni] = best + lp[ti * n + ni];
            adv[ti * n + ni] = a;
        }
    }
    let mut path = vec![0u32; t];
    let mut ni = n - 1;
    for ti in (0..t).rev() {
        path[ti] = ni as u32;
        if ti > 0 && adv[ti * n + ni] {
            ni -= 1;
        }
    }
    debug_assert_eq!(ni, 0);
    path
}

/// RAD-TTS's forward-sum as CTC with a blank (arXiv 2108.10447, NVIDIA's
/// `ForwardSumLoss`): per frame, classes are a blank with a fixed logit
/// `blank` plus the tokens with `s_n = log_softmax_n(z) + log prior`,
/// renormalized over all N+1; paths are CTC paths over
/// `∅ t₀ ∅ t₁ … t_{N−1} ∅` (every token ≥ 1 frame, blanks optional).
/// Returns `(log Z, ∂(−log Z)/∂z [t·N + n])`:
/// `∂/∂z_n = (q_n − γ_n) − p_n·(γ_∅ − q_∅)` with `q` the renormalized class
/// probabilities, `γ` the class posteriors, `p = softmax(z)`.
pub fn forward_sum_ctc(z: &dyn Fn(usize, usize) -> f64, n: usize, t: usize, prior: &[f64], blank: f64) -> (f64, Vec<f64>) {
    let ninf = f64::NEG_INFINITY;
    let ne = 2 * n + 1;
    // per-frame: p (softmax z), lq (tokens), lqb (blank)
    let mut p = vec![0.0; n * t];
    let mut lq = vec![0.0; n * t];
    let mut lqb = vec![0.0; t];
    for ti in 0..t {
        let mut m = ninf;
        for ni in 0..n {
            m = m.max(z(ni, ti));
        }
        let lse_z = m + (0..n).map(|ni| (z(ni, ti) - m).exp()).sum::<f64>().ln();
        let mut mm = blank;
        for ni in 0..n {
            let lz = z(ni, ti) - lse_z;
            p[ti * n + ni] = lz.exp();
            lq[ti * n + ni] = lz + prior[ti * n + ni];
            mm = mm.max(lq[ti * n + ni]);
        }
        let all = mm + ((blank - mm).exp() + (0..n).map(|ni| (lq[ti * n + ni] - mm).exp()).sum::<f64>()).ln();
        for ni in 0..n {
            lq[ti * n + ni] -= all;
        }
        lqb[ti] = blank - all;
    }
    let em = |ti: usize, e: usize| if e % 2 == 0 { lqb[ti] } else { lq[ti * n + (e - 1) / 2] };
    let mut a = vec![ninf; ne * t];
    a[0] = em(0, 0);
    a[1] = em(0, 1);
    for ti in 1..t {
        for e in 0..ne {
            let mut v = a[(ti - 1) * ne + e];
            if e >= 1 {
                v = lse(v, a[(ti - 1) * ne + e - 1]);
            }
            if e % 2 == 1 && e >= 3 {
                v = lse(v, a[(ti - 1) * ne + e - 2]);
            }
            if v > ninf {
                a[ti * ne + e] = v + em(ti, e);
            }
        }
    }
    let mut b = vec![ninf; ne * t];
    b[(t - 1) * ne + ne - 1] = 0.0;
    b[(t - 1) * ne + ne - 2] = 0.0;
    for ti in (0..t - 1).rev() {
        for e in 0..ne {
            let mut v = b[(ti + 1) * ne + e] + em(ti + 1, e);
            if e + 1 < ne {
                v = lse(v, b[(ti + 1) * ne + e + 1] + em(ti + 1, e + 1));
            }
            if e % 2 == 1 && e + 2 < ne {
                v = lse(v, b[(ti + 1) * ne + e + 2] + em(ti + 1, e + 2));
            }
            b[ti * ne + e] = v;
        }
    }
    let lz = lse(a[(t - 1) * ne + ne - 1], a[(t - 1) * ne + ne - 2]);
    let mut grad = vec![0.0; n * t];
    for ti in 0..t {
        let post = |e: usize| {
            let (x, y) = (a[ti * ne + e], b[ti * ne + e]);
            if x == ninf || y == ninf { 0.0 } else { (x + y - lz).exp() }
        };
        let gb: f64 = (0..ne).step_by(2).map(post).sum();
        let qb = lqb[ti].exp();
        for ni in 0..n {
            let g = post(2 * ni + 1);
            grad[ti * n + ni] = (lq[ti * n + ni].exp() - g) - p[ti * n + ni] * (gb - qb);
        }
    }
    (lz, grad)
}

/// Losses and `∂(L_fs + w_bin·L_bin)/∂z` for one utterance of `n` tokens
/// and `t ≥ n` frames. Both losses are per frame (divided by `t`).
/// With `blank = None`, `L_fs = −log Σ_paths Π_t A[t, path_t]` over
/// blank-free monotonic paths, `∂/∂z = (A − γ)/t`; with `Some(b)` it is
/// [`forward_sum_ctc`]. `L_bin = −Σ_t log A[t, mas_t]`,
/// `∂/∂z = (A − onehot(mas))/t`, with `A = softmax_n(z + log prior)`.
pub fn utt_host(z: &dyn Fn(usize, usize) -> f64, n: usize, t: usize, prior: &[f64], w_bin: f64, blank: Option<f64>) -> UttOut {
    assert!(t >= n && n > 0, "need frames ≥ tokens ({t} < {n})");
    let lp = log_attn(z, n, t, prior);
    let (lz, gfs) = match blank {
        None => {
            let (lz, gamma) = forward_sum(&lp, n, t);
            let g = lp.iter().zip(&gamma).map(|(l, g)| l.exp() - g).collect::<Vec<_>>();
            (lz, g)
        }
        Some(bl) => forward_sum_ctc(z, n, t, prior, bl),
    };
    let path = mas(&lp, n, t);
    let tf = t as f64;
    let mut grad = vec![0.0; n * t];
    let mut loss_bin = 0.0;
    let mut sharp = 0.0;
    for ti in 0..t {
        let mut mx: f64 = 0.0;
        for ni in 0..n {
            let a = lp[ti * n + ni].exp();
            mx = mx.max(a);
            let hard = if path[ti] as usize == ni { 1.0 } else { 0.0 };
            grad[ni * t + ti] = gfs[ti * n + ni] / tf + w_bin * (a - hard) / tf;
        }
        loss_bin -= lp[ti * n + path[ti] as usize];
        sharp += mx;
    }
    let mut dur = vec![0u32; n];
    for &p in &path {
        dur[p as usize] += 1;
    }
    UttOut { loss_fs: -lz / tf, loss_bin: loss_bin / tf, grad, dur, sharp: sharp / tf }
}

/// Host step for a padded batch: `z` is `[B, N, T]`; `lens[b] = (n_b, t_b)`.
/// Returns `(G [B,N,T] (batch mean), mean L_fs, mean L_bin, mean sharpness,
/// durations)`. Utterances run on parallel threads.
pub fn batch_host(
    z: &[f32],
    nmax: usize,
    tmax: usize,
    lens: &[(usize, usize)],
    w_bin: f64,
    prior_scaling: f64,
    blank: Option<f64>,
) -> (Vec<f32>, f64, f64, f64, Vec<Vec<u32>>) {
    let bsz = lens.len();
    let outs: Vec<UttOut> = std::thread::scope(|s| {
        let hs: Vec<_> = lens
            .iter()
            .enumerate()
            .map(|(b, &(n, t))| {
                s.spawn(move || {
                    let prior = log_prior(n, t, prior_scaling);
                    let zf = |ni: usize, ti: usize| z[(b * nmax + ni) * tmax + ti] as f64;
                    utt_host(&zf, n, t, &prior, w_bin, blank)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut g = vec![0.0f32; bsz * nmax * tmax];
    let (mut lf, mut lb, mut sh) = (0.0, 0.0, 0.0);
    let inv = 1.0 / bsz as f64;
    let mut durs = Vec::with_capacity(bsz);
    for (b, (o, &(n, t))) in outs.into_iter().zip(lens).enumerate() {
        for ni in 0..n {
            for ti in 0..t {
                g[(b * nmax + ni) * tmax + ti] = (o.grad[ni * t + ti] * inv) as f32;
            }
        }
        lf += o.loss_fs * inv;
        lb += o.loss_bin * inv;
        sh += o.sharp * inv;
        durs.push(o.dur);
    }
    (g, lf, lb, sh, durs)
}
