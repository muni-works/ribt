//! Small builders over tang's tensor IR. Every op here is one
//! comprehension (`out[i] = ⊕_k body(x[view(i,k)]…)`); the layout is
//! `[B, C, T]` (batch, channels, frames) throughout.
//!
//! Views that read with an offset (convs, framing, overlap-add) are
//! *guarded*, even where the forward never leaves range: tang's backward
//! inverts a guarded `t + k − p` by eliminating `t`, which keeps every
//! gradient a small reduction. An unguarded offset view would become a
//! gated scatter whose cost multiplies by the frame count.

use tang::Scalar;
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

pub fn form(rank: usize, terms: &[(usize, i64)], offset: i64) -> Form {
    let mut f = Form::zero(rank);
    for &(d, c) in terms {
        f.coefs[d] += c;
    }
    f.offset = offset;
    f
}

pub fn view(forms: Vec<Form>, guard: bool) -> View {
    View { forms, guard }
}

pub fn pick(rank: usize, picks: &[Option<usize>]) -> View {
    View::pick(rank, picks)
}

pub fn arg(src: TensorId, view: View) -> Arg {
    Arg { src, view }
}

pub fn lit(v: f64) -> ExprId {
    ExprId::from_f64(v)
}

/// tanh-approximate GELU, written with exp only (`F.gelu(approximate="tanh")`).
/// `u` is clamped to ±15 (tanh is ±1 in f32 well before that): unclamped,
/// `exp(2u)` overflows for x ≳ 30 and the derivative becomes inf/inf = NaN.
pub fn gelu(x: ExprId) -> ExprId {
    let u = lit((2.0 / std::f64::consts::PI).sqrt()) * (x + lit(0.044715) * x * x * x);
    let u = ExprId::select(u - lit(15.0), lit(15.0), ExprId::select(lit(-15.0) - u, lit(-15.0), u));
    let th = lit(1.0) - lit(2.0) / ((lit(2.0) * u).exp() + lit(1.0));
    lit(0.5) * x * (lit(1.0) + th)
}

pub fn dims(g: &TensorGraph, t: TensorId) -> Vec<usize> {
    g.shape(t).iter().map(|d| d.n()).collect()
}

/// `y[b,o,t] = Σ_{c,k} w[o,c,k] · x[b,c,t+k−p] + bias[o]`, "same" zero padding.
pub fn conv1d(g: &mut TensorGraph, x: TensorId, w: TensorId, bias: TensorId) -> TensorId {
    let y = conv1d_nobias(g, x, w);
    add_channel(g, y, bias)
}

/// `y[b,o,t] = Σ_{c,k} w[o,c,k] · x[b,c,t+k−p]`, "same" zero padding.
pub fn conv1d_nobias(g: &mut TensorGraph, x: TensorId, w: TensorId) -> TensorId {
    let [b, ci, t] = dims(g, x)[..] else { panic!() };
    let [co, _, k] = dims(g, w)[..] else { panic!() };
    let p = (k as i64 - 1) / 2;
    // space (b, o, t, c, k)
    let xv = view(
        vec![form(5, &[(0, 1)], 0), form(5, &[(3, 1)], 0), form(5, &[(2, 1), (4, 1)], -p)],
        true,
    );
    let y = g
        .comp(
            &[b, co, t],
            &[ci, k],
            vec![arg(x, xv), arg(w, pick(5, &[Some(1), Some(3), Some(4)]))],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    y
}

/// Depthwise conv: `y[b,c,t] = Σ_k w[c,k] · x[b,c,t+k−p] + bias[c]`.
pub fn dwconv(g: &mut TensorGraph, x: TensorId, w: TensorId, bias: TensorId) -> TensorId {
    let [b, c, t] = dims(g, x)[..] else { panic!() };
    let k = dims(g, w)[1];
    let p = (k as i64 - 1) / 2;
    // space (b, c, t, k)
    let xv = view(
        vec![form(4, &[(0, 1)], 0), form(4, &[(1, 1)], 0), form(4, &[(2, 1), (3, 1)], -p)],
        true,
    );
    let y = g
        .comp(
            &[b, c, t],
            &[k],
            vec![arg(x, xv), arg(w, pick(4, &[Some(1), Some(3)]))],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    add_channel(g, y, bias)
}

/// `y[b,o,t] = Σ_c w[o,c] · x[b,c,t]` (a pointwise conv / per-frame linear).
pub fn linear(g: &mut TensorGraph, x: TensorId, w: TensorId) -> TensorId {
    let [b, ci, t] = dims(g, x)[..] else { panic!() };
    let co = dims(g, w)[0];
    assert_eq!(dims(g, w)[1], ci);
    // space (b, o, t, c)
    g.comp(
        &[b, co, t],
        &[ci],
        vec![
            arg(w, pick(4, &[Some(1), Some(3)])),
            arg(x, pick(4, &[Some(0), Some(3), Some(2)])),
        ],
        Some(Monoid::sum()),
        |v| vec![v[0] * v[1]],
    )
    .unwrap()
}

/// `x[b,c,t] + v[c]`.
pub fn add_channel(g: &mut TensorGraph, x: TensorId, v: TensorId) -> TensorId {
    channel_map(g, x, &[v], |x, p| x + p[0])
}

/// `f(x[b,c,t], p₀[c], p₁[c], …)` elementwise, per-channel params broadcast.
pub fn channel_map(
    g: &mut TensorGraph,
    x: TensorId,
    params: &[TensorId],
    f: impl FnOnce(ExprId, &[ExprId]) -> ExprId,
) -> TensorId {
    let sh = dims(g, x);
    let mut args = vec![arg(x, View::identity(3, 3))];
    args.extend(params.iter().map(|&p| arg(p, pick(3, &[Some(1)]))));
    g.comp(&sh, &[], args, None, |v| vec![f(v[0], &v[1..])]).unwrap()
}

/// Sum over channels: `[B,C,T] -> [B,1,T]` of `f(x)`.
fn channel_sum(g: &mut TensorGraph, x: TensorId, f: impl FnOnce(ExprId) -> ExprId) -> TensorId {
    let [b, c, t] = dims(g, x)[..] else { panic!() };
    // space (b, one, t, c)
    g.comp(
        &[b, 1, t],
        &[c],
        vec![arg(x, pick(4, &[Some(0), Some(3), Some(2)]))],
        Some(Monoid::sum()),
        |v| vec![f(v[0])],
    )
    .unwrap()
}

/// LayerNorm over channels (per frame), affine `gamma[c]`, `beta[c]`.
pub fn layer_norm(
    g: &mut TensorGraph,
    x: TensorId,
    gamma: TensorId,
    beta: TensorId,
    eps: f64,
) -> TensorId {
    let [_, c, _] = dims(g, x)[..] else { panic!() };
    let inv_c = 1.0 / c as f64;
    let s1 = channel_sum(g, x, |v| v);
    let mean = g.map(&[s1], |v| v[0] * lit(inv_c)).unwrap();
    // centered second moment, reading the mean through a broadcast view
    let sh = dims(g, x);
    let d = g
        .comp(
            &sh,
            &[],
            vec![
                arg(x, View::identity(3, 3)),
                arg(mean, pick(3, &[Some(0), None, Some(2)])),
            ],
            None,
            |v| vec![v[0] - v[1]],
        )
        .unwrap();
    let s2 = channel_sum(g, d, |v| v * v);
    let inv = g
        .map(&[s2], |v| lit(1.0) / (v[0] * lit(inv_c) + lit(eps)).sqrt())
        .unwrap();
    g.comp(
        &sh,
        &[],
        vec![
            arg(d, View::identity(3, 3)),
            arg(inv, pick(3, &[Some(0), None, Some(2)])),
            arg(gamma, pick(3, &[Some(1)])),
            arg(beta, pick(3, &[Some(1)])),
        ],
        None,
        |v| vec![v[0] * v[1] * v[2] + v[3]],
    )
    .unwrap()
}

/// `Σ_all f(a, b)` of two equally shaped tensors, as a `[1]` tensor.
pub fn sum_all2(
    g: &mut TensorGraph,
    a: TensorId,
    b: Option<TensorId>,
    f: impl FnOnce(&[ExprId]) -> ExprId,
) -> TensorId {
    let sh = dims(g, a);
    let r = sh.len();
    let picks: Vec<Option<usize>> = (1..=r).map(Some).collect();
    let mut args = vec![arg(a, pick(r + 1, &picks))];
    if let Some(b) = b {
        assert_eq!(dims(g, b), sh);
        args.push(arg(b, pick(r + 1, &picks)));
    }
    g.comp(&[1], &sh, args, Some(Monoid::sum()), |v| vec![f(v)])
        .unwrap()
}

/// `|x|` with a well-defined (zero) gradient at 0.
pub fn abs(x: ExprId) -> ExprId {
    ExprId::select(x, x, -x)
}

/// `x` forced to its own buffer: an identity written as a reduction over a
/// size-1 dim, which the optimizer never inlines into readers.
pub fn materialize(g: &mut TensorGraph, x: TensorId) -> TensorId {
    let sh = dims(g, x);
    let r = sh.len();
    let picks: Vec<Option<usize>> = (0..r).map(Some).collect();
    g.comp(&sh, &[1], vec![arg(x, pick(r + 1, &picks))], Some(Monoid::sum()), |v| vec![v[0]]).unwrap()
}
