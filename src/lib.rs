//! ribt: RIBT (Real-time, In-Body Talk), a small English TTS (lilt G2P,
//! aligner, acoustic model, Vocos-style vocoder) defined, trained and run
//! in tang's tensor IR. See `the ribt paper`.

pub mod acoustic;
pub mod align;
pub mod barge;
pub mod body;
pub mod ckpt;
pub mod data;
pub mod dsp;
pub mod flow;
pub mod fx;
pub mod gan;
pub mod glottis;
pub mod ir;
pub mod lilt;
pub mod model;
pub mod persona;
pub mod runtime;
#[cfg(feature = "cpu")]
pub mod tang_cpu;
#[cfg(feature = "opencl")]
pub mod tang_cl;
pub mod reward;
pub mod surprise;
