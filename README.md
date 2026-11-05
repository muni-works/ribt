# ribt

Real-time, In-Body Talk: a small English text-to-speech model for robots. First audio in 17.6 ms
(median, streaming, M4 Max), 37× faster than real time, 54M parameters. Trained and run entirely in
[tang](https://github.com/muni-works/tang)'s tensor IR, with no PyTorch in any training loop, so the same
programs compile to Metal, CUDA, wgpu, OpenCL, plain C and WebGPU.

- Site and live demo: [ribt.dev](https://ribt.dev)
- Weights: [huggingface.co/municipal-robotics/ribt](https://huggingface.co/municipal-robotics/ribt)
- Paper: [muni.works/research/ribt](https://muni.works/research/ribt)

## Pipeline

text → **lilt** (misaki's English G2P, ported to Rust) → **acoustic** (FastSpeech2-shaped encoder,
variance adaptor, ConvNeXt decoder) → **flow** (OT-CFM decoder distilled to a one-step shortcut student)
→ **vocoder** (Vocos 24 kHz, imported into our graph) → 24 kHz audio.

| module | what |
|---|---|
| `src/lilt/` | misaki 0.9.4 `en.G2P` (US) port: tokenizer, averaged-perceptron tagger, G2P, embedded lexicons, numbers, Kokoro vocab, unknown-word ladder, streaming `Phonemizer::push` |
| `src/model.rs`, `src/gan.rs` | vocoder graph, log-mel frontend, iSTFT, losses, AdamW, F0 source; MPD + MRD discriminators |
| `src/align.rs` | aligner graph, host CTC forward-sum and MAS |
| `src/acoustic.rs` | acoustic model graphs (fixed-shape buckets) |
| `src/flow.rs` | OT-CFM decoder (teacher and shortcut student) |
| `src/runtime.rs` | one GPU backend per build: compile and run IR programs |
| `src/dsp.rs`, `src/data.rs`, `src/ckpt.rs` | host DSP, data loading, safetensors checkpoints |
| `web/` | the browser build: lilt as wasm, every model on WebGPU from tang's exported WGSL |

Binaries: `ribt` (vocoder, lilt tools), `ribt-tts` (acoustic, flow, synthesis, streaming),
`ribt-align` (aligner, durations), `ribt-web-export` (WGSL plans and fp16 weights for `web/`).

## Build

ribt is its own cargo workspace, pinned to a tang git rev. Exactly one GPU backend per build:

    cargo build --release                                          # macOS, Metal (default)
    cargo build --release --no-default-features --features cuda    # Linux, NVIDIA
    cargo build --release --no-default-features --features wgpu
    cargo build --release --no-default-features --features opencl  # e.g. Adreno
    cargo build --release --no-default-features --features cpu     # C lowering, system cc

Tests: `cargo test --release` (GPU tests are `#[ignore]`d; `cargo test --release --test lilt_parity` for lilt).

## Speak

    hf download municipal-robotics/ribt --local-dir ribt-weights
    W=ribt-weights
    target/release/ribt-tts synth --run $W --ckpt acoustic.safetensors \
        --flow $W/flow.safetensors --nfe 1 --shortcut --temp 0.3 \
        --voc $W/vocoder.safetensors --ref $W/ref.wav \
        --seed 0 --pitch 1 --accent minnesota \
        --text "Hi, I'm right here. What do you need?" --out-dir out/

Streaming (prints each committed phoneme and its sample offset to stderr): the same flags with
`ribt-tts stream`. Knobs: `--pitch` (semitones), `--rate`, `--accent neutral|midwest|minnesota`.
Text to phonemes only: `target/release/ribt lilt say "Hannah paid \$3.50 at 4:15."`.

## In the browser

    web/build.sh ribt-weights          # wasm + WGSL graphs + fp16 weights + parity references
    python3 -m http.server 8737 -d web # then /index.html and /parity.html

See [`web/README.md`](web/README.md) for the JS API and parity numbers.

## Training

Every component trains in the IR: `ribt train` / `ribt gan` (vocoder), `ribt-align train` then
`ribt-align durs` (aligner and durations), `ribt-tts train` (acoustic), `ribt-tts flow-train`
(flow teacher, then `--distill` for the student). Each binary's flags are documented at the top of its
source file. Touch `RUN/STOP` to checkpoint and exit; rerun to resume. The paper describes the recipe
and the experiments behind it.

## Where the voice comes from

ribt's voice is distilled from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)'s `af_heart`
voice (Apache-2.0): we rendered 135 hours of af_heart and trained our own models on it. No Kokoro
weights are in ribt. Kokoro's own training data included synthetic audio from closed commercial TTS
providers, so the voice inherits that lineage. Details in [`release.toml`](release.toml).

## License

Code: MIT ([`LICENSE`](LICENSE)). lilt is a port of hexgrad's [misaki](https://github.com/hexgrad/misaki)
(Apache-2.0); its embedded lexicons ship under Apache-2.0, with attribution in
[`data/lilt/NOTICE`](data/lilt/NOTICE). The imported vocoder weights are
[Vocos](https://github.com/gemelo-ai/vocos) by Hubert Siuzdak (MIT).

Built by [Municipal Robotics](https://muni.works).
