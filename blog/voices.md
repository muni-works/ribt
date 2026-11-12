---
title: "Multiple voices for ribt"
date: 2026-11-12
slug: ribt-voices
---

ribt now speaks in many voices. The shipped model speaks as af_heart (a female voice the base was fine-tuned to). With this update, you can fit voices for any speaker in minutes, all sharing the same model weights.

## The voice ladder

| rung | what | file size | quality | when to use |
|---|---|---|---|---|
| 1 | zero-shot | ~3 KB | baseline | quick test, no training data |
| 2 | embedding fit | ~1 MB | better | you have 10-50 clips of the speaker |
| 3 | adapter fit | ~1–16 MB | best | you want the highest quality |

Every voice runs the same acoustic/flow/vocoder graphs at the same speed. Only voice-file size changes.

## Quick start

```bash
# Zero-shot: derive a voice from two reference clips
ribt-tts synth --run $W --ckpt acoustic.safetensors \
  --flow $W/flow.safetensors --nfe 1 --shortcut --temp 0.3 \
  --voc $W/vocoder.safetensors --ref ref1.wav+ref2.wav \
  --save-voice myvoice.voice --name "My Voice" \
  --text "Hello world." --out-dir out/

# Speak with a saved voice
ribt-tts synth --run $W --ckpt acoustic.safetensors \
  --flow $W/flow.safetensors --nfe 1 --shortcut --temp 0.3 \
  --voc $W/vocoder.safetensors --voice myvoice.voice \
  --text "Hello world." --out-dir out/
```

## Fitting a voice

```bash
# Prepare your clips: each speaker needs a directory with WAV files and a lines.tsv
# Format: id<TAB>metadata<TAB>text (text is the last column)

# Embedding fit (rung 2): train only the 640 speaker vectors
ribt-tts voice-fit --run $W --ckpt acoustic.safetensors \
  --voice init.voice --lines fit.tsv --wav-dir ./clips \
  --steps 300 --lr 3e-3 --batch 8 --rank 0 \
  --out voice7127.voice

# Adapter fit (rung 3): also train a low-rank update on the decoder
ribt-tts voice-fit --run $W --ckpt acoustic.safetensors \
  --voice init.voice --lines fit.tsv --wav-dir ./clips \
  --steps 300 --lr 3e-3 --batch 8 --rank 8 \
  --out voice7127.voice
```

## Quality vs rank

On speaker 7127 (male, 14 sentences, 419 words), rank 0→128 gives:

| rank | ECAPA similarity | WER | voice size |
|---|---|---|---|
| 0 | 0.441 | 4.53% | 3 KB |
| 8 | 0.664 | 4.53% | 1.0 MB |
| 32 | 0.713 | 4.30% | 3.9 MB |
| 128 | 0.756 | 3.82% | 15.7 MB |

Real clips: 0.789 similarity, ~3% WER. All ranks run at the same speed (decoder 8–13 ms); only voice-file size changes. Rank 128 is the default for best quality.

## Aligner durations

For best results, use the aligner's tokens and durations (`ribt-align durs` output) via `--phon` and `--durs`. Without them, durations come from DTW against the model's own mel, which can misalign (e.g., "derived" → "arrived").

## Blending voices

```bash
# Interpolate two voices: (1−t)·A + t·B
ribt-tts voice-blend --a voiceA.voice --b voiceB.voice --t 0.3 --out voiceC.voice
```

See `src/bin/ribt-tts/voice.rs` for full command documentation.

## How it works

A voice is a small file (3 KB to 16 MB) that holds:

1. **Two speaker vectors** (640 f32): `voice.se` after the text encoder, `voice.sd` at the decoder input
2. **Optional low-rank decoder adapters**: rank-R updates on each decoder block's two pointwise layers
3. **Metadata**: name, pitch, rate, accent, source, flow settings (nfe, temp)

The weights stay shared; every voice runs the same acoustic/flow/vocoder graphs. The adapter is folded into the weights at load, so synthesis runs the same graphs at the same cost.

## The design

The voice ladder reflects the tradeoff between quality and file size:

1. **Zero-shot**: The reference encoder's two speaker vectors, saved as a voice file. No training needed, but quality is baseline.
2. **Embedding fit**: Train just the 640 speaker vectors on the target speaker's clips. Quality improves significantly with 10-50 clips.
3. **Adapter fit**: Also train a low-rank update on the decoder's pointwise layers. This is the main quality knob—rank 128 beats the real-clip similarity on all four test speakers.

The adapter is stored in the voice file and folded into the weights at load, so synthesis runs the same graphs at the same cost. Rank is the main quality knob, and it costs voice-file size, not speed.

## Example voices

I've fitted voices for four speakers (7127, 4446, 1995, 5142) from the LibriTTS test-clean set. The rank-128 voices beat their real-clip similarity on all four:

| speaker | rank 128 similarity | real clip similarity |
|---|---|---|
| 7127 (male) | 0.756 | 0.789 |
| 4446 (female) | 0.801 | 0.744 |
| 1995 (female) | 0.719 | 0.701 |
| 5142 (male) | 0.769 | 0.715 |

All voices are in `target/voice-spike/voices/` in the repo.

## Next steps

- **Multi-voice base model**: Train a base model on multiple speakers, then fit adapters on top. This is the long-term plan for a production multi-voice system.
- **More voices**: Fit voices for more speakers, including children and non-native English speakers.
- **Real-time synthesis**: The current implementation is offline. Real-time synthesis is possible but needs more work.

## Feedback

This is a spike. The API and interface will change as we learn what works. If you try it, let me know what you think.
