//! Barge-in (the ribt paper): stop speaking mid-word, without a
//! click, and know exactly what was said.
//!
//! The synthesis engine renders ~0.5 s chunks ahead of playback, so a cancel
//! that waited for the chunk boundary would be far too slow. The cut is made
//! on the *playout* side instead: [`Playout`] hands the audio device fixed
//! blocks (default 10 ms); a cancel takes effect at the next block, which is a
//! raised-cosine fade, and then the stream is silent and the engine stops
//! rendering. Worst-case audio after the signal = the block already in the
//! device + the fade (10 + 10 ms by default).
//!
//! [`spoken`] maps the last audible sample to the token and word being said,
//! for the dialogue manager, and [`resume_word`] picks where to restart.

/// What the device has been given and what is still queued.
#[derive(Clone, Debug)]
pub struct Playout {
    pub buf: Vec<f32>,
    /// samples handed to the device so far
    pub played: usize,
    pub block: usize,
    pub fade: usize,
    cut: Option<usize>,
}

/// Where a cancel cut the utterance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cut {
    /// sample at which the signal arrived (= samples handed to the device)
    pub signal: usize,
    /// one past the last audible (non-zero-gain) sample
    pub end: usize,
}

impl Playout {
    pub fn new(block: usize, fade: usize) -> Self {
        Self {
            buf: Vec::new(),
            played: 0,
            block,
            fade,
            cut: None,
        }
    }
    pub fn push(&mut self, x: &[f32]) {
        if self.cut.is_none() {
            self.buf.extend_from_slice(x);
        }
    }
    pub fn cancelled(&self) -> bool {
        self.cut.is_some()
    }
    /// The next device block: `block` samples, or fewer at the end of what has
    /// been rendered (None = nothing to give: underrun, or done).
    pub fn next_block(&mut self) -> Option<Vec<f32>> {
        let lim = self.cut.unwrap_or(usize::MAX).min(self.buf.len());
        if self.played >= lim {
            return None;
        }
        let a = self.played;
        let b = (a + self.block).min(lim);
        let mut y = self.buf[a..b].to_vec();
        if let Some(end) = self.cut {
            let f0 = end - self.fade;
            for (i, v) in y.iter_mut().enumerate() {
                let s = a + i;
                if s >= f0 {
                    let x = (s - f0) as f32 + 0.5;
                    *v *= 0.5 * (1.0 + (std::f32::consts::PI * x / self.fade as f32).cos());
                }
            }
        }
        self.played = b;
        Some(y)
    }
    /// Barge-in: everything already handed to the device plays; the next
    /// `fade` samples fade out; nothing after that is ever played. If fewer
    /// than `fade` samples are rendered, the fade is padded with what exists.
    pub fn cancel(&mut self) -> Cut {
        if let Some(end) = self.cut {
            return Cut {
                signal: self.played,
                end,
            };
        }
        let end = (self.played + self.fade).min(self.buf.len().max(self.played));
        self.fade = end - self.played;
        self.buf.truncate(end);
        self.cut = Some(end);
        Cut {
            signal: self.played,
            end,
        }
    }
}

/// What had been said when the audio stopped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spoken {
    /// frame of the last audible sample
    pub frame: usize,
    /// token being rendered at that frame (index into durs)
    pub token: usize,
    /// lilt word containing that token (word_end.len() = past the last word)
    pub word: usize,
    /// fraction of that word's frames that were heard
    pub word_frac: f32,
    /// words fully heard
    pub words_done: usize,
}

/// Map a sample position to the token and word being spoken. `durs` are the
/// committed token durations (frames), `word_end[w]` the token end index of
/// lilt word w (exclusive), `hop` samples per frame.
pub fn spoken(durs: &[u32], word_end: &[usize], hop: usize, sample: usize) -> Spoken {
    let frame = sample / hop;
    let mut starts = Vec::with_capacity(durs.len() + 1);
    let mut pos = 0usize;
    for &d in durs {
        starts.push(pos);
        pos += d as usize;
    }
    starts.push(pos);
    // last token whose start ≤ frame (and that has frames)
    let token = (0..durs.len())
        .rev()
        .find(|&i| starts[i] <= frame && durs[i] > 0)
        .unwrap_or(0);
    let word = word_end
        .iter()
        .position(|&e| e > token)
        .unwrap_or(word_end.len());
    let (wa, wb) = if word < word_end.len() {
        let a = if word == 0 { 1 } else { word_end[word - 1] };
        (
            starts[a.min(durs.len())],
            starts[word_end[word].min(durs.len())],
        )
    } else {
        (pos, pos)
    };
    let word_frac = if wb > wa {
        ((frame.saturating_sub(wa)) as f32 / (wb - wa) as f32).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let words_done = if word_frac >= 1.0 { word + 1 } else { word }.min(word_end.len());
    Spoken {
        frame,
        token,
        word,
        word_frac,
        words_done,
    }
}

/// Word to restart from: the interrupted word if less than `keep` of it was
/// heard (say it again), else the next one.
pub fn resume_word(s: &Spoken, keep: f32) -> usize {
    if s.word_frac >= keep {
        s.words_done.max(s.word + 1)
    } else {
        s.word
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_within_block_plus_fade() {
        let mut p = Playout::new(240, 240);
        p.push(&vec![1.0; 24000]);
        for _ in 0..5 {
            p.next_block();
        }
        let c = p.cancel();
        assert_eq!(c.signal, 1200);
        assert_eq!(c.end, 1440);
        let f = p.next_block().unwrap();
        assert_eq!(f.len(), 240);
        assert!(f[0] > 0.99 && f[239] < 1e-3, "{} {}", f[0], f[239]);
        // monotone fade, no step
        assert!(f.windows(2).all(|w| w[1] <= w[0] + 1e-6));
        assert!(p.next_block().is_none());
        p.push(&[1.0; 100]);
        assert!(p.next_block().is_none(), "nothing plays after a cancel");
    }

    #[test]
    fn cancel_near_end_of_rendered_audio() {
        let mut p = Playout::new(240, 240);
        p.push(&vec![1.0; 300]);
        p.next_block();
        let c = p.cancel();
        assert_eq!((c.signal, c.end), (240, 300));
        let f = p.next_block().unwrap();
        assert_eq!(f.len(), 60);
        assert!(f[59] < 1e-2);
    }

    #[test]
    fn spoken_maps_words() {
        // toks: [0] w0 = toks 1..3 (+space 3), w1 = 4..6, [0]
        let durs = [5, 4, 4, 2, 3, 3, 1, 5];
        let word_end = [4, 7];
        let hop = 10;
        let s = spoken(&durs, &word_end, hop, 0);
        assert_eq!((s.token, s.word, s.words_done), (0, 0, 0));
        let s = spoken(&durs, &word_end, hop, 10 * 10); // frame 10: token 2, mid word 0
        assert_eq!((s.token, s.word), (2, 0));
        assert!(s.word_frac > 0.4 && s.word_frac < 0.6, "{}", s.word_frac);
        assert_eq!(resume_word(&s, 0.8), 0);
        let s = spoken(&durs, &word_end, hop, 16 * 10); // frame 16: token 4 (word 1 starts at frame 15)
        assert_eq!((s.token, s.word, s.words_done), (4, 1, 1));
        assert_eq!(resume_word(&s, 0.8), 1);
        let s = spoken(&durs, &word_end, hop, 23 * 10); // final pad
        assert_eq!(s.word, 2);
        assert_eq!(s.words_done, 2);
    }
}
