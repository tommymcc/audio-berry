//! Decides whether a take passes, since the model varies between seeds.

use crate::quality::{self, Quality};
use crate::store::NarrationStore;
use crate::accuracy::{self, Accuracy};
use any_tts::AudioSamples;

/// What a take must meet in one output directory.
pub struct Standard {
    speed: f64,
    noise_limit_db: f32,
}

/// How a take measured against the standard.
pub struct Verdict {
    pub quality: Quality,
    pub accuracy: Accuracy,
    plausible: bool,
    clean: bool,
}

impl Standard {
    /// The standard of an output directory, from its speed and noise limit.
    pub fn of(store: &NarrationStore) -> Standard {
        Standard { speed: store.speed(), noise_limit_db: store.noise_limit_db() }
    }

    /// Measures a take of `text`, given what was heard in it.
    pub fn judge(&self, text: &str, audio: &AudioSamples, transcript: &str) -> Verdict {
        let quality = quality::measure(&audio.samples, audio.sample_rate);
        let characters = text.chars().count();
        Verdict {
            plausible: quality::plausible_length_at(characters, quality.seconds, self.speed),
            clean: quality.clean(self.noise_limit_db),
            accuracy: accuracy::compare(text, transcript),
            quality,
        }
    }
}

impl Verdict {
    /// Whether the take is good enough to keep without trying another.
    pub fn passes(&self) -> bool {
        self.plausible && self.clean && self.accuracy.acceptable()
    }

    /// Orders takes: plausible length matters most, then clean audio, then accuracy.
    pub fn rank(&self) -> (bool, bool, f32) {
        (self.plausible, self.clean, self.accuracy.score())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(seconds: f32) -> AudioSamples {
        AudioSamples::new(vec![0.0; (seconds * 24_000.0) as usize], 24_000)
    }

    const TEXT: &str = "The quick brown fox jumps over the lazy dog, again and again and again.";

    #[test]
    fn a_take_of_the_right_length_heard_right_passes() {
        let standard = Standard { speed: 1.0, noise_limit_db: -55.0 };
        let verdict = standard.judge(TEXT, &silence(4.0), TEXT);
        assert!(verdict.passes());
        assert_eq!(verdict.rank(), (true, true, 1.0));
    }

    #[test]
    fn a_take_far_too_long_for_its_speed_fails() {
        // About 70 characters plausibly takes 2.3 to 11.8 seconds at normal speed.
        let normal = Standard { speed: 1.0, noise_limit_db: -55.0 };
        assert!(!normal.judge(TEXT, &silence(20.0), TEXT).passes());
        let slow = Standard { speed: 0.5, noise_limit_db: -55.0 };
        assert!(slow.judge(TEXT, &silence(20.0), TEXT).passes());
    }

    #[test]
    fn a_take_heard_wrong_fails_but_ranks_by_accuracy() {
        let standard = Standard { speed: 1.0, noise_limit_db: -55.0 };
        let wrong = standard.judge(TEXT, &silence(4.0), "The quick brown cat.");
        assert!(!wrong.passes());
        assert!(wrong.rank() < standard.judge(TEXT, &silence(4.0), TEXT).rank());
    }
}
