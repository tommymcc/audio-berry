//! Decides whether a take passes, since the model varies between seeds.

use crate::quality::{self, Quality};
use crate::store::NarrationStore;
use crate::accuracy::{self, Accuracy, Allowances};
use any_tts::AudioSamples;

/// What a take must meet in one output directory, in one voice.
pub struct Standard {
    speed: f64,
    noise_limit_db: f32,
    allowances: Allowances,
}

/// How a take measured against the standard.
pub struct Verdict {
    pub quality: Quality,
    pub accuracy: Accuracy,
    plausible: bool,
    clean: bool,
}

impl Standard {
    /// The standard of an output directory, from its speed and noise limit, and of
    /// its voice, from the words that voice may be heard saying as others.
    pub fn of(store: &NarrationStore, allowances: Allowances) -> Standard {
        Standard { speed: store.speed(), noise_limit_db: store.noise_limit_db(), allowances }
    }

    /// Measures a take of `text`, given what was heard in it.
    pub fn judge(&self, text: &str, audio: &AudioSamples, transcript: &str) -> Verdict {
        let quality = quality::measure(&audio.samples, audio.sample_rate);
        let characters = text.chars().count();
        Verdict {
            plausible: quality::plausible_length_at(characters, quality.seconds, self.speed),
            clean: quality.clean(self.noise_limit_db),
            accuracy: accuracy::compare(text, transcript, &self.allowances),
            quality,
        }
    }

    /// Whether the words of `phrase`, from the book, were heard in a row in
    /// `transcript`, counting the voice's allowed mishearings as heard.
    pub fn heard(&self, phrase: &str, transcript: &str) -> bool {
        accuracy::heard_phrase(transcript, phrase, &self.allowances)
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

    fn standard(speed: f64) -> Standard {
        Standard { speed, noise_limit_db: -55.0, allowances: Allowances::default() }
    }

    const TEXT: &str = "The quick brown fox jumps over the lazy dog, again and again and again.";

    #[test]
    fn a_take_of_the_right_length_heard_right_passes() {
        let standard = standard(1.0);
        let verdict = standard.judge(TEXT, &silence(4.0), TEXT);
        assert!(verdict.passes());
        assert_eq!(verdict.rank(), (true, true, 1.0));
    }

    #[test]
    fn a_take_far_too_long_for_its_speed_fails() {
        // About 70 characters plausibly takes 2.3 to 11.8 seconds at normal speed.
        let normal = standard(1.0);
        assert!(!normal.judge(TEXT, &silence(20.0), TEXT).passes());
        let slow = standard(0.5);
        assert!(slow.judge(TEXT, &silence(20.0), TEXT).passes());
    }

    #[test]
    fn a_take_heard_wrong_fails_but_ranks_by_accuracy() {
        let standard = standard(1.0);
        let wrong = standard.judge(TEXT, &silence(4.0), "The quick brown cat.");
        assert!(!wrong.passes());
        assert!(wrong.rank() < standard.judge(TEXT, &silence(4.0), TEXT).rank());
    }

    #[test]
    fn a_take_with_mishearings_the_voice_allows_passes() {
        let text = "In the morning, in the boat, in silence, he sat in the stern in the rain.";
        let heard = "And the morning, and the boat, and silence, he sat and the stern and the rain.";
        assert!(!standard(1.0).judge(text, &silence(4.0), heard).passes());
        let allowances = Allowances::parse("in = and").unwrap();
        let accent = Standard { allowances, ..standard(1.0) };
        let verdict = accent.judge(text, &silence(4.0), heard);
        assert!(verdict.passes());
        assert_eq!(verdict.accuracy.recall, 1.0);
    }

    #[test]
    fn a_phrase_is_heard_through_the_voice_s_allowances() {
        let heard = "He sat, and the boat, all night.";
        assert!(!standard(1.0).heard("in the boat", heard));
        let accent = Standard { allowances: Allowances::parse("in = and").unwrap(), ..standard(1.0) };
        assert!(accent.heard("in the boat", heard));
        assert!(!accent.heard("in the ship", heard));
    }
}
