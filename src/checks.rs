//! Decides whether a take passes, since the model varies between seeds.

use crate::quality::{self, Quality};
use crate::store::NarrationStore;
use crate::accuracy::{self, Accuracy, Allowances};
use crate::takes::Judged;
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

    /// Whether the take passes and every word was heard, with nothing more.
    pub fn flawless(&self) -> bool {
        self.plausible && self.clean && self.accuracy.exact()
    }

    /// How the take ranks against others of the same chunk. `heard` is whether
    /// the phrases a `retake` asks for were heard, and true when none were.
    pub fn rank(&self, heard: bool) -> Rank {
        Rank { plausible: self.plausible, clean: self.clean, heard, score: self.accuracy.score() }
    }

    /// How `book` compares the take with others: it passes if it `passes`.
    pub fn judged(&self) -> Judged<Rank> {
        Judged { passes: self.passes(), rank: self.rank(true) }
    }

    /// How `retake` compares the take with others, given whether its phrases
    /// were heard. It passes if it's a plausible length, clean and the phrases
    /// were heard, which its rank already puts first, so between new takes and
    /// the current one the rank alone decides, as it always has; accuracy only
    /// ranks.
    pub fn judged_for_retake(&self, heard: bool) -> Judged<Rank> {
        Judged { passes: self.plausible && self.clean && heard, rank: self.rank(heard) }
    }
}

/// Decides which of a chunk's takes is better, which compares greater, after
/// `Judged` puts passing takes first. A later score belongs here as a field.
///
/// The fields compare in order: plausible length matters most, then clean audio,
/// then hearing a `retake`'s phrases, then accuracy, as recall minus extra words,
/// so that among takes that pass, the one heard most nearly as written wins.
/// Ties keep the earlier take.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Rank {
    plausible: bool,
    clean: bool,
    heard: bool,
    score: f32,
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
        let Rank { plausible, clean, heard, score, .. } = verdict.rank(true);
        assert_eq!((plausible, clean, heard, score), (true, true, true, 1.0));
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
        assert!(wrong.rank(true) < standard.judge(TEXT, &silence(4.0), TEXT).rank(true));
    }

    /// A take of a plausible length, heard with this recall and extra words.
    fn verdict(clean: bool, recall: f32, extra: f32) -> Verdict {
        let quality = Quality { floor_db: -60.0, speech_db: -20.0, peak_db: -2.0, seconds: 8.0 };
        let accuracy = Accuracy { recall, extra, transcript: String::new() };
        Verdict { quality, accuracy, plausible: true, clean }
    }

    #[test]
    fn among_passing_takes_the_most_accurate_ranks_higher() {
        let takes = [verdict(true, 0.95, 0.05), verdict(true, 1.0, 0.0), verdict(true, 0.97, 0.0)];
        let ranked: Vec<Rank> = takes.iter().map(|verdict| verdict.rank(true)).collect();
        assert!(ranked[1] > ranked[0], "higher recall and fewer extra words");
        assert!(ranked[2] > ranked[0], "fewer extra words");
        assert!(ranked[1] > ranked[2], "higher recall");
    }

    #[test]
    fn a_book_take_that_passes_beats_a_more_accurate_one_that_fails() {
        let passing = verdict(true, 0.92, 0.06);
        let noisy = verdict(false, 1.0, 0.0);
        let mut overlong = verdict(true, 1.0, 0.0);
        overlong.plausible = false;
        let wordy = verdict(true, 1.0, 0.13);
        assert!(passing.passes() && !wordy.passes());
        assert!(wordy.rank(true) > passing.rank(true), "by rank alone the wordy take would win");
        for failing in [noisy, overlong, wordy] {
            assert!(passing.judged() > failing.judged());
        }
    }

    #[test]
    fn a_retake_judges_by_rank_alone() {
        let acceptable = verdict(true, 0.92, 0.06);
        let unacceptable = verdict(true, 0.91, 0.0);
        assert!(unacceptable.judged_for_retake(true) > acceptable.judged_for_retake(true));
        assert!(acceptable.judged_for_retake(true) > unacceptable.judged_for_retake(false));
    }

    #[test]
    fn a_retake_phrase_heard_ranks_above_accuracy() {
        let heard = verdict(true, 0.95, 0.05);
        let missed = verdict(true, 1.0, 0.0);
        assert!(heard.rank(true) > missed.rank(false));
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
