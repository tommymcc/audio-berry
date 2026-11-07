//! Decides whether a take is good enough for its goal, and which of a chunk's
//! takes is best, since the model varies between seeds.

use crate::pitch;
use crate::quality::{self, Quality};
use crate::store::KeepRules;
use crate::accuracy::{self, Accuracy, Allowances};
use crate::takes::Judged;
use any_tts::AudioSamples;

/// What a take must meet in one output directory, in one voice.
pub struct Standard {
    speed: f64,
    noise_limit_db: f32,
    /// The pitch range wanted, in semitones, if any; see `Verdict::rank`.
    pitch_range_target: Option<f32>,
    allowances: Allowances,
}

/// What a chunk's takes are rendered for: the take policy's goal. It decides how
/// many takes the default policy tries, how they compare, and which is good
/// enough to stop at.
#[derive(Clone, Copy, Debug)]
pub enum Goal<'a> {
    /// `book` narrating a chunk: a take is good enough when it passes.
    Book,
    /// A `retake` of a chunk, asking for these phrases from the book to be heard,
    /// or none: a take is good enough when it's flawless with its phrases heard.
    Retake { phrases: &'a [String] },
}

impl Goal<'_> {
    /// The most takes the default policy renders, stopping sooner at one that's
    /// good enough; if none is, it keeps the best of them. A retake tries one
    /// more new take than `book` does.
    pub fn most_takes(&self) -> u64 {
        match self {
            Goal::Book => 5,
            Goal::Retake { .. } => 6,
        }
    }
}

/// How a take measured against the standard, for one goal.
pub struct Verdict {
    pub quality: Quality,
    pub accuracy: Accuracy,
    /// The take's pitch range in semitones, measured only when there's a
    /// target, and `None` too when too little of it is voiced to measure.
    pub pitch_range: Option<f32>,
    /// How near the pitch range is to the target, from `pitch::nearness`, or
    /// 0 with no target or no range.
    nearness: f32,
    plausible: bool,
    clean: bool,
    aim: Aim,
}

/// The goal a verdict was judged for, without its phrases, which it no longer
/// needs once they're heard or not. It's more than whether they were heard: a
/// retake asking for no phrases is still judged as a retake, not as `book`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Aim {
    Book,
    /// Whether the retake's phrases were all heard, true when there were none.
    Retake { heard: bool },
}

impl Standard {
    /// The standard of an output directory, from the noise limit and pitch range
    /// target in its `rules` and the `speed` it narrates at, and of its voice, from
    /// the words that voice may be heard saying as others.
    pub fn of(rules: &KeepRules, speed: f64, allowances: Allowances) -> Standard {
        let KeepRules { noise_limit_db, pitch_range_target, .. } = *rules;
        Standard { speed, noise_limit_db, pitch_range_target, allowances }
    }

    /// The pitch range target takes are ranked by nearness to, if any.
    pub fn pitch_range_target(&self) -> Option<f32> {
        self.pitch_range_target
    }

    /// Measures a take of `text` for `goal`, given what was heard in it.
    pub fn judge(&self, goal: &Goal, text: &str, audio: &AudioSamples, transcript: &str) -> Verdict {
        let quality = quality::measure(&audio.samples, audio.sample_rate);
        let characters = text.chars().count();
        let target = self.pitch_range_target;
        let pitch_range = target.and_then(|_| pitch::range_semitones(&audio.samples, audio.sample_rate));
        let nearness = match (pitch_range, target) {
            (Some(range), Some(target)) => pitch::nearness(range, target),
            _ => 0.0,
        };
        let aim = match goal {
            Goal::Book => Aim::Book,
            Goal::Retake { phrases } => {
                Aim::Retake { heard: phrases.iter().all(|phrase| self.heard(phrase, transcript)) }
            }
        };
        Verdict {
            pitch_range,
            nearness,
            plausible: quality::plausible_length_at(characters, quality.seconds, self.speed),
            clean: quality.clean(self.noise_limit_db),
            accuracy: accuracy::compare(text, transcript, &self.allowances),
            quality,
            aim,
        }
    }

    /// Whether the words of `phrase`, from the book, were heard in a row in
    /// `transcript`, counting the voice's allowed mishearings as heard.
    fn heard(&self, phrase: &str, transcript: &str) -> bool {
        accuracy::heard_phrase(transcript, phrase, &self.allowances)
    }
}

impl Verdict {
    /// How the take compares with others of its chunk, for its goal. For `book`
    /// it passes if it's a plausible length, clean and heard well enough. For
    /// `retake` it passes if it's a plausible length, clean and its phrases were
    /// heard, which its rank already puts first, so between new takes and the
    /// current one the rank alone decides, as it always has; accuracy only ranks.
    pub fn judged(&self) -> Judged<Rank> {
        let passes = match self.aim {
            Aim::Book => self.passes(),
            Aim::Retake { heard } => self.plausible && self.clean && heard,
        };
        Judged { passes, rank: self.rank() }
    }

    /// Whether the take is good enough to keep without trying another: for
    /// `book` one that passes, and for `retake` one that's flawless, every word
    /// heard with nothing more, with its phrases heard.
    pub fn good_enough(&self) -> bool {
        match self.aim {
            Aim::Book => self.passes(),
            Aim::Retake { heard } => self.flawless() && heard,
        }
    }

    /// Whether the phrases a `retake` asks for were all heard, and true when
    /// none were asked for, as for `book`.
    pub fn heard(&self) -> bool {
        match self.aim {
            Aim::Book => true,
            Aim::Retake { heard } => heard,
        }
    }

    /// Whether the take is a plausible length, clean and heard well enough.
    fn passes(&self) -> bool {
        self.plausible && self.clean && self.accuracy.acceptable()
    }

    /// Whether the take passes and every word was heard, with nothing more.
    fn flawless(&self) -> bool {
        self.plausible && self.clean && self.accuracy.exact()
    }

    /// How the take ranks against others of the same chunk. Nearness to the
    /// pitch range target counts only for a take that passes with its phrases
    /// heard, so it never lifts one that fails, and failing takes rank as
    /// without a target.
    fn rank(&self) -> Rank {
        let heard = self.heard();
        let nearness = if self.passes() && heard { in_steps(self.nearness) } else { 0.0 };
        Rank { plausible: self.plausible, clean: self.clean, heard, nearness, score: self.accuracy.score() }
    }
}

/// Decides which of a chunk's takes is better, which compares greater, after
/// `Judged` puts passing takes first. A later score belongs here as a field.
///
/// The fields compare in order: plausible length matters most, then clean audio,
/// then hearing a `retake`'s phrases, then nearness to the pitch range target,
/// then accuracy, as recall minus extra words. So with a target, the take that
/// passes nearest it wins, since every take that passes is heard well enough,
/// unless another is about as near; otherwise, among takes that pass, the one
/// heard most nearly as written wins. Ties keep the earlier take.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Rank {
    plausible: bool,
    clean: bool,
    heard: bool,
    /// How near the pitch range is to the target, in `NEARNESS_STEP`s up to 1,
    /// for a take that passes with its phrases heard; 0 for any other, or with
    /// no target.
    nearness: f32,
    score: f32,
}

/// Nearness counts in steps this big, so takes about as near the target rank by
/// accuracy instead: against 12.4 semitones, 12.1 and 12.6 are both in the top
/// step, but 11.0 is in the next. Ranges measure within about 0.01 semitones of
/// librosa's, so a step is far wider than the measuring error; how small a
/// difference can be heard hasn't been tested.
const NEARNESS_STEP: f32 = 0.05;

/// `nearness` rounded up to a whole number of `NEARNESS_STEP`s. Up, so that an
/// exact match isn't a step of its own, and a measured range, however far off,
/// still ranks above none.
fn in_steps(nearness: f32) -> f32 {
    (nearness / NEARNESS_STEP).ceil() * NEARNESS_STEP
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(seconds: f32) -> AudioSamples {
        AudioSamples::new(vec![0.0; (seconds * 24_000.0) as usize], 24_000)
    }

    fn standard(speed: f64) -> Standard {
        Standard { speed, noise_limit_db: -55.0, pitch_range_target: None, allowances: Allowances::default() }
    }

    const TEXT: &str = "The quick brown fox jumps over the lazy dog, again and again and again.";

    #[test]
    fn a_take_of_the_right_length_heard_right_passes() {
        let standard = standard(1.0);
        let verdict = standard.judge(&Goal::Book, TEXT, &silence(4.0), TEXT);
        assert!(verdict.passes());
        let Rank { plausible, clean, heard, nearness, score } = verdict.rank();
        assert_eq!((plausible, clean, heard, nearness, score), (true, true, true, 0.0, 1.0));
        assert_eq!(verdict.pitch_range, None, "measured only with a target");
    }

    #[test]
    fn a_take_far_too_long_for_its_speed_fails() {
        // About 70 characters plausibly takes 2.3 to 11.8 seconds at normal speed.
        let normal = standard(1.0);
        assert!(!normal.judge(&Goal::Book, TEXT, &silence(20.0), TEXT).passes());
        let slow = standard(0.5);
        assert!(slow.judge(&Goal::Book, TEXT, &silence(20.0), TEXT).passes());
    }

    #[test]
    fn a_take_heard_wrong_fails_but_ranks_by_accuracy() {
        let standard = standard(1.0);
        let wrong = standard.judge(&Goal::Book, TEXT, &silence(4.0), "The quick brown cat.");
        assert!(!wrong.passes());
        assert!(wrong.rank() < standard.judge(&Goal::Book, TEXT, &silence(4.0), TEXT).rank());
    }

    /// A take of a plausible length, heard with this recall and extra words.
    fn verdict(clean: bool, recall: f32, extra: f32) -> Verdict {
        let quality = Quality { floor_db: -60.0, speech_db: -20.0, peak_db: -2.0, seconds: 8.0 };
        let accuracy = Accuracy { recall, extra, transcript: String::new() };
        Verdict { quality, accuracy, pitch_range: None, nearness: 0.0, plausible: true, clean, aim: Aim::Book }
    }

    /// As `verdict`, with this pitch range measured against a target of 12.4 semitones.
    fn ranged(clean: bool, recall: f32, extra: f32, range: Option<f32>) -> Verdict {
        let nearness = range.map_or(0.0, |range| pitch::nearness(range, 12.4));
        Verdict { pitch_range: range, nearness, ..verdict(clean, recall, extra) }
    }

    /// `verdict` as judged for a `retake` whose phrases were or weren't heard.
    fn for_retake(verdict: Verdict, heard: bool) -> Verdict {
        Verdict { aim: Aim::Retake { heard }, ..verdict }
    }

    #[test]
    fn with_a_target_the_passing_take_nearest_it_beats_more_accurate_ones() {
        let flat = ranged(true, 1.0, 0.0, Some(7.0));
        let near = ranged(true, 0.95, 0.03, Some(12.0));
        let too_wide = ranged(true, 1.0, 0.0, Some(18.0));
        assert!(near.passes() && flat.passes() && too_wide.passes());
        assert!(near.judged() > flat.judged());
        assert!(near.judged() > too_wide.judged());
        assert!(too_wide.judged() > flat.judged(), "18 is nearer 12.4 than 7 is, by ratio");
    }

    #[test]
    fn with_a_target_an_unmeasured_take_ranks_below_measured_ones() {
        let unmeasured = ranged(true, 1.0, 0.0, None);
        let measured = ranged(true, 0.95, 0.03, Some(6.0));
        assert!(measured.judged() > unmeasured.judged());
    }

    #[test]
    fn nearness_never_lifts_a_failing_take() {
        let passing = ranged(true, 0.92, 0.06, Some(6.0));
        let noisy = ranged(false, 1.0, 0.0, Some(12.4));
        let misheard = ranged(true, 0.8, 0.0, Some(12.4));
        assert!(!noisy.passes() && !misheard.passes());
        assert!(passing.judged() > noisy.judged());
        assert!(passing.judged() > misheard.judged());
        // Failing takes rank by accuracy, as they do without a target.
        let worse = ranged(true, 0.7, 0.0, Some(12.4));
        let better = ranged(true, 0.8, 0.0, Some(3.0));
        assert!(better.judged() > worse.judged());
        assert_eq!(better.rank(), verdict(true, 0.8, 0.0).rank());
    }

    #[test]
    fn a_retake_counts_nearness_only_with_its_phrases_heard() {
        let near = |heard| for_retake(ranged(true, 0.95, 0.03, Some(12.4)), heard);
        let flat = for_retake(ranged(true, 1.0, 0.0, Some(6.0)), true);
        assert!(near(true).judged() > flat.judged());
        assert!(flat.judged() > near(false).judged());
        assert_eq!(near(false).rank(), for_retake(verdict(true, 0.95, 0.03), false).rank());
    }

    #[test]
    fn a_take_is_measured_only_against_a_target() {
        let hum = (0..96_000).map(|i| (i as f32 * std::f32::consts::TAU * 120.0 / 24_000.0).sin() * 0.3).collect();
        let audio = AudioSamples::new(hum, 24_000);
        assert_eq!(standard(1.0).judge(&Goal::Book, TEXT, &audio, TEXT).pitch_range, None);
        let targeted = Standard { pitch_range_target: Some(12.4), ..standard(1.0) };
        let verdict = targeted.judge(&Goal::Book, TEXT, &audio, TEXT);
        let range = verdict.pitch_range.unwrap();
        assert!(range < 0.5, "a steady hum is flat: {range}");
        assert_eq!(verdict.rank().nearness, in_steps(pitch::nearness(range, 12.4)));
    }

    #[test]
    fn takes_about_as_near_the_target_rank_by_accuracy() {
        // 12.1 and 12.6 are both within a step of 12.4.
        let accurate = ranged(true, 1.0, 0.0, Some(12.1));
        let nearer = ranged(true, 0.95, 0.03, Some(12.6));
        assert!(pitch::nearness(12.6, 12.4) > pitch::nearness(12.1, 12.4));
        assert!(accurate.judged() > nearer.judged());
    }

    #[test]
    fn takes_clearly_nearer_the_target_win_over_accuracy() {
        let accurate = ranged(true, 1.0, 0.0, Some(11.0));
        let nearer = ranged(true, 0.95, 0.03, Some(12.2));
        assert!(nearer.judged() > accurate.judged());
    }

    #[test]
    fn nearness_counts_in_steps_rounded_up() {
        assert_eq!(in_steps(1.0), 1.0);
        assert_eq!(in_steps(0.97), 1.0);
        assert!((in_steps(0.92) - 0.95).abs() < 1e-6);
        assert!((in_steps(0.01) - NEARNESS_STEP).abs() < 1e-6, "far off still beats unmeasured");
        assert_eq!(in_steps(0.0), 0.0);
    }

    #[test]
    fn among_passing_takes_the_most_accurate_ranks_higher() {
        let takes = [verdict(true, 0.95, 0.05), verdict(true, 1.0, 0.0), verdict(true, 0.97, 0.0)];
        let ranked: Vec<Rank> = takes.iter().map(|verdict| verdict.rank()).collect();
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
        assert!(wordy.rank() > passing.rank(), "by rank alone the wordy take would win");
        for failing in [noisy, overlong, wordy] {
            assert!(passing.judged() > failing.judged());
        }
    }

    #[test]
    fn a_retake_judges_by_rank_alone() {
        // 30 words, so one missed or extra word is about 0.03.
        let text = format!("{TEXT} {TEXT} at last.");
        let standard = standard(1.0);
        let judge = |phrase: &str, transcript: &str| {
            let phrases = [phrase.to_string()];
            standard.judge(&Goal::Retake { phrases: &phrases }, &text, &silence(5.0), transcript)
        };
        // Two words missed and two added: recall 0.93 and extra 0.07, acceptable.
        let acceptable = text.replacen("quick brown", "slow red", 1);
        // The last three missed: recall 0.90, which isn't, but nearer as written.
        let unacceptable = text.replace(" again. at last.", "");
        assert!(compare(&text, &acceptable).acceptable() && !compare(&text, &unacceptable).acceptable());
        assert!(judge("lazy dog", &unacceptable).judged() > judge("lazy dog", &acceptable).judged());
        assert!(judge("at last", &acceptable).judged() > judge("at last", &unacceptable).judged());
    }

    /// How `transcript` was heard against `text`, allowing nothing.
    fn compare(text: &str, transcript: &str) -> Accuracy {
        accuracy::compare(text, transcript, &Allowances::default())
    }

    #[test]
    fn a_retake_phrase_heard_ranks_above_accuracy() {
        let standard = standard(1.0);
        let phrases = ["lazy dog".to_string()];
        let retake = Goal::Retake { phrases: &phrases };
        let heard = standard.judge(&retake, TEXT, &silence(4.0), "The quick brown fox jumps over the lazy dog.");
        let missed = standard.judge(&retake, TEXT, &silence(4.0), &TEXT.replace("lazy", "hazy"));
        assert!(heard.heard() && !missed.heard());
        assert!(missed.accuracy.score() > heard.accuracy.score());
        assert!(heard.judged() > missed.judged());
    }

    #[test]
    fn a_retake_is_good_enough_only_flawless_with_its_phrases_heard() {
        let standard = standard(1.0);
        let phrases = ["lazy dog".to_string()];
        let retake = Goal::Retake { phrases: &phrases };
        assert!(standard.judge(&retake, TEXT, &silence(4.0), TEXT).good_enough());
        let missed_one = TEXT.replacen("again ", "", 1);
        let passing = standard.judge(&retake, TEXT, &silence(4.0), &missed_one);
        assert!(passing.passes() && !passing.good_enough(), "it passes for book, but isn't flawless");
        assert!(standard.judge(&Goal::Book, TEXT, &silence(4.0), &missed_one).good_enough());
        let other_phrase = ["lazy cat".to_string()];
        let unheard = standard.judge(&Goal::Retake { phrases: &other_phrase }, TEXT, &silence(4.0), TEXT);
        assert!(!unheard.good_enough());
        let no_phrases = standard.judge(&Goal::Retake { phrases: &[] }, TEXT, &silence(4.0), TEXT);
        assert!(no_phrases.heard() && no_phrases.good_enough());
    }

    #[test]
    fn a_retake_tries_one_take_more_than_book() {
        assert_eq!(Goal::Book.most_takes(), 5);
        assert_eq!(Goal::Retake { phrases: &[] }.most_takes(), 6);
    }

    #[test]
    fn a_take_with_mishearings_the_voice_allows_passes() {
        let text = "In the morning, in the boat, in silence, he sat in the stern in the rain.";
        let heard = "And the morning, and the boat, and silence, he sat and the stern and the rain.";
        assert!(!standard(1.0).judge(&Goal::Book, text, &silence(4.0), heard).passes());
        let allowances = Allowances::parse("in = and").unwrap();
        let accent = Standard { allowances, ..standard(1.0) };
        let verdict = accent.judge(&Goal::Book, text, &silence(4.0), heard);
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
