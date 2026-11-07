//! How long any-tts expects OmniVoice to take to say a text, which decides how
//! much audio it generates and whether it splits the request. This mirrors
//! `estimate_target_tokens_with_reference`, `text_weight`, `char_weight` and
//! `add_final_punctuation` in any-tts's `src/models/omnivoice/model.rs`, and
//! how its `generate_token_chunks` decides to split, at the revision Cargo.toml
//! pins.

use unicode_general_category::{GeneralCategory, get_general_category};

/// any-tts reads the estimated seconds above which it splits a request from this.
const THRESHOLD_VAR: &str = "OMNIVOICE_AUDIO_CHUNK_THRESHOLD";
/// any-tts's `DEFAULT_AUDIO_CHUNK_THRESHOLD`, used when `THRESHOLD_VAR` is unset
/// or not a number.
const DEFAULT_THRESHOLD_SECONDS: f32 = 30.0;
/// any-tts reads the length of the pieces it splits a request into from this,
/// and doesn't split when it's 0 or less.
const DURATION_VAR: &str = "OMNIVOICE_AUDIO_CHUNK_DURATION";
/// any-tts's `DEFAULT_AUDIO_CHUNK_DURATION`.
const DEFAULT_DURATION_SECONDS: f32 = 15.0;
/// OmniVoice's audio tokens per second: 24 kHz audio with a hop of 960 samples.
const FRAME_RATE: f32 = 25.0;

/// Endings any-tts leaves a voice sample's transcript with. It adds a full stop
/// to a transcript ending in anything else. Each weighs 0.5, as punctuation.
const TRANSCRIPT_ENDINGS: &str = "!\"'),.:;?]}‘’“”………、。】！），：；？";

/// The estimated seconds above which any-tts splits a request into pieces it
/// generates separately and cross-fades, with 0.1 s of silence at each join, as
/// the environment sets it. `None` when any-tts doesn't split.
pub fn split_above_seconds() -> Option<f64> {
    split_above(std::env::var(THRESHOLD_VAR).ok().as_deref(), std::env::var(DURATION_VAR).ok().as_deref())
}

/// The threshold any-tts settles on from the values of `THRESHOLD_VAR` and
/// `DURATION_VAR`, if set. A value that isn't a number gives the default.
fn split_above(threshold: Option<&str>, duration: Option<&str>) -> Option<f64> {
    let read = |value: Option<&str>, default| value.and_then(|value| value.parse::<f32>().ok()).unwrap_or(default);
    let threshold = read(threshold, DEFAULT_THRESHOLD_SECONDS);
    let duration = read(duration, DEFAULT_DURATION_SECONDS);
    let frames = (threshold.max(0.0) * FRAME_RATE).round();
    (duration > 0.0 && frames > 0.0).then_some(f64::from(threshold))
}

/// Warns, naming `what`, if `text` at `pace` is estimated long enough that
/// any-tts will split it.
pub fn warn_if_split(pace: &Pace, text: &str, what: &str) {
    let Some(limit) = split_above_seconds() else { return };
    let seconds = pace.seconds(text);
    if seconds > limit {
        eprintln!(
            "Warning: {what} is estimated at {seconds:.1}s, over the {limit}s above which any-tts splits it \
             into pieces joined by short silences."
        );
    }
}

/// A voice's estimated speaking time per unit of text weight, at one speed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pace {
    seconds_per_weight: f64,
}

impl Pace {
    /// The pace any-tts estimates from a voice sample of `sample_seconds` saying
    /// `transcript`, at `speed`. any-tts measures the sample after removing
    /// silences over 200 ms, which leaves it no longer than the whole sample this
    /// is given, so this pace is never faster than any-tts's.
    pub fn of_voice(sample_seconds: f64, transcript: &str, speed: f64) -> Pace {
        let transcript = transcript.trim();
        let mut weight = text_weight(transcript);
        if !transcript.chars().last().is_some_and(|last| TRANSCRIPT_ENDINGS.contains(last)) {
            weight += char_weight('.');
        }
        // At least 0.5: the transcript ends in one of TRANSCRIPT_ENDINGS or gains
        // a full stop.
        Pace { seconds_per_weight: sample_seconds / weight / speed }
    }

    /// The estimated seconds of audio for `text`. any-tts lengthens estimates
    /// under 2 s, which this leaves out.
    pub fn seconds(&self, text: &str) -> f64 {
        text_weight(text) * self.seconds_per_weight
    }

    /// The text weight estimated to take `seconds`.
    pub fn weight_for(&self, seconds: f64) -> f64 {
        seconds / self.seconds_per_weight
    }
}

/// The estimated speaking time of `text`, where a Latin letter weighs 1.
pub fn text_weight(text: &str) -> f64 {
    text.chars().map(char_weight).sum()
}

/// Relative speaking time of a character, where a Latin letter is 1.0.
pub fn char_weight(ch: char) -> f64 {
    const CJK: f64 = 3.0;
    const HANGUL: f64 = 2.5;
    const KANA: f64 = 2.2;
    const SYLLABIC: f64 = 3.0;
    const INDIC: f64 = 1.8;
    const THAI_LAO: f64 = 1.5;
    const KHMER_MYANMAR: f64 = 1.8;
    const ABJAD: f64 = 1.5;
    const ALPHABET: f64 = 1.0;
    const PUNCTUATION: f64 = 0.5;
    const SPACE: f64 = 0.2;
    const DIGIT: f64 = 3.5;
    const MARK: f64 = 0.0;
    // Each entry covers code points up to its limit, above the previous entry.
    const RANGES: &[(u32, f64)] = &[
        (0x058F, ALPHABET),
        (0x08FF, ABJAD),
        (0x0DFF, INDIC),
        (0x0EFF, THAI_LAO),
        (0x0FFF, INDIC),
        (0x109F, KHMER_MYANMAR),
        (0x10FF, ALPHABET),
        (0x11FF, HANGUL),
        (0x139F, SYLLABIC),
        (0x177F, ALPHABET),
        (0x17FF, KHMER_MYANMAR),
        (0x18FF, ALPHABET),
        (0x19DF, INDIC),
        (0x19FF, KHMER_MYANMAR),
        (0x1C7F, INDIC),
        (0x1CBF, ALPHABET),
        (0x1CFF, INDIC),
        (0x1EFF, ALPHABET),
        (0x30FF, KANA),
        (0x312F, CJK),
        (0x318F, HANGUL),
        (0x9FFF, CJK),
        (0xA4CF, SYLLABIC),
        (0xA7FF, ALPHABET),
        (0xA82F, INDIC),
        (0xA87F, ALPHABET),
        (0xA95F, INDIC),
        (0xA97F, HANGUL),
        (0xA9DF, INDIC),
        (0xA9FF, KHMER_MYANMAR),
        (0xAA5F, INDIC),
        (0xAA7F, KHMER_MYANMAR),
        (0xAAFF, INDIC),
        (0xAB2F, SYLLABIC),
        (0xABBF, ALPHABET),
        (0xABFF, INDIC),
        (0xD7AF, HANGUL),
        (0xFAFF, CJK),
        (0xFDFF, ABJAD),
        (0xFE6F, ALPHABET),
        (0xFEFF, ABJAD),
        (0xFFEF, ALPHABET),
    ];
    use GeneralCategory as Category;
    let code = ch as u32;
    if ch.is_ascii_alphabetic() {
        return ALPHABET;
    }
    if code == 0x20 {
        return SPACE;
    }
    if code == 0x0640 {
        return MARK;
    }
    match get_general_category(ch) {
        Category::NonspacingMark | Category::SpacingMark | Category::EnclosingMark => MARK,
        Category::ConnectorPunctuation
        | Category::DashPunctuation
        | Category::OpenPunctuation
        | Category::ClosePunctuation
        | Category::InitialPunctuation
        | Category::FinalPunctuation
        | Category::OtherPunctuation
        | Category::MathSymbol
        | Category::CurrencySymbol
        | Category::ModifierSymbol
        | Category::OtherSymbol => PUNCTUATION,
        Category::SpaceSeparator | Category::LineSeparator | Category::ParagraphSeparator => SPACE,
        Category::DecimalNumber | Category::LetterNumber | Category::OtherNumber => DIGIT,
        _ => {
            let index = RANGES.partition_point(|(limit, _)| *limit < code);
            match RANGES.get(index) {
                Some((_, weight)) => *weight,
                None if code > 0x20000 => CJK,
                None => ALPHABET,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-9, "{actual} isn't {expected}");
    }

    #[test]
    fn weights_follow_any_tts() {
        for (text, weight) in [("Ab", 2.0), ("a b", 2.2), ("a, b.", 3.2), ("7", 3.5), ("é", 1.0), ("“’", 1.0), ("中", 3.0)] {
            assert_close(text_weight(text), weight);
        }
    }

    #[test]
    fn a_transcript_without_a_final_stop_is_weighed_with_one() {
        // "ab" weighs 2, and 2.5 with the full stop any-tts adds.
        assert_close(Pace::of_voice(5.0, "ab", 1.0).seconds("ab"), 4.0);
        assert_close(Pace::of_voice(5.0, " ab. ", 1.0).seconds("ab."), 5.0);
    }

    #[test]
    fn transcript_endings_weigh_half_a_letter() {
        assert!(TRANSCRIPT_ENDINGS.chars().all(|ending| char_weight(ending) == 0.5));
    }

    #[test]
    fn estimates_match_any_tts() {
        // From any-tts's length_estimate_matches_the_python_package, which uses
        // its default reference of "Nice to meet you." in 25 tokens. Estimates
        // under 50 tokens, which any-tts lengthens, are left out.
        let pace = Pace::of_voice(25.0 / f64::from(FRAME_RATE), "Nice to meet you.", 1.0);
        for (text, tokens) in [("Hello world, this is a test of the estimate.", 64), ("Ünïcode — “quoted” 1865!", 52)] {
            assert_eq!((pace.seconds(text) * f64::from(FRAME_RATE)) as usize, tokens, "{text}");
        }
    }

    #[test]
    fn the_split_threshold_is_read_as_any_tts_reads_it() {
        assert_eq!(split_above(None, None), Some(30.0));
        assert_eq!(split_above(Some("24.5"), None), Some(24.5));
        assert_eq!(split_above(Some("long"), Some("short")), Some(30.0));
        assert_eq!(split_above(Some("0"), None), None);
        assert_eq!(split_above(Some("0.01"), None), None, "under half a frame");
        assert_eq!(split_above(Some("-3"), None), None);
        assert_eq!(split_above(None, Some("0")), None);
    }

    #[test]
    fn speed_divides_the_estimate() {
        let normal = Pace::of_voice(10.0, "Some words.", 1.0);
        let fast = Pace::of_voice(10.0, "Some words.", 2.0);
        assert_close(fast.seconds("Other words here."), normal.seconds("Other words here.") / 2.0);
        assert_close(fast.weight_for(1.0), normal.weight_for(2.0));
    }
}
