//! Finds pauses in the middle of a phrase, such as "the night in which I was ⏸
//! born", which OmniVoice sometimes makes and which sound wrong even when every
//! word is heard.
//!
//! Silence is found in the audio first, since the checker's word end times run
//! on into the silence after a word and are too coarse to measure a pause by.
//! The checker's word start times only place each silence between two words
//! heard, and the alignment with the chunk's text says whether the text has
//! punctuation there.

use crate::accuracy::{self, Allowances, Pair};
use crate::checker::Word;
use crate::quality;
use any_tts::AudioSamples;
use std::fmt;

/// Silence is measured in frames this long.
const FRAME_SECONDS: f32 = 0.01;
/// A frame is silent when its level is at most this far above the take's
/// silence floor.
const SILENCE_ABOVE_FLOOR_DB: f32 = 10.0;

/// The pause limit unless one is given, in seconds. Listening to 9 pauses at a
/// word boundary with no punctuation, each of 0.64 s or more sounded wrong (6 of
/// 6), and each of 0.51 s or less sounded fine (3 of 3), whatever the words.
pub const DEFAULT_PAUSE_LIMIT_SECONDS: f32 = 0.6;

/// The limits `--pause-limit` can give, in seconds.
pub const PAUSE_LIMITS: std::ops::RangeInclusive<f32> = 0.3..=5.0;

/// The shortest silence in the middle of a phrase that fails a take, or none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PauseLimit {
    /// Takes aren't checked for pauses.
    Off,
    Seconds(f32),
}

impl PauseLimit {
    pub const DEFAULT: PauseLimit = PauseLimit::Seconds(DEFAULT_PAUSE_LIMIT_SECONDS);

    /// The limit in seconds, or `None` when takes aren't checked for pauses.
    pub fn seconds(self) -> Option<f32> {
        match self {
            PauseLimit::Off => None,
            PauseLimit::Seconds(seconds) => Some(seconds),
        }
    }

    /// How `narration.txt` saves it: `off`, or the seconds.
    pub fn saved(self) -> String {
        match self {
            PauseLimit::Off => "off".into(),
            PauseLimit::Seconds(seconds) => seconds.to_string(),
        }
    }
}

impl fmt::Display for PauseLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PauseLimit::Off => write!(f, "off"),
            PauseLimit::Seconds(seconds) => write!(f, "{seconds} s"),
        }
    }
}

/// A pause limit given on the command line.
pub fn parse_pause_limit(value: &str) -> Result<PauseLimit, String> {
    valid_pause_limit(value).ok_or_else(|| format!("must be {}", pause_limits_text()))
}

/// What a pause limit can be, as errors name it.
pub fn pause_limits_text() -> String {
    format!("off or a length from {} to {} seconds", PAUSE_LIMITS.start(), PAUSE_LIMITS.end())
}

/// A pause limit from text, if it's `off` or a length within `PAUSE_LIMITS`.
pub fn valid_pause_limit(value: &str) -> Option<PauseLimit> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("off") {
        return Some(PauseLimit::Off);
    }
    value.parse::<f32>().ok().filter(|seconds| PAUSE_LIMITS.contains(seconds)).map(PauseLimit::Seconds)
}

/// A stretch of silence, in seconds from the start of the take.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Silence {
    start: f32,
    end: f32,
}

impl Silence {
    fn seconds(&self) -> f32 {
        self.end - self.start
    }
}

/// A take's silence floor, in dB: the level of the quietest 5% of its
/// `FRAME_SECONDS` frames, ignoring digital silence. It isn't the floor the
/// noise limit judges, which is over 50 ms frames and counts digital silence:
/// the pause limit was calibrated by ear against silences measured this way,
/// and 10 ms frames find where a silence starts and ends to within 10 ms.
fn silence_floor(levels: &[f32]) -> Option<f32> {
    quality::floor_ignoring_digital_silence(levels)
}

/// Each stretch of `samples` at least `shortest` seconds long whose
/// `FRAME_SECONDS` frames are all within `SILENCE_ABOVE_FLOOR_DB` of the
/// silence floor, in order. Digital silence counts as silent. Audio that's all
/// digital silence has no silence floor, and no silences are found in it.
fn silences(samples: &[f32], sample_rate: u32, shortest: f32) -> Vec<Silence> {
    let size = ((sample_rate as f32 * FRAME_SECONDS) as usize).max(1);
    let levels = quality::levels_of_frames(samples, size);
    let Some(floor) = silence_floor(&levels) else {
        return Vec::new();
    };
    let shortest_frames = (shortest / FRAME_SECONDS).round() as usize;
    let at = |frame: usize| frame as f32 * FRAME_SECONDS;
    let mut found = Vec::new();
    let mut start = None;
    for (frame, level) in levels.iter().enumerate() {
        let silent = quality::to_db(*level) <= floor + SILENCE_ABOVE_FLOOR_DB;
        match (silent, start) {
            (true, None) => start = Some(frame),
            (false, Some(first)) => {
                if frame - first >= shortest_frames {
                    found.push(Silence { start: at(first), end: at(frame) });
                }
                start = None;
            }
            _ => {}
        }
    }
    if let Some(first) = start.filter(|first| levels.len() - first >= shortest_frames) {
        found.push(Silence { start: at(first), end: at(levels.len()) });
    }
    found
}

/// A mid-phrase pause: a silence between two words of the text with no
/// punctuation between them.
#[derive(Clone, Debug, PartialEq)]
pub struct MidPhrasePause {
    /// The word of the text before the pause, as written.
    pub before: String,
    /// The word of the text after it, as written.
    pub after: String,
    pub seconds: f32,
}

impl fmt::Display for MidPhrasePause {
    /// As `was … born 0.72 s`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} … {} {:.2} s", self.before, self.after, self.seconds)
    }
}

/// The mid-phrase pauses joined by ` | `, as `report.tsv` and the take lines
/// show them.
fn list(pauses: &[MidPhrasePause]) -> String {
    pauses.iter().map(MidPhrasePause::to_string).collect::<Vec<_>>().join(" | ")
}

/// What checking a take for mid-phrase pauses found.
#[derive(Clone, Debug, PartialEq)]
pub enum MidPhrasePauses {
    /// The pause limit is off.
    Off,
    /// The take has no word times to place silences by, as a take checked
    /// before they were saved.
    Unmeasured,
    /// Each one found, which may be none.
    Measured(Vec<MidPhrasePause>),
}

impl MidPhrasePauses {
    /// Checks a take of `text` against `limit`, given the words heard in it
    /// with their times, if known. See `mid_phrase`.
    pub fn check(
        limit: PauseLimit,
        text: &str,
        words: Option<&[Word]>,
        audio: &AudioSamples,
        allowances: &Allowances,
    ) -> MidPhrasePauses {
        match (limit.seconds(), words) {
            (None, _) => MidPhrasePauses::Off,
            (Some(_), None) => MidPhrasePauses::Unmeasured,
            (Some(shortest), Some(words)) => {
                MidPhrasePauses::Measured(mid_phrase(text, words, audio, shortest, allowances))
            }
        }
    }

    /// Whether none was found. True when unmeasured or off, so such a take
    /// isn't failed for one.
    pub fn none_found(&self) -> bool {
        match self {
            MidPhrasePauses::Measured(found) => found.is_empty(),
            MidPhrasePauses::Off | MidPhrasePauses::Unmeasured => true,
        }
    }

    /// Those found, joined by ` | `, or nothing.
    pub fn list(&self) -> String {
        match self {
            MidPhrasePauses::Measured(found) => list(found),
            MidPhrasePauses::Off | MidPhrasePauses::Unmeasured => String::new(),
        }
    }

    /// The `pauses` column of `report.tsv`: those found, `unmeasured`, or
    /// nothing with the pause limit off.
    pub fn column(&self) -> String {
        match self {
            MidPhrasePauses::Unmeasured => "unmeasured".into(),
            other => other.list(),
        }
    }
}

/// The silences of at least `shortest` seconds in a take of `text` that fall
/// between two words of the text with no punctuation between them, given the
/// words heard in it with their times.
///
/// Each silence is placed by its middle: after the last word of the text paired
/// with a word heard that starts before then, and before the next word paired.
/// A word heard as starting inside the silence and ending after it is said
/// after it, so the silence is placed before that word instead: the checker's
/// start times can fall a little early. Words of the text missed between the
/// two words either side are part of the boundary, so
/// punctuation next to them explains the pause too. A silence that starts
/// before the first word heard starts, or ends after the last one ends, or with
/// no paired word on one side, is the take's start or end, not a pause: the
/// checker's start times can fall inside the silence before the first word. A
/// boundary with more than one silence is named once, with the longest.
fn mid_phrase(
    text: &str,
    words: &[Word],
    audio: &AudioSamples,
    shortest: f32,
    allowances: &Allowances,
) -> Vec<MidPhrasePause> {
    let (Some(first), Some(last)) = (words.first(), words.iter().map(|word| word.end).reduce(f32::max)) else {
        return Vec::new();
    };
    let silences: Vec<Silence> = silences(&audio.samples, audio.sample_rate, shortest)
        .into_iter()
        .filter(|silence| silence.start > first.start && silence.end < last)
        .collect();
    if silences.is_empty() {
        return Vec::new();
    }
    // The words heard as one transcript, and where each starts in it.
    let mut heard = String::new();
    let mut offsets = Vec::with_capacity(words.len());
    for word in words {
        if !heard.is_empty() {
            heard.push(' ');
        }
        offsets.push(heard.len());
        heard.push_str(&word.text);
    }
    let pairs = accuracy::pairs(text, &heard, allowances);
    // The word heard at a byte of `heard`.
    let word_at = |byte: usize| &words[offsets.partition_point(|&offset| offset <= byte) - 1];
    let said_at = |pair: &Pair| word_at(pair.heard.start).start;
    let follows = |pair: &Pair, silence: &Silence| {
        said_at(pair) > silence.start && word_at(pair.heard.end - 1).end > silence.end
    };
    let mut pauses: Vec<(usize, MidPhrasePause)> = Vec::new();
    for silence in silences {
        let middle = (silence.start + silence.end) / 2.0;
        let mut next = pairs.partition_point(|pair| said_at(pair) < middle);
        while next > 0 && follows(&pairs[next - 1], &silence) {
            next -= 1;
        }
        let (Some(before), Some(after)) = (next.checked_sub(1).map(|index| &pairs[index]), pairs.get(next)) else {
            continue;
        };
        let between = text.get(before.written.end..after.written.start).unwrap_or_default();
        if punctuated(between) {
            continue;
        }
        let pause = MidPhrasePause {
            before: text[before.written.clone()].to_string(),
            after: text[after.written.clone()].to_string(),
            seconds: silence.seconds(),
        };
        match pauses.iter_mut().find(|(at, _)| *at == next) {
            Some((_, found)) if found.seconds < pause.seconds => *found = pause,
            Some(_) => {}
            None => pauses.push((next, pause)),
        }
    }
    pauses.into_iter().map(|(_, pause)| pause).collect()
}

/// The marks that end a sentence or a phrase, and brackets. Quotation marks
/// aren't among them: in `call me "an old-fashioned fellow"` the phrase runs on
/// into the quotation.
const PAUSE_MARKS: [char; 15] = ['.', ',', ';', ':', '!', '?', '…', '—', '–', '(', ')', '[', ']', '{', '}'];

/// Whether the text between two words, with any words missed between them,
/// has a mark that explains a pause: one of `PAUSE_MARKS`, or a hyphen used
/// as a dash, beside a space or another hyphen, as in " - " or "--". A hyphen
/// joining two words, as in "well-known", and symbols such as "%" or "&" don't.
fn punctuated(between: &str) -> bool {
    let chars: Vec<char> = between.chars().collect();
    let dash_beside = |index: Option<usize>| {
        index.and_then(|index| chars.get(index)).is_some_and(|c| c.is_whitespace() || *c == '-')
    };
    chars.iter().enumerate().any(|(index, c)| {
        PAUSE_MARKS.contains(c) || (*c == '-' && (dash_beside(index.checked_sub(1)) || dash_beside(Some(index + 1))))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 24_000;

    /// Steady noise at `db`, the same every time.
    fn room_tone(seconds: f32, db: f32) -> Vec<f32> {
        // Uniform noise from -1 to 1 has an RMS of 1/sqrt(3).
        let scale = 10f32.powf(db / 20.0) * 3f32.sqrt();
        // A linear congruential generator with the Numerical Recipes constants,
        // from a fixed seed, so every run gets the same noise.
        let mut state = 12_345u32;
        (0..(RATE as f32 * seconds) as usize)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                scale * (state as f32 / u32::MAX as f32 * 2.0 - 1.0)
            })
            .collect()
    }

    fn voice(seconds: f32) -> Vec<f32> {
        (0..(RATE as f32 * seconds) as usize).map(|i| 0.2 * (i as f32 * 0.05).sin()).collect()
    }

    /// Audio laid out as `parts`: positive seconds of voice, negative seconds of
    /// room tone at -70 dB.
    fn take(parts: &[f32]) -> AudioSamples {
        let mut samples = Vec::new();
        for &part in parts {
            samples.extend(if part > 0.0 { voice(part) } else { room_tone(-part, -70.0) });
        }
        AudioSamples::new(samples, RATE)
    }

    fn near(silence: &Silence, start: f32, end: f32) -> bool {
        (silence.start - start).abs() < 0.015 && (silence.end - end).abs() < 0.015
    }

    #[test]
    fn silences_as_long_as_the_limit_are_found_and_shorter_ones_are_not() {
        let audio = take(&[1.0, -0.7, 1.0, -0.5, 1.0, -0.6, 1.0]);
        let found = silences(&audio.samples, RATE, 0.6);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(near(&found[0], 1.0, 1.7), "{found:?}");
        assert!(near(&found[1], 4.2, 4.8), "{found:?}");
    }

    #[test]
    fn silence_is_measured_against_the_take_s_own_floor() {
        // Room tone 15 dB above the floor isn't silence; 5 dB above it is.
        let mut samples = room_tone(1.0, -70.0);
        samples.extend(voice(1.0));
        samples.extend(room_tone(1.0, -55.0));
        samples.extend(voice(1.0));
        samples.extend(room_tone(1.0, -65.0));
        samples.extend(voice(1.0));
        let found = silences(&samples, RATE, 0.6);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(near(&found[0], 0.0, 1.0) && near(&found[1], 4.0, 5.0), "{found:?}");
    }

    #[test]
    fn digital_silence_is_silent_but_gives_no_floor() {
        let mut samples = vec![0.0; RATE as usize];
        samples.extend(voice(1.0));
        samples.extend(room_tone(1.0, -70.0));
        let found = silences(&samples, RATE, 0.6);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(near(&found[0], 0.0, 1.0) && near(&found[1], 2.0, 3.0), "{found:?}");
        assert!(silences(&vec![0.0; RATE as usize], RATE, 0.6).is_empty());
    }

    fn word(text: &str, start: f32, end: f32) -> Word {
        Word { text: text.into(), start, end }
    }

    /// Words heard at these times, each running on 0.3 s into what follows, as
    /// Parakeet's end times do.
    fn heard(times: &[(&str, f32, f32)]) -> Vec<Word> {
        times.iter().map(|&(text, start, end)| word(text, start, end + 0.3)).collect()
    }

    const BORN: &str = "It was the night in which I was born, on the foaming sea.";

    /// "in which I was ⏸ born, ⏸ on the sea", with silence before and after.
    fn born_take() -> (AudioSamples, Vec<Word>) {
        let audio = take(&[-0.8, 1.0, -0.72, 0.5, -0.9, 1.0, -1.0]);
        let words = heard(&[
            ("in", 0.8, 1.1),
            ("which", 1.1, 1.4),
            ("I", 1.4, 1.6),
            ("was", 1.6, 1.8),
            ("born", 2.52, 3.02),
            ("on", 3.92, 4.1),
            ("the", 4.1, 4.3),
            ("sea", 4.3, 4.92),
        ]);
        (audio, words)
    }

    #[test]
    fn a_pause_where_the_text_has_no_punctuation_is_named_with_its_words_and_length() {
        let (audio, words) = born_take();
        let pauses = mid_phrase(BORN, &words, &audio, 0.6, &Allowances::default());
        assert_eq!(pauses.len(), 1, "{pauses:?}");
        assert_eq!((pauses[0].before.as_str(), pauses[0].after.as_str()), ("was", "born"));
        assert!((pauses[0].seconds - 0.72).abs() < 0.015, "{pauses:?}");
        let shown = MidPhrasePause { seconds: 0.72, ..pauses[0].clone() };
        assert_eq!(shown.to_string(), "was … born 0.72 s");
    }

    #[test]
    fn a_word_heard_as_starting_early_in_the_pause_before_it_still_follows_the_pause() {
        // "born" is said after the 0.72 s silence, but heard as starting 0.2 s
        // into it, before its middle. The comma after "born" doesn't explain
        // the pause before it.
        let (audio, mut words) = born_take();
        words[4].start = 2.0;
        let pauses = mid_phrase(BORN, &words, &audio, 0.6, &Allowances::default());
        assert_eq!(pauses.len(), 1, "{pauses:?}");
        assert_eq!((pauses[0].before.as_str(), pauses[0].after.as_str()), ("was", "born"));
    }

    #[test]
    fn a_pause_shorter_than_the_limit_is_not_counted() {
        let (audio, words) = born_take();
        assert!(mid_phrase(BORN, &words, &audio, 0.8, &Allowances::default()).is_empty());
    }

    #[test]
    fn silence_at_the_start_or_end_of_a_take_is_not_a_pause() {
        let audio = take(&[-1.2, 1.0, -1.2]);
        let words = heard(&[("in", 1.2, 1.5), ("which", 1.5, 2.2)]);
        assert!(mid_phrase("in which", &words, &audio, 0.6, &Allowances::default()).is_empty());
    }

    #[test]
    fn a_first_word_heard_as_starting_inside_the_opening_silence_is_no_pause() {
        // A noise, then 0.95 s of silence before "One day", which the checker
        // hears as starting part way through the silence.
        let audio = take(&[0.3, -0.95, 1.0, -0.3]);
        let words = heard(&[("One", 0.75, 1.3), ("day", 1.3, 2.25)]);
        assert!(mid_phrase("One day", &words, &audio, 0.6, &Allowances::default()).is_empty());
    }

    #[test]
    fn punctuation_beside_a_word_missed_explains_a_pause() {
        // "boat," wasn't heard, so the pause falls between "the" and "sank",
        // and the comma after the missed word explains it.
        let audio = take(&[0.5, -0.8, 1.0]);
        let words = heard(&[("into", 0.0, 0.25), ("the", 0.25, 0.5), ("sank", 1.3, 1.6), ("slowly", 1.6, 2.3)]);
        let text = "into the boat, sank slowly";
        assert!(mid_phrase(text, &words, &audio, 0.6, &Allowances::default()).is_empty());
        let text = "into the boat sank slowly";
        let pauses = mid_phrase(text, &words, &audio, 0.6, &Allowances::default());
        assert_eq!((pauses[0].before.as_str(), pauses[0].after.as_str()), ("the", "sank"));
    }

    #[test]
    fn words_heard_through_allowances_and_numbers_place_a_pause() {
        let audio = take(&[0.6, -0.7, 1.0]);
        let words = heard(&[("and", 0.0, 0.2), ("the", 0.2, 0.6), ("twenty", 1.3, 1.7), ("one", 1.7, 2.3)]);
        let allowances = Allowances::parse("in = and").unwrap();
        let pauses = mid_phrase("in the 21 boats", &words, &audio, 0.6, &allowances);
        assert_eq!((pauses[0].before.as_str(), pauses[0].after.as_str()), ("the", "21"));
    }

    #[test]
    fn a_take_with_no_words_heard_has_no_pauses() {
        let audio = take(&[0.5, -1.0, 0.5]);
        assert!(mid_phrase("in which", &[], &audio, 0.6, &Allowances::default()).is_empty());
    }

    #[test]
    fn punctuation_is_a_sentence_or_phrase_mark_a_dash_or_a_bracket() {
        for between in
            [", ", "; ", ". ", ": ", "! ", "? ", "… ", "—", " – ", ",\" ", " (", ") ", " [", " - ", "--", " -", "- "]
        {
            assert!(punctuated(between), "{between:?}");
        }
        for between in [" ", "-", "’", "'", "\u{a0}", " \"", "\" ", " “", "” ", " % ", " £", " & ", "*", "_", " / ", " # "] {
            assert!(!punctuated(between), "{between:?}");
        }
        assert!(punctuated(" boat, "), "punctuation beside a word missed");
        assert!(!punctuated(" boat "));
    }

    #[test]
    fn pauses_are_listed_with_a_plain_separator() {
        let pause =
            |before: &str, after: &str, seconds| MidPhrasePause { before: before.into(), after: after.into(), seconds };
        let found = MidPhrasePauses::Measured(vec![pause("was", "born", 0.72), pause("the", "sea", 1.1)]);
        assert_eq!(found.list(), "was … born 0.72 s | the … sea 1.10 s");
        assert_eq!(found.column(), found.list());
        assert!(!found.none_found());
        let none = MidPhrasePauses::Measured(Vec::new());
        assert_eq!((none.list(), none.column(), none.none_found()), (String::new(), String::new(), true));
        let unmeasured = MidPhrasePauses::Unmeasured;
        let shown = (unmeasured.list(), unmeasured.column(), unmeasured.none_found());
        assert_eq!(shown, (String::new(), "unmeasured".to_string(), true));
        assert_eq!((MidPhrasePauses::Off.column(), MidPhrasePauses::Off.none_found()), (String::new(), true));
    }

    #[test]
    fn a_take_is_checked_only_with_word_times_and_a_limit() {
        let (audio, words) = born_take();
        let none = Allowances::default();
        let check = |limit, words| MidPhrasePauses::check(limit, BORN, words, &audio, &none);
        assert_eq!(check(PauseLimit::Off, Some(&words)), MidPhrasePauses::Off);
        assert_eq!(check(PauseLimit::DEFAULT, None), MidPhrasePauses::Unmeasured);
        let found = check(PauseLimit::DEFAULT, Some(&words));
        assert!(matches!(&found, MidPhrasePauses::Measured(found) if found.len() == 1), "{found:?}");
    }

    #[test]
    fn a_pause_limit_is_off_or_a_length_from_0_3_to_5_seconds() {
        assert_eq!(parse_pause_limit("0.6"), Ok(PauseLimit::Seconds(0.6)));
        assert_eq!(parse_pause_limit(" 5 "), Ok(PauseLimit::Seconds(5.0)));
        assert_eq!(parse_pause_limit("Off"), Ok(PauseLimit::Off));
        for bad in ["0.2", "6", "long", "", "-1"] {
            assert_eq!(parse_pause_limit(bad), Err("must be off or a length from 0.3 to 5 seconds".to_string()), "{bad}");
        }
        for limit in [PauseLimit::Off, PauseLimit::DEFAULT, PauseLimit::Seconds(1.25)] {
            assert_eq!(valid_pause_limit(&limit.saved()), Some(limit));
        }
        assert_eq!(PauseLimit::DEFAULT.to_string(), "0.6 s");
    }
}
