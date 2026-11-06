//! Suggests the sections of a long recording that would make the best voice
//! samples: whole phrases of clear, steady speech with few pauses and a quiet
//! floor once cleaned. Finding them by ear is the slow part of adding a voice.

use crate::audio::decode_audio_up_to;
use crate::clean::{self, CLEANING_RATE, DEFAULT_REDUCTION_DB, shell_word};
use crate::quality::{self, FRAME_SECONDS};
use crate::transcribe::{CHECK_MODEL, Phrase, Transcriber};
use crate::voice::SAMPLE_RATE;
use any_tts::AudioSamples;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// How many sections to suggest unless told otherwise.
pub const DEFAULT_COUNT: usize = 3;

/// Hears where the phrases are, and what's said in the sections suggested. On a
/// 34 second recording on the CPU, base.en found the phrases in 10 seconds with
/// a peak of 0.7 GB of memory, and medium.en in 97 seconds with 5.4 GB, with
/// timestamps as rough. Its words are only shown: a voice sample's first use
/// transcribes it with medium.en.
const MODEL: &str = CHECK_MODEL;

/// The longest recording looked through. Memory grows with length, since the
/// whole recording is cleaned at once: on the CPU, 10 minutes took 3.5 minutes
/// with a peak of 0.81 GB, and 20 took 7 with 1.07 GB.
const MAX_RECORDING_SECONDS: f64 = 30.0 * 60.0;

/// OmniVoice recommends 3-10 seconds of reference, and 11.4 is what's been
/// tested for narration. Shorter sections hold too little of the voice.
const MIN_SECONDS: f64 = 8.0;
const MAX_SECONDS: f64 = 15.0;

/// The silence kept before the first word and after the last, at most.
const MARGIN_SECONDS: f64 = 0.2;
/// The shortest silence a section may start or end in where Whisper heard a
/// break between phrases.
const MIN_CUT_PAUSE_SECONDS: f64 = 0.15;
/// Elsewhere, a gap between words must be this long, so it's not the silence
/// inside a word, such as before a "p".
const MIN_WORD_GAP_SECONDS: f64 = 0.25;
/// Whisper's timestamps are rough, so a pause this close to where Whisper put
/// a break between phrases counts as that break.
const TIMESTAMP_SLACK_SECONDS: f64 = 0.5;
/// A pause this long is where a speaker breaks between phrases, so it's a
/// good place to cut even where Whisper heard no break.
const PHRASE_PAUSE_SECONDS: f64 = 0.6;

/// A frame is speech when it's this far from the recording's floor to its
/// loudest speech. Cleaned outdoor recordings keep a floor of about -45 dB, so
/// a fixed level, as `quality` uses for takes, would hear their pauses as speech.
const SPEECH_FRACTION_OF_RANGE: f32 = 0.3;
/// The loudest speech, as the level this fraction of frames are quieter than.
const LOUD_FRACTION: f32 = 0.95;
/// A frame whose raw peak reaches this is probably clipped.
const CLIPPED_PEAK: f32 = 0.99;
/// Whisper's confidence, as an average log probability, counted for time in a
/// section where it heard no words, such as music or other noise.
const UNHEARD_LOGPROB: f64 = -2.0;

/// Frames are counted exactly, since `FRAME_SECONDS` as an `f32` isn't quite
/// 0.05, and times are rounded to tenths.
const FRAMES_PER_SECOND: f64 = 20.0;
const _: () = assert!((FRAME_SECONDS as f64 * FRAMES_PER_SECOND - 1.0).abs() < 1e-6);
/// Times this close are the same time: sums of frame times aren't exact.
const TIME_EPSILON: f64 = 1e-6;

// How sections are scored, in points. One point is about one noticeable flaw.
// The weights were tuned on one 34 second outdoor recording, where they rank
// the two sections a person picked by ear highest.

/// For a section that's all speech, since silence holds none of the voice.
const SPEECH_POINTS: f64 = 10.0;
/// For each end between sentences, or between phrases, so the clone hears
/// the speaker's phrasing whole.
const SENTENCE_BREAK_POINTS: f64 = 1.0;
const PHRASE_BREAK_POINTS: f64 = 0.5;
/// Pauses up to this long are part of natural speech: both sections a person
/// picked by ear held pauses of nearly a second.
const NATURAL_PAUSE_SECONDS: f64 = 1.0;
/// Lost per second of pause beyond that, which wastes the reference on silence
/// and teaches the clone to hesitate.
const LONG_PAUSE_POINTS_PER_SECOND: f64 = 4.0;
/// A floor this quiet costs nothing: takes from it pass the default noise limit.
const QUIET_FLOOR_DB: f32 = -60.0;
/// Lost per dB of floor above that. Floors within a recording differ by a few
/// dB, so this breaks ties rather than outweighing the speech.
const FLOOR_POINTS_PER_DB: f64 = 0.1;
/// Lost for any clipping, which every take copies.
const CLIPPING_POINTS: f64 = 3.0;
/// And per clipped frame, since some recordings clip all through, and then
/// the fewest clipped frames is best.
const CLIPPED_FRAME_POINTS: f64 = 0.1;
/// Speech's level varies this much, as a standard deviation, in sections that
/// sounded steady: 5 to 7 dB on one recording.
const STEADY_SPREAD_DB: f32 = 6.0;
/// Lost per dB of spread beyond that, for a voice that comes and goes.
const SPREAD_POINTS_PER_DB: f64 = 0.5;
/// Whisper's average log probability for clear speech is about -0.1 to -0.3.
const CLEAR_LOGPROB: f64 = -0.3;
/// Lost per unit of log probability below that, for speech that's hard to
/// make out, or isn't speech.
const UNCLEAR_POINTS: f64 = 5.0;
/// The length aimed for, near the 11.4 seconds tested for narration.
const IDEAL_SECONDS: f64 = 11.0;
/// Lost per second away from it.
const LENGTH_POINTS_PER_SECOND: f64 = 0.3;

/// Suggests the best `count` sections of `recording` to use as a voice sample,
/// with what's said in each and the `clean` command that makes it, cleaning
/// with `reduction_db` if given, as `clean` would.
pub fn suggest(recording: &Path, count: usize, reduction_db: Option<f32>) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let noisy = decode_audio_up_to(recording, CLEANING_RATE, MAX_RECORDING_SECONDS + 1.0)?;
    let length = noisy.len() as f64 / f64::from(CLEANING_RATE);
    if length > MAX_RECORDING_SECONDS {
        return Err(too_long(recording).into());
    }
    println!("Cleaning {length:.1}s of {}, then finding its phrases with Whisper {MODEL}", recording.display());
    let cleaned = clean::clean_samples(&noisy, reduction_db.unwrap_or(DEFAULT_REDUCTION_DB))?;
    let frames = Frames::measure(&noisy, &cleaned);
    // The raw recording is the largest thing held, and isn't needed again.
    drop(noisy);
    println!("Cleaned in {:.1}s", started.elapsed().as_secs_f32());

    let started = Instant::now();
    let audio = AudioSamples::new(cleaned, SAMPLE_RATE);
    let mut transcriber = Transcriber::start(MODEL)?;
    let phrases = transcriber.transcribe_phrases(&audio)?;
    let cleaned = audio.samples;
    println!("Found {} phrases in {:.1}s", phrases.len(), started.elapsed().as_secs_f32());

    let samples_between = |start: f64, end: f64| {
        let index = |seconds: f64| ((seconds * f64::from(SAMPLE_RATE)).round() as usize).min(cleaned.len());
        &cleaned[index(start)..index(end)]
    };
    let floor_of = |start: f64, end: f64| quality::measure(samples_between(start, end), SAMPLE_RATE).floor_db;
    let length = cleaned.len() as f64 / f64::from(SAMPLE_RATE);
    let sections = best_sections(&phrases, &frames, length, count, floor_of);
    if sections.is_empty() {
        return Err(format!(
            "found no stretch of {MIN_SECONDS:.0} to {MAX_SECONDS:.0} seconds of speech in {} that starts and \
             ends in a pause",
            recording.display()
        )
        .into());
    }

    // Heard again, section by section, because Whisper's phrases cross the
    // cuts: joining them would show words from outside a section, or miss
    // some. This takes a few seconds, and gives the words the sample holds.
    let started = Instant::now();
    let mut heard = Vec::new();
    for section in &sections {
        // Clamped as clean saves it, so this is heard as the voice sample will be.
        let samples = samples_between(section.start, section.end).iter().map(|s| s.clamp(-1.0, 1.0)).collect();
        heard.push(transcriber.transcribe(&AudioSamples::new(samples, SAMPLE_RATE))?);
    }
    println!("Transcribed them in {:.1}s", started.elapsed().as_secs_f32());

    println!();
    if sections.len() < count {
        println!("Found {} sections that don't overlap, of the {count} asked for:", sections.len());
    }
    for (number, (section, text)) in (1..).zip(sections.iter().zip(heard)) {
        let output = sample_path(recording, section.start);
        println!("{number}. {}", section.summary());
        println!("   Roughly: \"{text}\"");
        let (start, end) = (format!("{:.1}", section.start), format!("{:.1}", section.end));
        println!("   {}", clean::command_line(recording, &start, &end, &output, reduction_db));
    }
    println!(
        "\nMake each with its clean command, then hear how each clones with audition, and keep the one most \
         like the speaker:\n    audio-berry audition {}",
        shell_word(&sample_path(recording, sections[0].start))
    );
    Ok(())
}

/// Why `recording` is refused for being too long, and how to cut a stretch of it.
fn too_long(recording: &Path) -> String {
    let part = Path::new(recording.file_stem().unwrap_or_default()).with_extension("part.wav");
    format!(
        "{} is longer than {:.0} minutes, the most sections looks through at once. Cut a stretch of it \
         first, such as 10 minutes starting 5 minutes in, and look through that:\n    \
         ffmpeg -ss 300 -t 600 -i {} {}",
        recording.display(),
        MAX_RECORDING_SECONDS / 60.0,
        shell_word(recording),
        shell_word(&part)
    )
}

/// Where the voice sample of the section starting at `start` goes, named by
/// its start so it stays the same section whatever the ranking:
/// `voices/<recording's name>-<start>s.wav`.
fn sample_path(recording: &Path, start: f64) -> PathBuf {
    let stem = recording.file_stem().unwrap_or_default().to_string_lossy();
    Path::new("voices").join(format!("{stem}-{start:.1}s.wav"))
}

/// The cleaned recording's level and whether the raw recording clipped, in
/// `FRAME_SECONDS` frames.
struct Frames {
    levels_db: Vec<f32>,
    clipped: Vec<bool>,
}

impl Frames {
    /// Measures `noisy`, at the cleaning rate, and `cleaned`, at the voice sample rate.
    fn measure(noisy: &[f32], cleaned: &[f32]) -> Frames {
        let levels_db = quality::frame_levels_db(cleaned, SAMPLE_RATE);
        let size = quality::frame_size(CLEANING_RATE);
        let clipped = noisy.chunks(size).map(|frame| quality::peak(frame) >= CLIPPED_PEAK).collect();
        Frames { levels_db, clipped }
    }

    /// Which frames are speech, judged against the whole recording's floor
    /// and loudest speech.
    fn speaking(&self) -> Vec<bool> {
        let mut sorted = self.levels_db.clone();
        sorted.sort_by(f32::total_cmp);
        let floor = quality::floor_of_sorted(&sorted).unwrap_or(0.0);
        let loud = sorted.get((sorted.len() as f32 * LOUD_FRACTION) as usize).copied().unwrap_or(0.0);
        let threshold = floor + SPEECH_FRACTION_OF_RANGE * (loud - floor);
        self.levels_db.iter().map(|level| *level > threshold).collect()
    }
}

fn frame_at(seconds: f64) -> usize {
    (seconds * FRAMES_PER_SECOND).round().max(0.0) as usize
}

fn seconds_at(frame: usize) -> f64 {
    frame as f64 / FRAMES_PER_SECOND
}

/// Rounds a start down to a tenth of a second, into the pause before it.
fn tenth_before(seconds: f64) -> f64 {
    (seconds * 10.0 + TIME_EPSILON).floor() / 10.0
}

/// Rounds an end up to a tenth of a second, into the pause after it.
fn tenth_after(seconds: f64) -> f64 {
    (seconds * 10.0 - TIME_EPSILON).ceil() / 10.0
}

/// A stretch of silence, in frames `start..end`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pause {
    start: usize,
    end: usize,
}

impl Pause {
    fn seconds(&self) -> f64 {
        seconds_at(self.end - self.start)
    }
}

/// Every silence in `speaking`, including any at either end.
fn pauses(speaking: &[bool]) -> Vec<Pause> {
    let mut pauses = Vec::new();
    let mut start = None;
    for (frame, &speaking) in speaking.iter().chain([&true]).enumerate() {
        match (speaking, start) {
            (false, None) => start = Some(frame),
            (true, Some(begun)) => {
                pauses.push(Pause { start: begun, end: frame });
                start = None;
            }
            _ => {}
        }
    }
    pauses
}

/// How well a section starting or ending in a pause keeps to the speaker's phrasing.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Break {
    /// Between sentences, as Whisper heard them.
    Sentence,
    /// Between phrases that aren't whole sentences, or a long pause.
    Phrase,
    /// A short gap between words.
    Word,
}

impl Break {
    /// How much a section gains from starting or ending here.
    fn points(self) -> f64 {
        match self {
            Break::Sentence => SENTENCE_BREAK_POINTS,
            Break::Phrase => PHRASE_BREAK_POINTS,
            Break::Word => 0.0,
        }
    }
}

/// A pause a section can start or end in.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cut {
    pause: Pause,
    at: Break,
}

impl Cut {
    /// Where a section starting here starts: just before the first word.
    fn start(&self) -> f64 {
        seconds_at(self.pause.start).max(seconds_at(self.pause.end) - MARGIN_SECONDS)
    }

    /// Where a section ending here ends: just after the last word.
    fn end(&self) -> f64 {
        seconds_at(self.pause.end).min(seconds_at(self.pause.start) + MARGIN_SECONDS)
    }
}

/// The pauses a section could start or end in, and what each breaks. A pause
/// near where Whisper heard one phrase end and the next begin breaks between
/// them, judged by their punctuation. Other pauses break only between words,
/// unless they're long.
fn cuts(phrases: &[Phrase], speaking: &[bool]) -> Vec<Cut> {
    let ends_sentence = |phrase: &Phrase| phrase.text.trim_end().ends_with(['.', '!', '?']);
    let starts_sentence = |phrase: &Phrase| phrase.text.starts_with(char::is_uppercase);
    // Where Whisper heard a break, give or take its slack, and whether it's
    // between sentences. Silence before the first phrase and after the last
    // counts too.
    let breaks: Vec<(f64, f64, bool)> = (0..=phrases.len())
        .map(|index| {
            let before = index.checked_sub(1).map(|before| &phrases[before]);
            let after = phrases.get(index);
            let from = before.map_or(0.0, |phrase| phrase.end);
            let to = after.map_or(f64::INFINITY, |phrase| phrase.start);
            let sentence = before.is_none_or(ends_sentence) && after.is_none_or(starts_sentence);
            (from - TIMESTAMP_SLACK_SECONDS, to + TIMESTAMP_SLACK_SECONDS, sentence)
        })
        .collect();
    pauses(speaking)
        .into_iter()
        .filter_map(|pause| {
            let (start, end) = (seconds_at(pause.start), seconds_at(pause.end));
            let heard = breaks.iter().filter(|(from, to, _)| start < *to && end > *from);
            let at = match heard.map(|(_, _, sentence)| *sentence).reduce(|a, b| a || b) {
                Some(true) => Break::Sentence,
                Some(false) => Break::Phrase,
                None if pause.seconds() > PHRASE_PAUSE_SECONDS => Break::Phrase,
                None if pause.seconds() >= MIN_WORD_GAP_SECONDS - TIME_EPSILON => Break::Word,
                None => return None,
            };
            (pause.seconds() >= MIN_CUT_PAUSE_SECONDS - TIME_EPSILON).then_some(Cut { pause, at })
        })
        .collect()
}

/// A stretch of the recording that could become a voice sample, from one pause
/// to another, and what makes it good or bad.
#[derive(Clone, Debug, PartialEq)]
struct Section {
    /// In seconds, to a tenth, in the pauses either side.
    start: f64,
    end: f64,
    from: Break,
    to: Break,
    /// The fraction of frames that are speech.
    speech: f64,
    /// Silence beyond `NATURAL_PAUSE_SECONDS` in its pauses, in seconds.
    excess_pause: f64,
    /// How much the speech's level varies, as the standard deviation in dB.
    level_spread_db: f32,
    clipped_frames: usize,
    /// Whisper's average log probability for the words of the phrases it
    /// overlaps, weighted by how much of each it holds, with `UNHEARD_LOGPROB`
    /// for the rest.
    logprob: f64,
    floor_db: f32,
}

impl Section {
    /// Measures the section from `start` to `end`, between `from` and `to`.
    fn measure(phrases: &[Phrase], speaking: &[bool], frames: &Frames, (start, end): (f64, f64), (from, to): (Break, Break), floor_db: f32) -> Section {
        let span = frame_at(start).min(speaking.len())..frame_at(end).min(speaking.len());
        let inside = &speaking[span.clone()];
        let speech = inside.iter().filter(|speaking| **speaking).count() as f64 / inside.len().max(1) as f64;
        let excess_pause = pauses(inside).iter().map(|pause| (pause.seconds() - NATURAL_PAUSE_SECONDS).max(0.0)).sum();

        let levels: Vec<f32> = span.clone().filter(|&frame| speaking[frame]).map(|frame| frames.levels_db[frame]).collect();
        let count = levels.len().max(1) as f32;
        let mean = levels.iter().sum::<f32>() / count;
        let variance = levels.iter().map(|level| (level - mean).powi(2)).sum::<f32>() / count;
        let clipped_frames = span.filter(|&frame| frames.clipped.get(frame).copied().unwrap_or(false)).count();

        let overlap = |phrase: &Phrase| (phrase.end.min(end) - phrase.start.max(start)).max(0.0);
        let heard: f64 = phrases.iter().map(overlap).sum();
        let unheard = (end - start - heard).max(0.0);
        let logprob = (phrases.iter().map(|phrase| phrase.average_logprob * overlap(phrase)).sum::<f64>()
            + UNHEARD_LOGPROB * unheard)
            / (heard + unheard).max(TIME_EPSILON);
        Section {
            start,
            end,
            from,
            to,
            speech,
            excess_pause,
            level_spread_db: variance.sqrt(),
            clipped_frames,
            logprob,
            floor_db,
        }
    }

    fn seconds(&self) -> f64 {
        self.end - self.start
    }

    /// Higher is better. See the weights above.
    fn score(&self) -> f64 {
        let speech = SPEECH_POINTS * self.speech;
        let whole = self.from.points() + self.to.points();
        let pauses = LONG_PAUSE_POINTS_PER_SECOND * self.excess_pause;
        let floor = FLOOR_POINTS_PER_DB * f64::from((self.floor_db - QUIET_FLOOR_DB).max(0.0));
        let clipping = match self.clipped_frames {
            0 => 0.0,
            frames => CLIPPING_POINTS + CLIPPED_FRAME_POINTS * frames as f64,
        };
        let spread = SPREAD_POINTS_PER_DB * f64::from((self.level_spread_db - STEADY_SPREAD_DB).max(0.0));
        let unclear = UNCLEAR_POINTS * (CLEAR_LOGPROB - self.logprob).max(0.0);
        let length = LENGTH_POINTS_PER_SECOND * (self.seconds() - IDEAL_SECONDS).abs();
        speech + whole - pauses - floor - clipping - spread - unclear - length
    }

    fn summary(&self) -> String {
        let clipped = if self.clipped_frames > 0 { ", clipped" } else { "" };
        format!(
            "{:.1}-{:.1} ({:.1}s), floor {:.1} dB, {:.0}% speech{clipped}",
            self.start,
            self.end,
            self.seconds(),
            self.floor_db,
            100.0 * self.speech
        )
    }
}

/// Every section from one cut to another lasting `MIN_SECONDS` to
/// `MAX_SECONDS`, in a recording `length` seconds long, measured, with its
/// floor from `floor_of`, given a start and end in seconds.
fn sections(phrases: &[Phrase], frames: &Frames, length: f64, floor_of: impl Fn(f64, f64) -> f32) -> Vec<Section> {
    let speaking = frames.speaking();
    let cuts = cuts(phrases, &speaking);
    // clean refuses an end past the recording's, so the last pause ends here.
    let last = tenth_before(length);
    let mut sections = Vec::new();
    for (index, from) in cuts.iter().enumerate() {
        let start = tenth_before(from.start());
        for to in &cuts[index + 1..] {
            let end = tenth_after(to.end()).min(last);
            if end - start > MAX_SECONDS + TIME_EPSILON {
                break;
            }
            // Ending before the pause would cut the last word.
            if end - start < MIN_SECONDS - TIME_EPSILON || end < seconds_at(to.pause.start) - TIME_EPSILON {
                continue;
            }
            let floor_db = floor_of(start, end);
            sections.push(Section::measure(phrases, &speaking, frames, (start, end), (from.at, to.at), floor_db));
        }
    }
    sections
}

/// The best `count` sections that don't overlap, best first.
fn best_sections(
    phrases: &[Phrase],
    frames: &Frames,
    length: f64,
    count: usize,
    floor_of: impl Fn(f64, f64) -> f32,
) -> Vec<Section> {
    let mut candidates = sections(phrases, frames, length, floor_of);
    candidates.sort_by(|a, b| b.score().total_cmp(&a.score()));
    let mut chosen: Vec<Section> = Vec::new();
    for candidate in candidates {
        if chosen.len() == count {
            break;
        }
        if chosen.iter().all(|section| candidate.end <= section.start || candidate.start >= section.end) {
            chosen.push(candidate);
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEECH_DB: f32 = -20.0;
    const PAUSE_DB: f32 = -50.0;

    /// A recording with speech at the given times in seconds and silence
    /// between, and Whisper's phrases for it.
    struct Recording {
        phrases: Vec<Phrase>,
        frames: Frames,
        length: f64,
    }

    fn recording(length: f64, spoken: &[(f64, f64)], phrases: &[(f64, f64, &str)]) -> Recording {
        let frames = frame_at(length);
        let mut levels_db = vec![PAUSE_DB; frames];
        for &(start, end) in spoken {
            levels_db[frame_at(start)..frame_at(end)].fill(SPEECH_DB);
        }
        Recording {
            phrases: phrases.iter().map(|&(start, end, text)| phrase(start, end, text)).collect(),
            frames: Frames { levels_db, clipped: vec![false; frames] },
            length,
        }
    }

    /// Phrases that are each a sentence, spoken exactly when Whisper says.
    fn sentences(length: f64, spoken: &[(f64, f64)]) -> Recording {
        let phrases: Vec<(f64, f64, &str)> = spoken.iter().map(|&(start, end)| (start, end, "A sentence.")).collect();
        recording(length, spoken, &phrases)
    }

    fn phrase(start: f64, end: f64, text: &str) -> Phrase {
        Phrase { start, end, text: text.to_string(), average_logprob: -0.2 }
    }

    impl Recording {
        fn best(&self, count: usize) -> Vec<Section> {
            best_sections(&self.phrases, &self.frames, self.length, count, |_, _| -50.0)
        }

        fn cuts(&self) -> Vec<Cut> {
            cuts(&self.phrases, &self.frames.speaking())
        }
    }

    #[test]
    fn pauses_include_those_at_either_end() {
        let speaking = [false, true, true, false, false, true, false];
        let found = pauses(&speaking);
        assert_eq!(found, [Pause { start: 0, end: 1 }, Pause { start: 3, end: 5 }, Pause { start: 6, end: 7 }]);
        assert!(pauses(&[true, true]).is_empty());
    }

    #[test]
    fn a_break_is_found_near_where_whisper_put_it() {
        // Whisper puts the break at 2.0-2.1, but the silence is 2.2-2.6.
        let recording = recording(6.0, &[(0.0, 2.2), (2.6, 6.0)], &[(0.0, 2.0, "One."), (2.1, 6.0, "Two.")]);
        assert_eq!(recording.cuts(), [Cut { pause: Pause { start: 44, end: 52 }, at: Break::Sentence }]);
    }

    #[test]
    fn breaks_are_judged_by_punctuation_and_length() {
        let recording = recording(
            12.0,
            // A gap between words inside the first phrase, a long one inside
            // the second, and one too short to cut in inside the third.
            &[(0.5, 2.0), (2.3, 4.0), (4.4, 5.0), (6.0, 8.0), (8.4, 9.0), (9.1, 11.5)],
            &[(0.5, 4.0, "A clause,"), (4.4, 8.0, "and then more"), (8.4, 11.5, "Of it.")],
        );
        let at: Vec<(f64, Break)> = recording.cuts().iter().map(|cut| (seconds_at(cut.pause.start), cut.at)).collect();
        assert_eq!(
            at,
            [
                (0.0, Break::Sentence),
                (2.0, Break::Word),
                (4.0, Break::Phrase),
                (5.0, Break::Phrase),
                (8.0, Break::Phrase),
                (11.5, Break::Sentence),
            ]
        );
    }

    #[test]
    fn sections_start_and_end_in_pauses_with_a_short_margin() {
        let recording = sentences(30.0, &[(1.0, 6.0), (7.0, 12.0), (13.0, 18.0), (25.0, 29.0)]);
        let best = &recording.best(1)[0];
        // Two of the three close sentences, about 11 seconds, rather than all
        // three, which run past 15.
        assert!(best.start == 0.8 || best.start == 6.8, "{best:?}");
        assert!((best.seconds() - 11.4).abs() < 1e-6, "{best:?}");
    }

    #[test]
    fn a_section_ending_in_the_last_pause_keeps_to_the_recording() {
        // The last pause runs from 11.75 to the end at 11.95, so rounding the
        // end up would pass the end of the recording.
        let best = sentences(11.95, &[(0.5, 5.0), (5.4, 11.75)]).best(1);
        assert_eq!(best.len(), 1);
        assert_eq!((best[0].start, best[0].end), (0.3, 11.9));
    }

    #[test]
    fn a_long_pause_counts_against_a_section() {
        let steady = sentences(14.0, &[(0.5, 6.0), (6.4, 11.0)]).best(1)[0].clone();
        let halting = sentences(14.0, &[(0.5, 4.0), (6.4, 11.0)]).best(1)[0].clone();
        assert_eq!(steady.excess_pause, 0.0);
        assert!((halting.excess_pause - (2.4 - NATURAL_PAUSE_SECONDS)).abs() < 1e-6, "{halting:?}");
        assert!(steady.score() > halting.score() + 5.0);
    }

    #[test]
    fn whole_sentences_clear_speech_and_a_quiet_floor_score_higher() {
        let base = sentences(14.0, &[(0.5, 6.0), (6.4, 11.0)]).best(1)[0].clone();
        let mid_sentence = Section { from: Break::Word, to: Break::Word, ..base.clone() };
        let unclear = Section { logprob: -0.9, ..base.clone() };
        let noisy = Section { floor_db: -35.0, ..base.clone() };
        let clipped = Section { clipped_frames: 2, ..base.clone() };
        let uneven = Section { level_spread_db: 10.0, ..base.clone() };
        for worse in [mid_sentence, unclear, noisy, clipped, uneven] {
            assert!(base.score() > worse.score() + 1.0, "{worse:?}");
        }
    }

    #[test]
    fn picks_dont_overlap() {
        let spoken: Vec<(f64, f64)> = (0..12).map(|index| (index as f64 * 5.0 + 0.5, index as f64 * 5.0 + 4.7)).collect();
        let best = sentences(60.0, &spoken).best(3);
        assert_eq!(best.len(), 3);
        for (index, a) in best.iter().enumerate() {
            for b in &best[index + 1..] {
                assert!(a.end <= b.start || a.start >= b.end, "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn a_clipped_section_is_passed_over() {
        let mut recording = sentences(30.0, &[(0.5, 5.0), (5.4, 10.5), (15.5, 20.0), (20.4, 25.5)]);
        recording.frames.clipped[frame_at(3.0)] = true;
        let best = &recording.best(1)[0];
        assert!(best.start > 15.0, "{best:?}");
    }

    #[test]
    fn time_with_no_words_heard_counts_as_unclear() {
        // Speech-level sound from 6.4 to 11, such as music, with no phrase there.
        let spoken = [(0.5, 6.0), (6.4, 11.0)];
        let music = recording(14.0, &spoken, &[(0.5, 6.0, "A sentence.")]).best(1)[0].clone();
        let speech = sentences(14.0, &spoken).best(1)[0].clone();
        assert!(music.logprob < -0.9, "{music:?}");
        assert!(speech.score() > music.score() + 3.0);
    }

    #[test]
    fn too_little_speech_gives_no_sections() {
        assert!(sentences(10.0, &[(0.5, 3.0), (3.4, 6.0)]).best(3).is_empty());
    }

    #[test]
    fn times_round_outwards_into_the_pauses() {
        assert_eq!(tenth_before(20.15), 20.1);
        assert_eq!(tenth_before(2.9), 2.9);
        assert_eq!(tenth_after(30.71), 30.8);
        assert_eq!(tenth_after(14.2), 14.2);
    }

    #[test]
    fn a_long_recording_is_refused_with_a_way_to_cut_it() {
        let refusal = too_long(Path::new("tapes/Dad interview.mp3"));
        assert!(refusal.contains("longer than 30 minutes"), "{refusal}");
        assert!(refusal.ends_with("ffmpeg -ss 300 -t 600 -i 'tapes/Dad interview.mp3' 'Dad interview.part.wav'"), "{refusal}");
    }

    #[test]
    fn samples_are_named_after_the_recording_and_their_start() {
        assert_eq!(sample_path(Path::new("/tmp/dad-raw.wav"), 20.1), Path::new("voices/dad-raw-20.1s.wav"));
    }

    #[test]
    fn the_speech_threshold_follows_the_recording() {
        // A cleaned outdoor recording: pauses at -45 dB, speech at -20.
        let mut levels_db = vec![-45.0; 40];
        levels_db.extend(vec![-20.0; 60]);
        let frames = Frames { levels_db, clipped: vec![false; 100] };
        assert_eq!(frames.speaking().iter().filter(|speaking| **speaking).count(), 60);
    }
}
