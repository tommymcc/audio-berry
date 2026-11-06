//! Judges a voice sample's floor, the level of its pauses, which takes copy,
//! and advises what to do about a noisy one.

use crate::book::MAX_TAKES;
use crate::clean::{self, DEFAULT_REDUCTION_DB};
use crate::quality::{self, DEFAULT_NOISE_LIMIT_DB, NOISE_LIMIT_VAR};
use crate::voice::SAMPLE_RATE;
use std::path::Path;

/// Takes copy the voice sample's floor, a few dB quieter. On one speaker, samples
/// cleaned to floors of -45.3, -48.9 and -58.0 dB gave typical takes of -53.6,
/// -55.7 and -59.3, so takes reach the noise limit when the sample's floor is
/// about 7 dB above it: about -48 dB for the default of -55.
const NOISY_ABOVE_LIMIT_DB: f32 = 7.0;
/// The same speaker's raw recordings measured -30.4 and -31.9 dB, and every
/// cleaned sample -44.4 or below, so a sample above this probably hasn't been
/// cleaned.
const RAW_SAMPLE_DB: f32 = -40.0;

/// What a voice sample's floor means for its takes. It's advice, since only
/// listening can say whether takes sound clean.
///
/// The floor is judged as a level, not relative to the speech, because takes'
/// floors are measured the same way, on the takes as rendered, against the
/// noise limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleCheck {
    floor_db: f32,
    /// The noise limit given in the environment, if any.
    noise_limit_db: Option<f32>,
    verdict: Verdict,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Verdict {
    /// Takes should pass the noise limit.
    Fine,
    /// Faint background, such as a cleaned outdoor recording: many takes will
    /// fail the noise limit.
    Noisy,
    /// Loud background, probably a recording that hasn't been cleaned.
    Raw,
}

impl SampleCheck {
    /// Checks the samples of a voice sample, at the rate voice samples are kept at.
    pub fn of(samples: &[f32], noise_limit_db: Option<f32>) -> SampleCheck {
        SampleCheck::from_floor(quality::measure(samples, SAMPLE_RATE).floor_db, noise_limit_db)
    }

    fn from_floor(floor_db: f32, noise_limit_db: Option<f32>) -> SampleCheck {
        let limit = noise_limit_db.unwrap_or(DEFAULT_NOISE_LIMIT_DB);
        let verdict = if floor_db > RAW_SAMPLE_DB {
            Verdict::Raw
        } else if floor_db > limit + NOISY_ABOVE_LIMIT_DB {
            Verdict::Noisy
        } else {
            Verdict::Fine
        };
        SampleCheck { floor_db, noise_limit_db, verdict }
    }

    /// Follows the sample's length on every load, so a sample with a transcript
    /// already shows its floor too.
    pub fn summary(&self) -> String {
        let floor = self.floor_db;
        match self.verdict {
            Verdict::Fine => format!(", floor {floor:.1} dB"),
            Verdict::Noisy => format!(", floor {floor:.1} dB, noisy, see \"Noise limit\" in the README"),
            Verdict::Raw => format!(", floor {floor:.1} dB, a raw recording? See \"Adding a voice\" in the README"),
        }
    }

    /// What to do about a noisy or raw voice sample at `path`, whose recording
    /// lasts `seconds`, if anything.
    pub fn advice(&self, path: &Path, seconds: f32) -> Option<String> {
        let floor = self.floor_db;
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let cleaned = path.with_file_name(format!("{stem}-clean.wav"));
        match self.verdict {
            Verdict::Fine => None,
            Verdict::Noisy => {
                let limit = match self.noise_limit_db {
                    Some(limit) => format!("the noise limit of {limit} dB set by {NOISE_LIMIT_VAR}"),
                    None => format!("the default noise limit of {DEFAULT_NOISE_LIMIT_DB} dB"),
                };
                // The next multiple of 5 dB above the floor takes are expected
                // to have, as -50 was for a -45.3 dB sample.
                let raised = ((floor - NOISY_ABOVE_LIMIT_DB) / 5.0).ceil() * 5.0;
                let path = path.display();
                let clean = clean::command_line(Path::new("recording.mp3"), "<start>", "<end>", &cleaned, Some(20.0));
                Some(format!(
                    "The voice sample's floor is {floor:.1} dB. Takes copy it, a few dB quieter, so many \
                     will have floors above {limit} and be retried, up to {MAX_TAKES} takes per chunk. \
                     Either:\n\
                     - keep this sample, and raise the limit when you narrate with book or retake, if test \
                     takes sound clean:\n    \
                     {NOISE_LIMIT_VAR}={raised} audio-berry book {path} book.epub output/book\n\
                     - or clean the original recording again into a new voice sample, with a higher \
                     --reduction than it had ({DEFAULT_REDUCTION_DB} dB by default). In tests, 20 dB gave \
                     takes about -56 dB and 30 about -59, with the clone less like the speaker:\n    \
                     {clean}\n\
                     See \"Noise limit\" in the README."
                ))
            }
            Verdict::Raw => {
                // Rounded down, since clean refuses an end past the recording's.
                let end = (seconds * 10.0).floor() / 10.0;
                let clean = clean::command_line(path, "0", &format!("{end:.1}"), &cleaned, None);
                Some(format!(
                    "The voice sample's floor is {floor:.1} dB, loud enough to hear in every take. If it's \
                     a raw recording, clean it into a new voice sample first:\n    \
                     {clean}\n\
                     See \"Adding a voice\" in the README."
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sample_is_judged_against_the_noise_limit() {
        let verdict = |floor, limit| SampleCheck::from_floor(floor, limit).verdict;
        // Raw recordings, whatever the limit.
        assert_eq!(verdict(-31.9, None), Verdict::Raw);
        assert_eq!(verdict(-31.9, Some(-30.0)), Verdict::Raw);
        // Cleaned at 15 dB: typical takes -53.6, failing -55 but passing -50.
        assert_eq!(verdict(-45.3, None), Verdict::Noisy);
        assert_eq!(verdict(-45.3, Some(-50.0)), Verdict::Fine);
        // Cleaned at 20 dB: typical takes -55.7, just passing -55.
        assert_eq!(verdict(-48.9, None), Verdict::Fine);
        assert_eq!(verdict(-48.9, Some(-60.0)), Verdict::Noisy);
        // Cleaned at 30 dB and without limit.
        assert_eq!(verdict(-58.0, None), Verdict::Fine);
        assert_eq!(verdict(-80.7, None), Verdict::Fine);
    }

    #[test]
    fn a_sample_check_measures_the_quiet_frames() {
        // Eight seconds of speech-level tone, then a second of background.
        let speech: Vec<f32> = (0..24_000 * 8).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let with_background = |level: f32| {
            let mut samples = speech.clone();
            samples.extend((0..24_000).map(|i| (i as f32 * 0.07).sin() * level));
            SampleCheck::of(&samples, None).verdict
        };
        // Silence: fine.
        assert_eq!(with_background(0.0), Verdict::Fine);
        // A sine at 0.008 is about -45 dB: noisy at the default limit.
        assert_eq!(with_background(0.008), Verdict::Noisy);
        // At 0.05, about -29 dB: raw.
        assert_eq!(with_background(0.05), Verdict::Raw);
    }

    #[test]
    fn a_noisy_sample_gets_both_ways_out() {
        let check = SampleCheck::from_floor(-45.3, None);
        assert!(check.summary().contains("floor -45.3 dB, noisy"), "{}", check.summary());
        let advice = check.advice(Path::new("voices/dad.wav"), 10.7).unwrap();
        assert!(advice.contains("-45.3 dB"), "{advice}");
        assert!(advice.contains("default noise limit of -55 dB"), "{advice}");
        assert!(advice.contains("up to 5 takes per chunk"), "{advice}");
        assert!(advice.contains("AUDIO_BERRY_NOISE_LIMIT_DB=-50 audio-berry book voices/dad.wav"), "{advice}");
        assert!(advice.contains("<end> voices/dad-clean.wav --reduction 20"), "{advice}");
        assert!(advice.contains("15 dB by default"), "{advice}");
        assert!(advice.contains("\"Noise limit\""), "{advice}");
    }

    #[test]
    fn a_raw_sample_is_cleaned_into_a_new_file() {
        let check = SampleCheck::from_floor(-31.9, None);
        let advice = check.advice(Path::new("voices/raw.wav"), 10.76).unwrap();
        assert!(advice.contains("clean voices/raw.wav 0 10.7 voices/raw-clean.wav"), "{advice}");
    }

    #[test]
    fn a_fine_sample_gets_only_its_floor() {
        let check = SampleCheck::from_floor(-45.3, Some(-50.0));
        assert_eq!(check.summary(), ", floor -45.3 dB");
        assert_eq!(check.advice(Path::new("voices/dad.wav"), 10.7), None);
    }
}
