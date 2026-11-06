//! The voice sample a narration is cloned from, and what's said in it.

use crate::audio::decode_audio;
use crate::book::MAX_TAKES;
use crate::clean::DEFAULT_REDUCTION_DB;
use crate::quality::{self, MAX_FLOOR_DB, NOISE_LIMIT_VAR};
use crate::store::fnv;
use crate::transcribe::{Transcriber, VOICE_TRANSCRIPTION_MODEL};
use any_tts::models::omnivoice::trim_long_audio;
use any_tts::{AudioSamples, ReferenceAudio};
use std::error::Error;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

/// The rate voice samples are kept at, as OmniVoice hears them.
pub const SAMPLE_RATE: u32 = 24_000;
/// OmniVoice recommends 3-10 seconds of reference and warns above 20, so longer
/// samples are cut to 15 at a pause.
const MAX_VOICE_SECONDS: f32 = 15.0;
const MIN_VOICE_SECONDS: f32 = 3.0;
const TRIM_VOICE_ABOVE_SECONDS: f32 = 20.0;

/// A voice sample as the model hears it, with what's said in it.
pub struct Voice {
    /// Always has a transcript.
    reference: ReferenceAudio,
}

impl Voice {
    /// Loads a voice sample and what's said in it. The text lives in a `.txt` file
    /// beside the sample, written by transcribing it the first time, and can be
    /// corrected by hand.
    pub fn load(path: &Path) -> Result<Voice, Box<dyn Error>> {
        let recording = decode_audio(path, SAMPLE_RATE)?;
        let samples = trim_long_audio(
            &recording,
            SAMPLE_RATE,
            MAX_VOICE_SECONDS,
            MIN_VOICE_SECONDS,
            TRIM_VOICE_ABOVE_SECONDS,
        );
        let check = SampleCheck::of(&samples, noise_limit_given());
        println!("Voice sample: using {:.1}s{}", seconds(&samples), check.summary());

        let transcript_path = path.with_extension("txt");
        let existing = match fs::read_to_string(&transcript_path) {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::NotFound => String::new(),
            Err(error) => return Err(error.into()),
        };
        let existing = existing.trim();
        if !existing.is_empty() {
            if samples.len() < recording.len() {
                println!(
                    "Note: {} was cut from {:.1}s to {:.1}s, so if {} describes the whole recording, \
                     it says more than the voice sample does. Delete it to transcribe the part kept, \
                     or cut the recording to {MAX_VOICE_SECONDS:.0}s or less.",
                    path.display(),
                    seconds(&recording),
                    seconds(&samples),
                    transcript_path.display()
                );
            }
            return Ok(Voice::new(samples, existing));
        }

        let heard = AudioSamples::new(samples, SAMPLE_RATE);
        let transcript = Transcriber::start(VOICE_TRANSCRIPTION_MODEL)?.transcribe(&heard)?;
        let samples = heard.samples;
        if transcript.is_empty() {
            return Err(format!("heard no speech in {}", path.display()).into());
        }
        fs::write(&transcript_path, format!("{transcript}\n"))?;
        println!("Heard in the voice sample: {transcript}");
        println!("Saved to {}. Correct any mistakes there.", transcript_path.display());
        // Advice once, on first use: the floor above is shown on every load.
        if let Some(advice) = check.advice(path, seconds(&recording)) {
            println!("{advice}");
        }
        Ok(Voice::new(samples, &transcript))
    }

    fn new(samples: Vec<f32>, transcript: &str) -> Voice {
        Voice { reference: ReferenceAudio::new(samples, SAMPLE_RATE).with_transcript(transcript) }
    }

    /// Identifies the voice, including its transcript since that changes the speech.
    /// Output directories save it, so the same sample must always give the same hash.
    pub fn hash(&self) -> u64 {
        let mut bytes: Vec<u8> = self.reference.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        bytes.extend_from_slice(self.transcript().as_bytes());
        fnv(&bytes, 0)
    }

    /// The sample and transcript in the form the model takes.
    pub fn reference(&self) -> &ReferenceAudio {
        &self.reference
    }

    /// What's said in the sample.
    pub fn transcript(&self) -> &str {
        self.reference.transcript.as_deref().unwrap_or_default()
    }

    /// Saves the sample as the model hears it. A long sample is trimmed, and its
    /// transcript only covers what's kept.
    pub fn save_heard(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        save(&self.reference.samples, path)
    }
}

/// Takes copy the voice sample's floor, a few dB quieter. On one speaker, samples
/// cleaned to floors of -45.3, -48.9 and -58.0 dB gave typical takes of -53.6,
/// -55.7 and -59.3, so takes reach the noise limit when the sample's floor is
/// about 7 dB above it: about -48 dB for the default of -55.
const NOISY_ABOVE_LIMIT_DB: f32 = 7.0;
/// The same speaker's raw recordings measured -30.4 and -31.9 dB, and every
/// cleaned sample -44.4 or below, so a sample above this probably hasn't been
/// cleaned.
const RAW_SAMPLE_DB: f32 = -40.0;

/// The noise limit given in the environment, without reporting an invalid value,
/// since `book` and `retake` report it. A limit saved in an output directory
/// isn't known when a voice sample loads, so that isn't used.
fn noise_limit_given() -> Option<f32> {
    std::env::var(NOISE_LIMIT_VAR).ok().and_then(|value| quality::valid_noise_limit(&value))
}

/// What a voice sample's floor means for its takes. It's advice, since only
/// listening can say whether takes sound clean.
///
/// The floor is judged as a level, not relative to the speech, because takes'
/// floors are measured the same way, on the takes as rendered, against the
/// noise limit.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SampleCheck {
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
    fn of(samples: &[f32], noise_limit_db: Option<f32>) -> SampleCheck {
        SampleCheck::from_floor(quality::measure(samples, SAMPLE_RATE).floor_db, noise_limit_db)
    }

    fn from_floor(floor_db: f32, noise_limit_db: Option<f32>) -> SampleCheck {
        let limit = noise_limit_db.unwrap_or(MAX_FLOOR_DB);
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
    fn summary(&self) -> String {
        let floor = self.floor_db;
        match self.verdict {
            Verdict::Fine => format!(", floor {floor:.1} dB"),
            Verdict::Noisy => format!(", floor {floor:.1} dB, noisy, see \"Noise limit\" in the README"),
            Verdict::Raw => format!(", floor {floor:.1} dB, a raw recording? See \"Adding a voice\" in the README"),
        }
    }

    /// What to do about a noisy or raw voice sample at `path`, whose recording
    /// lasts `seconds`, if anything.
    fn advice(&self, path: &Path, seconds: f32) -> Option<String> {
        let floor = self.floor_db;
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let cleaned = path.with_file_name(format!("{stem}-clean.wav"));
        let (path, cleaned) = (path.display(), cleaned.display());
        match self.verdict {
            Verdict::Fine => None,
            Verdict::Noisy => {
                let limit = match self.noise_limit_db {
                    Some(limit) => format!("the noise limit of {limit} dB set by {NOISE_LIMIT_VAR}"),
                    None => format!("the default noise limit of {MAX_FLOOR_DB} dB"),
                };
                // The next multiple of 5 dB above the floor takes are expected
                // to have, as -50 was for a -45.3 dB sample.
                let raised = ((floor - NOISY_ABOVE_LIMIT_DB) / 5.0).ceil() * 5.0;
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
                     audio-berry clean recording.mp3 <start> <end> {cleaned} --reduction 20\n\
                     See \"Noise limit\" in the README."
                ))
            }
            Verdict::Raw => {
                // Rounded down, since clean refuses an end past the recording's.
                let end = (seconds * 10.0).floor() / 10.0;
                Some(format!(
                    "The voice sample's floor is {floor:.1} dB, loud enough to hear in every take. If it's \
                     a raw recording, clean it into a new voice sample first:\n    \
                     audio-berry clean {path} 0 {end:.1} {cleaned}\n\
                     See \"Adding a voice\" in the README."
                ))
            }
        }
    }
}

fn seconds(samples: &[f32]) -> f32 {
    samples.len() as f32 / SAMPLE_RATE as f32
}

fn save(samples: &[f32], path: &Path) -> Result<(), Box<dyn Error>> {
    AudioSamples::new(samples.to_vec(), SAMPLE_RATE).save_wav(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hash_stays_the_same() {
        // Output directories save this hash, so changing it orphans every chunk.
        let samples = (0..240).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let voice = Voice::new(samples, "Hello there.");
        assert_eq!(voice.hash(), 3087178596103644096);
    }

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

    #[test]
    fn the_hash_follows_the_transcript() {
        let voice = |transcript| Voice::new(vec![0.1, 0.2], transcript);
        assert_ne!(voice("Hello there.").hash(), voice("Hello, there.").hash());
    }
}
