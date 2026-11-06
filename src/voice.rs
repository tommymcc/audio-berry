//! The voice sample a narration is cloned from, and what's said in it.

use crate::audio::decode_audio;
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
        println!("Voice sample: using {:.1}s", seconds(&samples));

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

        let clip = transcript_path.with_extension("transcribing.wav");
        save(&samples, &clip)?;
        let transcript = Transcriber::start(VOICE_TRANSCRIPTION_MODEL)
            .and_then(|mut transcriber| transcriber.transcribe(&clip));
        fs::remove_file(&clip)?;
        let transcript = transcript?;
        if transcript.is_empty() {
            return Err(format!("heard no speech in {}", path.display()).into());
        }
        fs::write(&transcript_path, format!("{transcript}\n"))?;
        println!("Heard in the voice sample: {transcript}");
        println!("Saved to {}. Correct any mistakes there.", transcript_path.display());
        println!("{}", SampleCheck::of(&samples).advice(path));
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

/// Takes copy the voice sample's background, a few dB quieter. On one speaker,
/// samples cleaned to floors of -45.3, -48.9 and -58.0 dB gave takes with floors
/// of about -53.6, -55.7 and -59.3, so a sample above about -50 dB gives takes
/// that fail the default noise limit of -55, or only just pass it.
const NOISY_SAMPLE_DB: f32 = -50.0;
/// The same speaker's uncleaned recordings measured -30.4 and -31.9 dB, and
/// every cleaned one -44.4 or below, so a sample above this probably hasn't been
/// cleaned.
const RAW_SAMPLE_DB: f32 = -40.0;

/// What a voice sample's background means for its takes, judged on first use.
/// It's advice, since only listening can say whether takes sound clean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SampleCheck {
    /// Takes should pass the default noise limit.
    Quiet { floor_db: f32 },
    /// Faint background, such as a cleaned outdoor recording: most takes will
    /// fail the default noise limit.
    Noisy { floor_db: f32 },
    /// Loud background, probably a recording that hasn't been cleaned.
    Raw { floor_db: f32 },
}

impl SampleCheck {
    /// Judges a voice sample by its floor, measured as for takes.
    pub fn of(samples: &[f32]) -> SampleCheck {
        SampleCheck::from_floor(quality::measure(samples, SAMPLE_RATE).floor_db)
    }

    fn from_floor(floor_db: f32) -> SampleCheck {
        if floor_db > RAW_SAMPLE_DB {
            SampleCheck::Raw { floor_db }
        } else if floor_db > NOISY_SAMPLE_DB {
            SampleCheck::Noisy { floor_db }
        } else {
            SampleCheck::Quiet { floor_db }
        }
    }

    /// What to tell someone about to narrate with the sample at `path`.
    pub fn advice(&self, path: &Path) -> String {
        let path = path.display();
        let clean = "./target/release/audio-berry clean recording.mp3 <start> <end>";
        match *self {
            SampleCheck::Quiet { floor_db } => {
                format!("Voice sample background: {floor_db:.1} dB, fine for the default noise limit.")
            }
            SampleCheck::Noisy { floor_db } => format!(
                "Voice sample background: {floor_db:.1} dB. Takes copy it, a few dB quieter, so many \
                 will measure above the default noise limit of {MAX_FLOOR_DB:.0} dB and be rendered \
                 again, up to 5 times each. Either:\n\
                 - keep this sample and raise the limit, if test takes sound clean:\n    \
                 {NOISE_LIMIT_VAR}=-50 ./target/release/audio-berry book {path} book.epub output/book\n\
                 - or clean the recording with a higher reduction than before (15 dB by default), \
                 which makes the clone less like the speaker:\n    \
                 {clean} {path} --reduction 20\n\
                 See \"Noise limit\" in the README."
            ),
            SampleCheck::Raw { floor_db } => format!(
                "Voice sample background: {floor_db:.1} dB, loud enough to hear in every take. If it's an \
                 uncleaned recording, clean it first:\n    \
                 {clean} {path}\n\
                 See \"Adding a voice\" in the README."
            ),
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
    fn a_sample_is_judged_by_its_floor() {
        assert_eq!(SampleCheck::from_floor(-35.0), SampleCheck::Raw { floor_db: -35.0 });
        // Samples cleaned at 15 and 20 dB, whose takes failed or only just passed.
        assert_eq!(SampleCheck::from_floor(-45.3), SampleCheck::Noisy { floor_db: -45.3 });
        assert_eq!(SampleCheck::from_floor(-48.9), SampleCheck::Noisy { floor_db: -48.9 });
        // Cleaned at 30 dB and without limit, whose takes passed.
        assert_eq!(SampleCheck::from_floor(-58.0), SampleCheck::Quiet { floor_db: -58.0 });
        assert_eq!(SampleCheck::from_floor(-80.7), SampleCheck::Quiet { floor_db: -80.7 });
    }

    #[test]
    fn a_sample_check_measures_the_quiet_frames() {
        let speech: Vec<f32> = (0..24_000 * 8).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let with_background = |level: f32| {
            let mut samples = speech.clone();
            samples.extend((0..24_000).map(|i| (i as f32 * 0.07).sin() * level));
            SampleCheck::of(&samples)
        };
        assert!(matches!(with_background(0.0), SampleCheck::Quiet { .. }));
        assert!(matches!(with_background(0.008), SampleCheck::Noisy { .. }));
        assert!(matches!(with_background(0.05), SampleCheck::Raw { .. }));
    }

    #[test]
    fn noisy_sample_advice_gives_both_ways_out() {
        let advice = SampleCheck::Noisy { floor_db: -45.3 }.advice(Path::new("voices/dad.wav"));
        assert!(advice.contains("-45.3 dB"), "{advice}");
        assert!(advice.contains("AUDIO_BERRY_NOISE_LIMIT_DB=-50 ./target/release/audio-berry book voices/dad.wav"));
        assert!(advice.contains("voices/dad.wav --reduction 20"));
        assert!(advice.contains("\"Noise limit\""));
    }

    #[test]
    fn a_quiet_sample_gets_one_line() {
        let advice = SampleCheck::Quiet { floor_db: -66.6 }.advice(Path::new("voices/me.wav"));
        assert_eq!(advice.lines().count(), 1);
        assert!(advice.contains("-66.6 dB"));
    }

    #[test]
    fn the_hash_follows_the_transcript() {
        let voice = |transcript| Voice::new(vec![0.1, 0.2], transcript);
        assert_ne!(voice("Hello there.").hash(), voice("Hello, there.").hash());
    }
}
