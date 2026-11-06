//! The voice sample a narration is cloned from, and what's said in it.

use crate::accuracy::Allowances;
use crate::audio::decode_audio;
use crate::quality::noise_limit_from_env_quietly;
use crate::sample_check::SampleCheck;
use crate::store::fnv;
use crate::transcribe::{Transcriber, VOICE_TRANSCRIPTION_MODEL};
use any_tts::models::omnivoice::trim_long_audio;
use any_tts::{AudioSamples, ReferenceAudio};
use std::error::Error;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

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
    /// Not part of the voice: they only decide whether a take was heard right.
    allowances: Allowances,
}

impl Voice {
    /// Loads a voice sample and what's said in it. The text lives in a `.txt` file
    /// beside the sample, written by transcribing it the first time, and can be
    /// corrected by hand. Any allowances are in another file beside it.
    pub fn load(path: &Path) -> Result<Voice, Box<dyn Error>> {
        let recording = decode_audio(path, SAMPLE_RATE)?;
        let samples = trim_long_audio(
            &recording,
            SAMPLE_RATE,
            MAX_VOICE_SECONDS,
            MIN_VOICE_SECONDS,
            TRIM_VOICE_ABOVE_SECONDS,
        );
        let check = SampleCheck::of(&samples, noise_limit_from_env_quietly());
        println!("Voice sample: using {:.1}s{}", seconds(&samples), check.summary());
        let allowances = read_allowances(path)?;

        let transcript_path = transcript_path(path);
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
            return Ok(Voice { allowances, ..Voice::new(samples, existing) });
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
        Ok(Voice { allowances, ..Voice::new(samples, &transcript) })
    }

    fn new(samples: Vec<f32>, transcript: &str) -> Voice {
        Voice {
            reference: ReferenceAudio::new(samples, SAMPLE_RATE).with_transcript(transcript),
            allowances: Allowances::default(),
        }
    }

    /// Identifies the voice, including its transcript since that changes the speech.
    /// Output directories save it, so the same sample must always give the same hash.
    /// Allowances aren't included, since they don't change the speech.
    pub fn hash(&self) -> u64 {
        let mut bytes: Vec<u8> = self.reference.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        bytes.extend_from_slice(self.transcript().as_bytes());
        fnv(&bytes, 0)
    }

    /// The sample and transcript in the form the model takes.
    pub fn reference(&self) -> &ReferenceAudio {
        &self.reference
    }

    /// Words this voice may be heard saying as others, such as "in" heard as "and".
    pub fn allowances(&self) -> &Allowances {
        &self.allowances
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

/// Where what's said in the voice sample at `sample` is kept, beside it.
pub fn transcript_path(sample: &Path) -> PathBuf {
    sample.with_extension("txt")
}

/// Where the words a voice may be heard saying as others are kept, beside the
/// voice sample at `sample`, such as `voices/me.allowances` for `voices/me.wav`.
pub fn allowances_path(sample: &Path) -> PathBuf {
    sample.with_extension("allowances")
}

/// Reads the allowances beside a voice sample. Most voices have none.
fn read_allowances(sample: &Path) -> Result<Allowances, Box<dyn Error>> {
    let path = allowances_path(sample);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Allowances::default()),
        Err(error) => return Err(format!("{}: {error}", path.display()).into()),
    };
    let allowances = Allowances::parse(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    println!("Allowances: {} from {}", allowances.len(), path.display());
    Ok(allowances)
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
    fn allowances_leave_the_hash_alone() {
        let samples = (0..240).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let allowances = Allowances::parse("in = and").unwrap();
        let voice = Voice { allowances, ..Voice::new(samples, "Hello there.") };
        assert_eq!(voice.allowances().len(), 1);
        assert_eq!(voice.hash(), 3087178596103644096);
    }

    #[test]
    fn allowances_are_kept_beside_the_voice_sample() {
        assert_eq!(allowances_path(Path::new("voices/me.wav")), Path::new("voices/me.allowances"));
    }

    #[test]
    fn no_allowances_file_means_none_and_a_bad_one_names_the_file_and_line() {
        let dir = std::env::temp_dir().join(format!("audio-berry-voice-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let sample = dir.join("me.wav");
        let path = allowances_path(&sample);
        let missing = read_allowances(&sample).map(|allowances| allowances.len());
        fs::write(&path, "in = and\nat had\n").unwrap();
        let bad = read_allowances(&sample).map(|allowances| allowances.len());
        fs::write(&path, "in = and  # his accent\n").unwrap();
        let good = read_allowances(&sample).map(|allowances| allowances.allows("in", "and"));
        fs::remove_dir_all(&dir).unwrap();
        assert_eq!(missing.unwrap(), 0);
        let wanted = format!("{}: line 2: \"at had\" isn't in the form \"in = and\"", path.display());
        assert_eq!(bad.unwrap_err().to_string(), wanted);
        assert!(good.unwrap());
    }

    #[test]
    fn the_hash_follows_the_transcript() {
        let voice = |transcript| Voice::new(vec![0.1, 0.2], transcript);
        assert_ne!(voice("Hello there.").hash(), voice("Hello, there.").hash());
    }
}
