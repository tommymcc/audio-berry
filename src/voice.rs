//! The voice sample a narration is cloned from, and what's said in it.

use crate::audio::decode_audio;
use crate::store::fnv;
use crate::transcribe::{Transcriber, VOICE_TRANSCRIPTION_MODEL};
use any_tts::models::omnivoice::trim_long_audio;
use any_tts::{AudioSamples, ReferenceAudio};
use std::error::Error;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

const SAMPLE_RATE: u32 = 24_000;
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
    fn the_hash_follows_the_transcript() {
        let voice = |transcript| Voice::new(vec![0.1, 0.2], transcript);
        assert_ne!(voice("Hello there.").hash(), voice("Hello, there.").hash());
    }
}
