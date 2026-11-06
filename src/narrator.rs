//! The text-to-speech model, which says text in a cloned voice.

use crate::voice::Voice;
use any_tts::models::omnivoice::preferred_runtime_choice;
use any_tts::{AudioSamples, ModelType, SynthesisRequest, TtsConfig, TtsModel, load_model};
use std::error::Error;
use std::path::Path;
use std::time::Instant;

pub struct Narrator {
    model: Box<dyn TtsModel>,
}

impl Narrator {
    pub fn load() -> Result<Narrator, Box<dyn Error>> {
        let started = Instant::now();
        let model = load_model(TtsConfig::new(ModelType::OmniVoice).with_preferred_runtime())?;
        println!("Model loaded in {:.0}s on {}", started.elapsed().as_secs_f32(), preferred_runtime_choice().device.label());
        Ok(Narrator { model })
    }

    /// Says `text` in `voice` at `speed`. A seed picks a repeatable take; without
    /// one, the model picks it from `OMNIVOICE_SEED`, as it does by default.
    pub fn say(&self, voice: &Voice, text: &str, speed: f64, seed: Option<u64>) -> Result<AudioSamples, Box<dyn Error>> {
        if let Some(seed) = seed {
            // SAFETY: nothing else reads or writes the environment during this
            // call, as the model is only used from this thread between requests.
            unsafe { std::env::set_var("OMNIVOICE_SEED", seed.to_string()) };
        }
        let mut request = SynthesisRequest::new(text).with_reference_audio(voice.reference().clone());
        // The model treats no speed as normal speed.
        if speed != 1.0 {
            request = request.with_speed(speed);
        }
        Ok(self.model.synthesize(&request)?)
    }

    /// Says `text` in `voice`, saves it to `output` and reports how fast that was.
    pub fn say_to_file(&self, voice: &Voice, text: &str, output: &Path, speed: f64) -> Result<(), Box<dyn Error>> {
        let started = Instant::now();
        let audio = self.say(voice, text, speed, None)?;
        let elapsed = started.elapsed().as_secs_f32();

        audio.save_wav(output)?;
        println!(
            "Wrote {}: {:.1}s of audio in {elapsed:.0}s ({:.2}x realtime)",
            output.display(),
            audio.duration_secs(),
            audio.duration_secs() / elapsed
        );
        Ok(())
    }
}

/// The `speak` command: says `text` in the voice sample's voice and saves it.
pub fn speak(voice: &Path, text: &str, output: &Path, speed: Option<f64>) -> Result<(), Box<dyn Error>> {
    let voice = Voice::load(voice)?;
    Narrator::load()?.say_to_file(&voice, text, output, speed.unwrap_or(1.0))
}
