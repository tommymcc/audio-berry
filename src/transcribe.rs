//! Hears what's said in audio with a local Whisper model, so generated speech can
//! be checked against its text, because the model sometimes produces fluent
//! nonsense that sounds clean and runs to a plausible length.

use crate::audio::resample_as_saved;
use any_tts::{AudioSamples, DeviceSelection};
use any_tts::download::download_file;
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_nn::ops::{log_softmax, softmax};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::whisper::{self, Config, audio, model::Whisper};
use flate2::Compression;
use flate2::write::ZlibEncoder;
use std::error::Error;
use std::fs;
use std::io::Write;
use tokenizers::Tokenizer;

/// Fast enough to check every take, at the cost of some misheard words.
pub const CHECK_MODEL: &str = "base.en";
/// A voice's transcript is only written once and every take depends on it.
pub const VOICE_TRANSCRIPTION_MODEL: &str = "medium.en";

/// Only used when a window is retried, so transcripts stay repeatable.
const SAMPLING_SEED: u64 = 42;
/// A fifth of a second at Whisper's sample rate.
const PAUSE_SAMPLES: usize = whisper::SAMPLE_RATE / 5;

/// Matches Whisper's own mel scale, which is linear below 1 kHz.
fn hz_to_mel(hz: f64) -> f64 {
    if hz < 1000.0 {
        3.0 * hz / 200.0
    } else {
        15.0 + 27.0 * (hz / 1000.0).ln() / 6.4f64.ln()
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15.0 {
        200.0 * mel / 3.0
    } else {
        1000.0 * (6.4f64.ln() * (mel - 15.0) / 27.0).exp()
    }
}

/// Overlapping triangles over the FFT bins, one row per mel band.
fn mel_filters(bands: usize) -> Vec<f32> {
    let bins = whisper::N_FFT / 2 + 1;
    let top = hz_to_mel(whisper::SAMPLE_RATE as f64 / 2.0);
    let edges: Vec<f64> = (0..bands + 2)
        .map(|index| mel_to_hz(top * index as f64 / (bands + 1) as f64))
        .collect();
    let mut filters = Vec::with_capacity(bands * bins);
    for edge in edges.windows(3) {
        let (low, middle, high) = (edge[0], edge[1], edge[2]);
        for bin in 0..bins {
            let hz = bin as f64 * whisper::SAMPLE_RATE as f64 / whisper::N_FFT as f64;
            let rise = (hz - low) / (middle - low);
            let fall = (high - hz) / (high - middle);
            let height = rise.min(fall).max(0.0);
            filters.push((height * 2.0 / (high - low)) as f32);
        }
    }
    filters
}

/// Where to end a window of audio that starts `samples`. Whisper skips words at
/// the start of a window that opens mid-phrase, so long audio is cut at its
/// quietest moment in the window's last ten seconds.
fn window_end(samples: &[f32]) -> usize {
    if samples.len() <= whisper::N_SAMPLES {
        return samples.len();
    }
    let search_from = whisper::N_SAMPLES - 10 * whisper::SAMPLE_RATE;
    let loudness = |frame: &[f32]| frame.iter().map(|sample| sample * sample).sum::<f32>();
    let quietest = samples[search_from..whisper::N_SAMPLES]
        .chunks_exact(PAUSE_SAMPLES)
        .enumerate()
        .min_by(|(_, a), (_, b)| loudness(a).total_cmp(&loudness(b)))
        .map_or(0, |(index, _)| index);
    search_from + quietest * PAUSE_SAMPLES + PAUSE_SAMPLES / 2
}

/// One decoding of a window of audio, with what Whisper's retry rules need.
struct Attempt {
    text: String,
    average_logprob: f64,
    no_speech: f64,
    compression_ratio: f64,
}

impl Attempt {
    fn repetitive(&self) -> bool {
        self.compression_ratio > whisper::COMPRESSION_RATIO_THRESHOLD
    }

    fn silent(&self) -> bool {
        self.no_speech > whisper::NO_SPEECH_THRESHOLD && self.average_logprob < whisper::LOGPROB_THRESHOLD
    }

    fn acceptable(&self) -> bool {
        self.silent() || (!self.repetitive() && self.average_logprob >= whisper::LOGPROB_THRESHOLD)
    }
}

fn compression_ratio(text: &str) -> Result<f64, Box<dyn Error>> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(text.as_bytes())?;
    Ok(text.len() as f64 / encoder.finish()?.len() as f64)
}

/// A loaded Whisper model. Decodes greedily, retrying a window at higher
/// temperatures when the result looks repetitive or unlikely.
pub struct Transcriber {
    model: Whisper,
    tokenizer: Tokenizer,
    device: Device,
    filters: Vec<f32>,
    start: u32,
    no_timestamps: u32,
    end: u32,
    no_speech: u32,
    /// Added to the logits at every step, to rule out tokens that aren't speech.
    suppress: Tensor,
    /// Added on the first step as well, so a transcript can't start blank.
    suppress_first: Tensor,
}

impl Transcriber {
    pub fn start(model: &str) -> Result<Self, Box<dyn Error>> {
        let repo = format!("openai/whisper-{model}");
        let config: Config = serde_json::from_slice(&fs::read(download_file(&repo, "config.json")?)?)?;
        let tokenizer = Tokenizer::from_file(download_file(&repo, "tokenizer.json")?)
            .map_err(|error| format!("couldn't read the {model} tokenizer: {error}"))?;
        let weights = download_file(&repo, "model.safetensors")?;
        let device = DeviceSelection::Auto.resolve()?;
        // Safety: the cached weights aren't changed while the app is running.
        let weights = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], whisper::DTYPE, &device)? };
        let filters = mel_filters(config.num_mel_bins);
        let token = |name: &str| {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| format!("the {model} tokenizer has no {name}"))
        };
        let (start, no_timestamps) = (token(whisper::SOT_TOKEN)?, token(whisper::NO_TIMESTAMPS_TOKEN)?);
        let (end, blank) = (token(whisper::EOT_TOKEN)?, token("Ġ")?);
        let no_speech = whisper::NO_SPEECH_TOKENS
            .iter()
            .find_map(|name| tokenizer.token_to_id(name))
            .ok_or_else(|| format!("the {model} tokenizer has no no-speech token"))?;

        let mut mask = vec![0f32; config.vocab_size];
        for &id in &config.suppress_tokens {
            mask[id as usize] = f32::NEG_INFINITY;
        }
        // Every special token comes after the end of text. Timestamps are among them.
        mask[end as usize + 1..].fill(f32::NEG_INFINITY);
        let suppress = Tensor::from_slice(&mask, mask.len(), &device)?;
        mask[blank as usize] = f32::NEG_INFINITY;
        mask[end as usize] = f32::NEG_INFINITY;
        let suppress_first = Tensor::from_slice(&mask, mask.len(), &device)?;

        Ok(Self {
            model: Whisper::load(&weights, config)?,
            tokenizer,
            device,
            filters,
            start,
            no_timestamps,
            end,
            no_speech,
            suppress,
            suppress_first,
        })
    }

    fn decode(&mut self, features: &Tensor, temperature: f64) -> Result<Attempt, Box<dyn Error>> {
        let temperature = (temperature > 0.0).then_some(temperature);
        let mut sampler = LogitsProcessor::new(SAMPLING_SEED, temperature, None);
        let mut tokens = vec![self.start, self.no_timestamps];
        let prompt = tokens.len();
        let limit = self.model.config.max_target_positions / 2;
        let mut logprob = 0.0;
        let mut no_speech = 0.0;
        for step in 0..limit {
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let hidden = self.model.decoder.forward(&input, features, step == 0)?;
            if step == 0 {
                let logits = self.model.decoder.final_linear(&hidden.narrow(1, 0, 1)?)?.squeeze(0)?.squeeze(0)?;
                no_speech = softmax(&logits, D::Minus1)?.i(self.no_speech as usize)?.to_scalar::<f32>()?.into();
            }
            let last = hidden.narrow(1, tokens.len() - 1, 1)?;
            let logits = self.model.decoder.final_linear(&last)?.squeeze(0)?.squeeze(0)?;
            let mask = if step == 0 { &self.suppress_first } else { &self.suppress };
            let logits = (logits + mask)?;
            let token = sampler.sample(&logits)?;
            logprob += f64::from(log_softmax(&logits, D::Minus1)?.i(token as usize)?.to_scalar::<f32>()?);
            if token == self.end {
                break;
            }
            tokens.push(token);
        }
        let text = self
            .tokenizer
            .decode(&tokens[prompt..], true)
            .map_err(|error| format!("couldn't decode a transcript: {error}"))?;
        Ok(Attempt {
            compression_ratio: compression_ratio(&text)?,
            average_logprob: logprob / (tokens.len() - prompt + 1) as f64,
            no_speech,
            text,
        })
    }

    fn transcribe_window(&mut self, mel: &Tensor) -> Result<String, Box<dyn Error>> {
        let features = self.model.encoder.forward(mel, true)?;
        let mut attempts = Vec::new();
        for temperature in whisper::TEMPERATURES {
            let attempt = self.decode(&features, temperature)?;
            let acceptable = attempt.acceptable();
            attempts.push(attempt);
            if acceptable {
                break;
            }
        }
        let best = attempts.into_iter().max_by(|a, b| {
            a.acceptable()
                .cmp(&b.acceptable())
                .then((!a.repetitive()).cmp(&!b.repetitive()))
                .then(a.average_logprob.total_cmp(&b.average_logprob))
        });
        Ok(best.filter(|attempt| !attempt.silent()).map(|attempt| attempt.text).unwrap_or_default())
    }

    /// What's said in `audio`, heard as it would be once saved as a WAV file.
    pub fn transcribe(&mut self, audio: &AudioSamples) -> Result<String, Box<dyn Error>> {
        let samples = resample_as_saved(audio, whisper::SAMPLE_RATE as u32)?;
        let bands = self.model.config.num_mel_bins;
        let mut rest = samples.as_slice();
        let mut text = Vec::new();
        while rest.len() >= whisper::HOP_LENGTH {
            let (window, after) = rest.split_at(window_end(rest));
            rest = after;
            // Padded with silence, so there are always enough frames.
            let mel = audio::pcm_to_mel(&self.model.config, window, &self.filters);
            let frames = mel.len() / bands;
            let mel = Tensor::from_vec(mel, (1, bands, frames), &self.device)?;
            text.push(self.transcribe_window(&mel.narrow(2, 0, whisper::N_FRAMES)?)?);
        }
        Ok(text.join(" ").split_whitespace().collect::<Vec<_>>().join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_filters_match_whisper() {
        let filters = mel_filters(80);
        assert_eq!(filters.len(), 80 * 201);
        for (index, expected) in [(1, 0.024862595), (202, 0.001990822), (203, 0.022871772)] {
            assert!((filters[index] - expected).abs() < 1e-7, "{index}: {}", filters[index]);
        }
    }

    #[test]
    fn long_audio_is_cut_at_a_pause() {
        let rate = whisper::SAMPLE_RATE;
        let mut audio = vec![0.5f32; 25 * rate];
        audio.extend(vec![0.0; rate / 2]);
        audio.extend(vec![0.5; 20 * rate]);
        let end = window_end(&audio);
        assert!((25 * rate..25 * rate + rate / 2).contains(&end), "{end}");
        assert_eq!(window_end(&audio[..10 * rate]), 10 * rate);
    }

    #[test]
    fn repeated_text_compresses_past_the_retry_threshold() {
        let threshold = whisper::COMPRESSION_RATIO_THRESHOLD;
        assert!(compression_ratio(&"I'm not sure what happened. ".repeat(20)).unwrap() > threshold);
        let sentence = "It was all very well to say drink me, but the wise little Alice was not going to do that.";
        assert!(compression_ratio(sentence).unwrap() < threshold);
    }

    #[test]
    fn unlikely_text_retries_unless_the_audio_seems_silent() {
        let attempt = |average_logprob, no_speech| Attempt {
            text: String::new(),
            average_logprob,
            no_speech,
            compression_ratio: 1.0,
        };
        assert!(attempt(-0.2, 0.1).acceptable());
        assert!(!attempt(-2.0, 0.1).acceptable());
        assert!(attempt(-2.0, 0.9).acceptable() && attempt(-2.0, 0.9).silent());
        assert!(!attempt(-0.2, 0.9).silent());
    }
}
