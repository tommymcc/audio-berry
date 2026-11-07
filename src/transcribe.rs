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
use color_eyre::eyre::{Result, eyre};
use fs_err as fs;
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
/// Timestamp tokens count in fiftieths of a second.
const TIMESTAMP_SECONDS: f64 = 0.02;
/// Whisper's own limit on where a window's first phrase may start, in timestamp tokens.
const MAX_INITIAL_TIMESTAMP: usize = 50;

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
    /// What was decoded after the prompt, and each token's log probability.
    tokens: Vec<u32>,
    logprobs: Vec<f64>,
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

fn compression_ratio(text: &str) -> Result<f64> {
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
    /// The first timestamp token, `<|0.00|>`. Every later token is a timestamp.
    timestamp_begin: u32,
    /// Added to the logits at every step, to rule out tokens that aren't speech.
    suppress: Tensor,
    /// Added on the first step as well, so a transcript can't start blank.
    suppress_first: Tensor,
    /// The same as `suppress`, but allowing timestamps.
    suppress_timed: Tensor,
}

impl Transcriber {
    pub fn start(model: &str) -> Result<Self> {
        let repo = format!("openai/whisper-{model}");
        let config: Config = serde_json::from_slice(&fs::read(download_file(&repo, "config.json")?)?)?;
        let tokenizer = Tokenizer::from_file(download_file(&repo, "tokenizer.json")?)
            .map_err(|error| eyre!("couldn't read the {model} tokenizer: {error}"))?;
        let weights = download_file(&repo, "model.safetensors")?;
        let device = DeviceSelection::Auto.resolve()?;
        // Safety: the cached weights aren't changed while the app is running.
        let weights = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], whisper::DTYPE, &device)? };
        let filters = mel_filters(config.num_mel_bins);
        let token = |name: &str| {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| eyre!("the {model} tokenizer has no {name}"))
        };
        let (start, no_timestamps) = (token(whisper::SOT_TOKEN)?, token(whisper::NO_TIMESTAMPS_TOKEN)?);
        let (end, blank) = (token(whisper::EOT_TOKEN)?, token("Ġ")?);
        let timestamp_begin = token("<|0.00|>")?;
        let no_speech = whisper::NO_SPEECH_TOKENS
            .iter()
            .find_map(|name| tokenizer.token_to_id(name))
            .ok_or_else(|| eyre!("the {model} tokenizer has no no-speech token"))?;

        let mut mask = vec![0f32; config.vocab_size];
        for &id in &config.suppress_tokens {
            mask[id as usize] = f32::NEG_INFINITY;
        }
        // Every special token comes after the end of text, and timestamps after the rest.
        mask[end as usize + 1..timestamp_begin as usize].fill(f32::NEG_INFINITY);
        let suppress_timed = Tensor::from_slice(&mask, mask.len(), &device)?;
        mask[timestamp_begin as usize..].fill(f32::NEG_INFINITY);
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
            timestamp_begin,
            suppress,
            suppress_first,
            suppress_timed,
        })
    }

    /// Decodes a window, with timestamps between its phrases if `timed`.
    fn decode(&mut self, features: &Tensor, temperature: f64, timed: bool) -> Result<Attempt> {
        let temperature = (temperature > 0.0).then_some(temperature);
        let mut sampler = LogitsProcessor::new(SAMPLING_SEED, temperature, None);
        let mut tokens = if timed { vec![self.start] } else { vec![self.start, self.no_timestamps] };
        let prompt = tokens.len();
        let limit = self.model.config.max_target_positions / 2;
        let mut logprobs = Vec::new();
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
            let mask = match (timed, step) {
                (true, _) => &self.suppress_timed,
                (false, 0) => &self.suppress_first,
                (false, _) => &self.suppress,
            };
            let mut logits = (logits + mask)?;
            if timed {
                let mut values = logits.to_vec1::<f32>()?;
                apply_timestamp_rules(&mut values, &tokens[prompt..], self.end, self.timestamp_begin);
                logits = Tensor::from_vec(values, logits.dims1()?, &self.device)?;
            }
            let token = sampler.sample(&logits)?;
            logprobs.push(f64::from(log_softmax(&logits, D::Minus1)?.i(token as usize)?.to_scalar::<f32>()?));
            if token == self.end {
                break;
            }
            tokens.push(token);
        }
        let tokens = tokens.split_off(prompt);
        let text = self.words(&tokens)?;
        Ok(Attempt {
            compression_ratio: compression_ratio(&text)?,
            average_logprob: logprobs.iter().sum::<f64>() / (tokens.len() + 1) as f64,
            no_speech,
            text,
            tokens,
            logprobs,
        })
    }

    /// The words in `tokens`, leaving out timestamps, which the tokenizer
    /// doesn't count as special.
    fn words(&self, tokens: &[u32]) -> Result<String> {
        let words: Vec<u32> = tokens.iter().copied().filter(|&token| token < self.timestamp_begin).collect();
        let text = self
            .tokenizer
            .decode(&words, true)
            .map_err(|error| eyre!("couldn't decode a transcript: {error}"))?;
        Ok(text)
    }

    /// The best decoding of a window, or `None` if it seems silent.
    fn transcribe_window(&mut self, mel: &Tensor, timed: bool) -> Result<Option<Attempt>> {
        let features = self.model.encoder.forward(mel, true)?;
        let mut attempts = Vec::new();
        for temperature in whisper::TEMPERATURES {
            let attempt = self.decode(&features, temperature, timed)?;
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
        Ok(best.filter(|attempt| !attempt.silent()))
    }

    /// Decodes `audio`, heard as it would be once saved as a WAV file, window by
    /// window, giving each window that isn't silent.
    fn transcribe_windows(&mut self, audio: &AudioSamples, timed: bool) -> Result<Vec<Heard>> {
        let samples = resample_as_saved(audio, whisper::SAMPLE_RATE as u32)?;
        let bands = self.model.config.num_mel_bins;
        let seconds = |samples: usize| samples as f64 / whisper::SAMPLE_RATE as f64;
        let mut start = 0;
        let mut windows = Vec::new();
        while samples.len() - start >= whisper::HOP_LENGTH {
            let window = &samples[start..start + window_end(&samples[start..])];
            // Padded with silence, so there are always enough frames.
            let mel = audio::pcm_to_mel(&self.model.config, window, &self.filters);
            let frames = mel.len() / bands;
            let mel = Tensor::from_vec(mel, (1, bands, frames), &self.device)?;
            if let Some(attempt) = self.transcribe_window(&mel.narrow(2, 0, whisper::N_FRAMES)?, timed)? {
                windows.push(Heard { start: seconds(start), seconds: seconds(window.len()), attempt });
            }
            start += window.len();
        }
        Ok(windows)
    }

    /// What's said in `audio`, heard as it would be once saved as a WAV file.
    pub fn transcribe(&mut self, audio: &AudioSamples) -> Result<String> {
        let text: Vec<String> = self.transcribe_windows(audio, false)?.into_iter().map(|heard| heard.attempt.text).collect();
        Ok(tidy(&text.join(" ")))
    }

    /// What's said in `audio`, phrase by phrase, with when each phrase was said.
    /// Whisper places its timestamps roughly, sometimes a few tenths of a second
    /// out, so look for the pauses near them before cutting there.
    pub fn transcribe_phrases(&mut self, audio: &AudioSamples) -> Result<Vec<Phrase>> {
        let mut phrases = Vec::new();
        for Heard { start: offset, seconds, attempt } in self.transcribe_windows(audio, true)? {
            for span in spans(&attempt.tokens, &attempt.logprobs, self.timestamp_begin, seconds) {
                let text = tidy(&self.words(&attempt.tokens[span.tokens])?);
                if !text.is_empty() {
                    phrases.push(Phrase {
                        start: offset + span.start,
                        end: offset + span.end,
                        text,
                        average_logprob: span.average_logprob,
                    });
                }
            }
        }
        Ok(phrases)
    }
}

/// A window of audio and its best decoding.
struct Heard {
    /// When the window starts in the audio, and how long it is, in seconds.
    start: f64,
    seconds: f64,
    attempt: Attempt,
}

/// Text with single spaces between words, and none at either end.
fn tidy(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A stretch of speech Whisper heard as one, usually a sentence or a clause.
#[derive(Clone, Debug, PartialEq)]
pub struct Phrase {
    /// Seconds from the start of the audio.
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// How sure Whisper was of the words: about -0.2 for clear speech, and -1 or
    /// less for speech it could barely make out.
    pub average_logprob: f64,
}

/// Where a phrase lies in a window's decoded tokens.
#[derive(Debug, PartialEq)]
struct Span {
    start: f64,
    end: f64,
    tokens: std::ops::Range<usize>,
    average_logprob: f64,
}

/// Splits a window's decoded tokens into phrases at their timestamps. Whisper
/// puts a timestamp either side of each phrase, and may leave the last one open,
/// which then runs to the end of the window. Times past the end, where Whisper
/// heard only padding, are kept to it.
fn spans(tokens: &[u32], logprobs: &[f64], timestamp_begin: u32, window_seconds: f64) -> Vec<Span> {
    let seconds = |token: u32| (f64::from(token - timestamp_begin) * TIMESTAMP_SECONDS).min(window_seconds);
    let mut spans = Vec::new();
    let mut close = |start: f64, words: std::ops::Range<usize>, end: f64| {
        if !words.is_empty() {
            let logprobs = &logprobs[words.clone()];
            let average_logprob = logprobs.iter().sum::<f64>() / logprobs.len() as f64;
            spans.push(Span { start, end: end.max(start), tokens: words, average_logprob });
        }
    };
    // The open phrase's start, and where its words begin.
    let mut open: Option<(f64, usize)> = None;
    for (index, &token) in tokens.iter().enumerate() {
        if token < timestamp_begin {
            continue;
        }
        match open.take() {
            Some((start, first)) if first < index => close(start, first..index, seconds(token)),
            _ => open = Some((seconds(token), index + 1)),
        }
    }
    if let Some((start, first)) = open {
        close(start, first..tokens.len(), window_seconds);
    }
    spans
}

/// Whisper's rules for where timestamps may go, applied to the logits for the
/// token after `decoded`: a window starts with a timestamp in its first second,
/// timestamps come in pairs between phrases and never go back, and a timestamp
/// is chosen whenever timestamps together are likelier than any one word.
fn apply_timestamp_rules(logits: &mut [f32], decoded: &[u32], end: u32, timestamp_begin: u32) {
    let (end, begin, vocab) = (end as usize, timestamp_begin as usize, logits.len());
    let is_timestamp = |token: &u32| *token >= timestamp_begin;
    let last_was_timestamp = decoded.last().is_some_and(is_timestamp);
    let penultimate_was_timestamp = decoded.len() < 2 || is_timestamp(&decoded[decoded.len() - 2]);
    if last_was_timestamp {
        if penultimate_was_timestamp {
            // A phrase has just opened, so words follow.
            logits[begin..].fill(f32::NEG_INFINITY);
        } else {
            // A phrase has just closed, so the next opens or the window ends.
            logits[..end].fill(f32::NEG_INFINITY);
        }
    }
    if let Some(&latest) = decoded.iter().rev().find(|token| is_timestamp(token)) {
        // The next phrase may open where the last closed, but time never goes back.
        let earliest = if last_was_timestamp && !penultimate_was_timestamp { latest } else { latest + 1 };
        logits[begin..(earliest as usize).min(vocab)].fill(f32::NEG_INFINITY);
    }
    if decoded.is_empty() {
        logits[..begin].fill(f32::NEG_INFINITY);
        logits[(begin + MAX_INITIAL_TIMESTAMP + 1).min(vocab)..].fill(f32::NEG_INFINITY);
    }
    let log_sum_exp = |values: &[f32]| {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max == f32::NEG_INFINITY {
            return max;
        }
        max + values.iter().map(|value| (value - max).exp()).sum::<f32>().ln()
    };
    let best_word = logits[..begin].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if log_sum_exp(&logits[begin..]) > best_word {
        logits[..begin].fill(f32::NEG_INFINITY);
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
            tokens: Vec::new(),
            logprobs: Vec::new(),
            average_logprob,
            no_speech,
            compression_ratio: 1.0,
        };
        assert!(attempt(-0.2, 0.1).acceptable());
        assert!(!attempt(-2.0, 0.1).acceptable());
        assert!(attempt(-2.0, 0.9).acceptable() && attempt(-2.0, 0.9).silent());
        assert!(!attempt(-0.2, 0.9).silent());
    }

    /// Token ids as in the English Whisper models.
    const END: u32 = 50256;
    const BEGIN: u32 = 50363;
    const VOCAB: usize = 51864;

    fn at(seconds: f64) -> u32 {
        BEGIN + (seconds / TIMESTAMP_SECONDS).round() as u32
    }

    #[test]
    fn phrases_are_split_at_their_timestamps() {
        // <|0.00|> Hello there. <|1.50|><|1.80|> How are you <|3.00|><|3.00|> I'm fine
        let tokens = [at(0.0), 10, 11, at(1.5), at(1.8), 20, 21, 22, at(3.0), at(3.0), 30];
        let logprobs = [-0.1, -0.2, -0.3, -0.1, -0.1, -0.4, -0.5, -0.6, -0.1, -0.1, -0.9];
        let spans = spans(&tokens, &logprobs, BEGIN, 4.2);
        assert_eq!(spans.len(), 3);
        assert_eq!((spans[0].start, spans[0].end, spans[0].tokens.clone()), (0.0, 1.5, 1..3));
        assert!((spans[0].average_logprob + 0.25).abs() < 1e-9);
        assert_eq!((spans[1].start, spans[1].end, spans[1].tokens.clone()), (1.8, 3.0, 5..8));
        // Left open, so it runs to the end of the window.
        assert_eq!((spans[2].start, spans[2].end, spans[2].tokens.clone()), (3.0, 4.2, 10..11));
    }

    #[test]
    fn timestamps_past_the_window_are_kept_to_it_and_empty_phrases_dropped() {
        let tokens = [at(0.5), 10, at(29.0), at(29.0)];
        let spans = spans(&tokens, &[0.0; 4], BEGIN, 12.0);
        assert_eq!(spans.len(), 1);
        assert_eq!((spans[0].start, spans[0].end), (0.5, 12.0));
        assert!(super::spans(&[], &[], BEGIN, 12.0).is_empty());
    }

    /// The tokens the rules allow after `decoded`, given logits that favour `favoured`.
    fn allowed(decoded: &[u32], favoured: u32) -> Vec<u32> {
        let mut logits = vec![0f32; VOCAB];
        logits[favoured as usize] = 20.0;
        apply_timestamp_rules(&mut logits, decoded, END, BEGIN);
        (0..VOCAB as u32).filter(|&token| logits[token as usize] > f32::NEG_INFINITY).collect()
    }

    #[test]
    fn a_window_opens_with_a_timestamp_in_its_first_second() {
        let first = allowed(&[], 10);
        assert_eq!(first, (at(0.0)..=at(1.0)).collect::<Vec<_>>());
    }

    #[test]
    fn timestamps_come_in_pairs_and_never_go_back() {
        // Just opened: words only.
        assert!(allowed(&[at(0.4)], 10).iter().all(|&token| token < BEGIN));
        // Just closed at 2.0: the end, or a timestamp from 2.0 on.
        let next = allowed(&[at(0.4), 10, at(2.0)], END);
        assert!(next.contains(&END) && next.contains(&at(2.0)) && !next.contains(&at(1.98)));
        assert!(!next.contains(&10));
        // Mid-phrase: a word, the end, or a timestamp after the last.
        let next = allowed(&[at(0.4), 10, at(2.0), at(2.2), 11], 10);
        assert!(next.contains(&10) && next.contains(&END) && next.contains(&at(2.22)) && !next.contains(&at(2.2)));
    }

    #[test]
    fn a_timestamp_is_forced_when_timestamps_together_outweigh_any_word() {
        // Every timestamp a little likely, and no word much likelier.
        let mut logits = vec![0f32; VOCAB];
        logits[10] = 3.0;
        apply_timestamp_rules(&mut logits, &[at(0.4), 10], END, BEGIN);
        assert_eq!(logits[10], f32::NEG_INFINITY);
        assert!(logits[at(1.0) as usize] > f32::NEG_INFINITY);
        // A word far likelier than all the timestamps together stays.
        let mut logits = vec![0f32; VOCAB];
        logits[10] = 20.0;
        apply_timestamp_rules(&mut logits, &[at(0.4), 10], END, BEGIN);
        assert_eq!(logits[10], 20.0);
    }
}
