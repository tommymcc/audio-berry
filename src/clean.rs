//! Makes a voice sample from a noisy recording, such as one with birds or wind,
//! by cleaning it with DeepFilterNet 3 and cutting out the part to clone.

use crate::audio::decode_audio;
use any_tts::AudioSamples;
use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::{Array2, ArrayView2, Axis};
use std::error::Error;
use std::path::Path;

/// How much noise cleaning removes at most. 15 dB keeps the voice most like the
/// speaker: on one outdoor recording, cleaning harder made takes quieter but
/// dropped voice ID against the speaker's own recording from 0.85 to 0.80 at
/// 20 dB and 0.74 with no limit.
pub const DEFAULT_REDUCTION_DB: f32 = 15.0;

/// DeepFilterNet treats 100 dB as no limit at all.
pub const REDUCTION_RANGE: std::ops::RangeInclusive<f32> = 1.0..=100.0;

/// DeepFilterNet 3 works at 48 kHz.
const CLEANING_RATE: usize = 48_000;
/// The rate voice samples are kept at, as OmniVoice hears them.
const SAMPLE_RATE: usize = 24_000;

/// Cleans the whole recording, so the model has context at the edges of the cut,
/// then writes `start..end` seconds of it as a 24 kHz mono 16-bit voice sample.
pub fn clean(
    recording: &Path,
    start: f64,
    end: f64,
    output: &Path,
    reduction_db: f32,
) -> Result<(), Box<dyn Error>> {
    if end <= start {
        return Err(format!("the end ({end}s) must come after the start ({start}s)").into());
    }
    let noisy = decode_audio(recording, CLEANING_RATE as u32)?;
    let length = noisy.len() as f64 / CLEANING_RATE as f64;
    if end > length {
        return Err(format!("{} is only {length:.1}s long, so can't cut to {end}s", recording.display()).into());
    }

    let started = std::time::Instant::now();
    let cleaned = denoise(&noisy, reduction_db)?;
    let sample = cut(&halve_rate(&cleaned)?, start, end, SAMPLE_RATE);
    let took = started.elapsed().as_secs_f32();

    let sample = AudioSamples::new(sample, SAMPLE_RATE as u32);
    sample.save_wav(output)?;
    let peak = sample.samples.iter().fold(0f32, |peak, s| peak.max(s.abs()));
    println!(
        "Wrote {}: {:.2}s, peak {:.1} dB, cleaned with up to {reduction_db} dB of reduction in {took:.1}s",
        output.display(),
        sample.duration_secs(),
        20.0 * peak.max(1e-5).log10(),
    );
    println!(
        "The first narration transcribes it to {}. Check that file's words.",
        output.with_extension("txt").display()
    );
    Ok(())
}

/// Runs DeepFilterNet 3 over 48 kHz mono audio, removing at most `reduction_db`
/// of noise. The output lines up with the input sample for sample.
fn denoise(noisy: &[f32], reduction_db: f32) -> Result<Vec<f32>, Box<dyn Error>> {
    // Run every stage on every frame, as the Python version does, rather than
    // skipping frames the model judges to be pure speech or pure noise.
    let params = RuntimeParams::default_with_ch(1)
        .with_atten_lim(reduction_db)
        .with_thresholds(f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY);
    let mut model = DfTract::new(DfParams::default(), &params)?;
    assert_eq!(model.sr, CLEANING_RATE, "the built-in model's sample rate");
    let hop = model.hop_size;
    // The STFT and the model's lookahead delay the output by this much.
    let delay = model.fft_size - hop + model.lookahead * hop;

    // Feed enough after the end to flush the delay. The model passes digital
    // silence through untouched without advancing, so mirror the end instead.
    let mut input = noisy.to_vec();
    let padded = (noisy.len() + delay).div_ceil(hop) * hop;
    input.extend(noisy.iter().rev().cycle().take(padded - noisy.len()));

    let input = Array2::from_shape_vec((1, padded), input)?;
    let mut enhanced = Array2::<f32>::zeros((1, padded));
    for (frame, out) in input
        .axis_chunks_iter(Axis(1), hop)
        .zip(enhanced.axis_chunks_iter_mut(Axis(1), hop))
    {
        model.process(frame, out)?;
    }
    Ok(enhanced.row(0).iter().skip(delay).take(noisy.len()).copied().collect())
}

/// Converts 48 kHz audio to the voice sample rate of 24 kHz.
fn halve_rate(samples: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let view = ArrayView2::from_shape((1, samples.len()), samples)?;
    let resampled = df::transforms::resample(view, CLEANING_RATE, SAMPLE_RATE, None)?;
    Ok(resampled.row(0).iter().take(samples.len() / 2).copied().collect())
}

/// The samples from `start` to `end` seconds, or as much of that as there is.
fn cut(samples: &[f32], start: f64, end: f64, sample_rate: usize) -> Vec<f32> {
    let index = |seconds: f64| ((seconds * sample_rate as f64).round() as usize).min(samples.len());
    samples[index(start)..index(end).max(index(start))].to_vec()
}

/// Reads a time in a recording the way ffmpeg does: seconds such as `20.1`,
/// optionally with an `s`, `ms` or `us` unit, or `[HH:]MM:SS[.fraction]`.
pub fn parse_time(text: &str) -> Option<f64> {
    let text = text.trim();
    let seconds = if text.contains(':') {
        let parts: Vec<&str> = text.split(':').collect();
        let (hours, minutes, seconds) = match parts.as_slice() {
            [minutes, seconds] => ("0", *minutes, *seconds),
            [hours, minutes, seconds] => (*hours, *minutes, *seconds),
            _ => return None,
        };
        let whole = |part: &str| part.bytes().all(|b| b.is_ascii_digit()).then(|| part.parse::<f64>().ok())?;
        let (hours, minutes) = (whole(hours)?, whole(minutes)?);
        let seconds = plain_number(seconds)?;
        if minutes >= 60.0 || seconds >= 60.0 {
            return None;
        }
        hours * 3600.0 + minutes * 60.0 + seconds
    } else if let Some(number) = text.strip_suffix("ms") {
        plain_number(number)? / 1e3
    } else if let Some(number) = text.strip_suffix("us") {
        plain_number(number)? / 1e6
    } else {
        plain_number(text.strip_suffix('s').unwrap_or(text))?
    };
    seconds.is_finite().then_some(seconds)
}

/// A number of digits with an optional fraction, such as `20` or `20.1`.
fn plain_number(text: &str) -> Option<f64> {
    let valid = !text.is_empty()
        && text.bytes().all(|b| b.is_ascii_digit() || b == b'.')
        && text.bytes().any(|b| b.is_ascii_digit());
    valid.then(|| text.parse().ok())?
}

/// A noise reduction level from text, if it's within `REDUCTION_RANGE`.
pub fn parse_reduction(text: &str) -> Option<f32> {
    text.trim().parse::<f32>().ok().filter(|db| REDUCTION_RANGE.contains(db))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_are_read_like_ffmpeg_reads_them() {
        for (text, seconds) in [
            ("20.1", 20.1),
            ("30", 30.0),
            (".5", 0.5),
            ("20.1s", 20.1),
            ("1500ms", 1.5),
            ("250000us", 0.25),
            ("1:05", 65.0),
            ("01:05.5", 65.5),
            ("1:02:03", 3723.0),
            ("0:00:20.1", 20.1),
        ] {
            let parsed = parse_time(text).unwrap_or_else(|| panic!("{text} didn't parse"));
            assert!((parsed - seconds).abs() < 1e-9, "{text} gave {parsed}");
        }
    }

    #[test]
    fn nonsense_times_are_refused() {
        for text in ["", "s", "-5", "1e3", "ten", "1:2:3:4", "1:60", "1:-5", "1.5:00", "0:61", "inf", "NaN", "20 s"] {
            assert_eq!(parse_time(text), None, "{text}");
        }
    }

    #[test]
    fn reduction_must_be_from_1_to_100_db() {
        assert_eq!(parse_reduction("15"), Some(15.0));
        assert_eq!(parse_reduction("100"), Some(100.0));
        for bad in ["0", "0.5", "-15", "101", "lots", "NaN"] {
            assert_eq!(parse_reduction(bad), None, "{bad}");
        }
    }

    #[test]
    fn cutting_takes_start_to_end() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert_eq!(cut(&samples, 0.2, 0.5, 100), samples[20..50]);
        assert_eq!(cut(&samples, 0.9, 2.0, 100), samples[90..]);
        assert!(cut(&samples, 2.0, 3.0, 100).is_empty());
    }

    #[test]
    fn halving_the_rate_keeps_a_voice_band_tone_and_drops_what_24_khz_cant_hold() {
        let tone = |hz: f32| -> Vec<f32> {
            (0..CLEANING_RATE)
                .map(|i| 0.5 * (std::f32::consts::TAU * hz * i as f32 / CLEANING_RATE as f32).sin())
                .collect()
        };
        let rms = |samples: &[f32]| (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        let voice = halve_rate(&tone(440.0)).unwrap();
        assert_eq!(voice.len(), SAMPLE_RATE);
        // Ignore the edges, where the filter starts and stops.
        let middle = &voice[1000..SAMPLE_RATE - 1000];
        assert!((rms(middle) - 0.5 / 2f32.sqrt()).abs() < 0.01, "{}", rms(middle));
        // The tone stays in time: compare with a 440 Hz tone made at 24 kHz.
        let expected: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| 0.5 * (std::f32::consts::TAU * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let error = rms(&middle.iter().zip(&expected[1000..]).map(|(a, b)| a - b).collect::<Vec<_>>());
        assert!(error < 0.01, "{error}");

        let too_high = halve_rate(&tone(15_000.0)).unwrap();
        assert!(rms(&too_high[1000..SAMPLE_RATE - 1000]) < 0.01);
    }
}
