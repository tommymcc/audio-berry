//! Makes a voice sample from a noisy recording, such as one with birds or wind,
//! by cleaning it with DeepFilterNet 3 and cutting out the part to clone.

use crate::audio::decode_audio;
use crate::quality;
use crate::voice::SAMPLE_RATE;
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
const CLEANING_RATE: u32 = 48_000;

/// DeepFilterNet outputs silence for a hop whose mean square is below this,
/// without moving its buffers along, which would shift all later audio.
const QUIET_HOP_MEAN_SQUARE: f32 = 1e-7;

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
    let noisy = decode_audio(recording, CLEANING_RATE)?;
    let length = noisy.len() as f64 / CLEANING_RATE as f64;
    if end > length {
        return Err(format!("{} is only {length:.1}s long, so can't cut to {end}s", recording.display()).into());
    }

    let started = std::time::Instant::now();
    let cleaned = denoise(&noisy, reduction_db)?;
    let sample = cut(&to_voice_rate(&cleaned)?, start, end, SAMPLE_RATE);
    let took = started.elapsed().as_secs_f32();

    // Clamp as the WAV does, so the peak reported is the peak saved.
    let sample: Vec<f32> = sample.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect();
    let measured = quality::measure(&sample, SAMPLE_RATE);
    AudioSamples::new(sample, SAMPLE_RATE).save_wav(output)?;
    println!(
        "Wrote {}: {:.2}s, peak {:.1} dB, cleaned with up to {reduction_db} dB of reduction in {took:.1}s",
        output.display(),
        measured.seconds,
        measured.peak_db,
    );
    println!(
        "The first command to use it transcribes it to {}. Check that file's words.",
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
    if model.sr != CLEANING_RATE as usize {
        return Err(format!("the built-in DeepFilterNet model works at {} Hz, not {CLEANING_RATE}", model.sr).into());
    }
    let hop = model.hop_size;
    // The STFT and the model's lookahead delay the output by this much.
    let delay = model.fft_size - hop + model.lookahead * hop;

    // Feed enough after the end to flush the delay, mirroring the end of the
    // recording rather than adding silence.
    let mut input = noisy.to_vec();
    let padded = (noisy.len() + delay).div_ceil(hop) * hop;
    input.extend(noisy.iter().rev().cycle().take(padded - noisy.len()));
    lift_quiet_hops(&mut input, hop);

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

/// Adds a faint fixed hiss, about -67 dB, to each hop too quiet for DeepFilterNet
/// to process, such as digital silence or a gated pause, so every hop moves the
/// model along. Cleaning removes most of it. Louder hops are left untouched, so
/// a recording without quiet hops is cleaned exactly as before.
fn lift_quiet_hops(samples: &mut [f32], hop: usize) {
    let mean_square = |samples: &[f32]| samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    for frame in samples.chunks_mut(hop) {
        if mean_square(frame) >= QUIET_HOP_MEAN_SQUARE {
            continue;
        }
        let mut hiss: Vec<f32> = (0..frame.len())
            .map(|_| {
                // xorshift64, for the same hiss on every run.
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        // Twice the threshold, and in step with what's there, so the sum clears it.
        let scale = (2.0 * QUIET_HOP_MEAN_SQUARE / mean_square(&hiss)).sqrt();
        let along: f32 = frame.iter().zip(&hiss).map(|(s, h)| s * h).sum();
        let sign = if along < 0.0 { -1.0 } else { 1.0 };
        hiss.iter_mut().for_each(|h| *h *= sign * scale);
        frame.iter_mut().zip(&hiss).for_each(|(s, h)| *s += h);
    }
}

/// Converts audio at the cleaning rate to the voice sample rate.
fn to_voice_rate(samples: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let view = ArrayView2::from_shape((1, samples.len()), samples)?;
    let resampled = df::transforms::resample(view, CLEANING_RATE as usize, SAMPLE_RATE as usize, None)?;
    let length = samples.len() * SAMPLE_RATE as usize / CLEANING_RATE as usize;
    Ok(resampled.row(0).iter().take(length).copied().collect())
}

/// The samples from `start` to `end` seconds, or as much of that as there is.
fn cut(samples: &[f32], start: f64, end: f64, sample_rate: u32) -> Vec<f32> {
    let index = |seconds: f64| ((seconds * sample_rate as f64).round() as usize).min(samples.len());
    samples[index(start)..index(end).max(index(start))].to_vec()
}

/// Reads a time in a recording the way ffmpeg does: seconds such as `20.1`,
/// optionally with an `s`, `ms` or `us` unit, or `[HH:]MM:SS[.fraction]`.
pub fn parse_time(text: &str) -> Result<f64, String> {
    seconds_in(text.trim()).ok_or_else(|| {
        format!("{text} isn't a time; give seconds, such as 20.1, 20.1s or 1500ms, or minutes and seconds, such as 1:05.5")
    })
}

fn seconds_in(text: &str) -> Option<f64> {
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

/// A reduction level from text, if it's within `REDUCTION_RANGE`.
pub fn parse_reduction(text: &str) -> Result<f32, String> {
    let reduction = text.trim().parse::<f32>().ok().filter(|db| REDUCTION_RANGE.contains(db));
    reduction.ok_or_else(|| {
        format!("--reduction must be from {} to {} dB, not {text}", REDUCTION_RANGE.start(), REDUCTION_RANGE.end())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

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
            let parsed = parse_time(text).unwrap_or_else(|error| panic!("{error}"));
            assert!((parsed - seconds).abs() < 1e-9, "{text} gave {parsed}");
        }
    }

    #[test]
    fn nonsense_times_are_refused() {
        for text in ["", "s", "-5", "1e3", "ten", "1:2:3:4", "1:60", "1:-5", "1.5:00", "0:61", "inf", "NaN", "20 s"] {
            assert!(parse_time(text).is_err(), "{text}");
        }
    }

    #[test]
    fn reduction_must_be_from_1_to_100_db() {
        assert_eq!(parse_reduction("15"), Ok(15.0));
        assert_eq!(parse_reduction("100"), Ok(100.0));
        for bad in ["0", "0.5", "-15", "101", "lots", "NaN"] {
            assert!(parse_reduction(bad).is_err(), "{bad}");
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
    fn converting_keeps_a_voice_band_tone_and_drops_what_24_khz_cant_hold() {
        let tone = |hz: f32, rate: u32| -> Vec<f32> {
            (0..rate).map(|i| 0.5 * (std::f32::consts::TAU * hz * i as f32 / rate as f32).sin()).collect()
        };
        let voice = to_voice_rate(&tone(440.0, CLEANING_RATE)).unwrap();
        assert_eq!(voice.len(), SAMPLE_RATE as usize);
        // Ignore the edges, where the filter starts and stops.
        let middle = &voice[1000..SAMPLE_RATE as usize - 1000];
        assert!((rms(middle) - 0.5 / 2f32.sqrt()).abs() < 0.01, "{}", rms(middle));
        // The tone stays in time: compare with the same tone made at 24 kHz.
        let expected = tone(440.0, SAMPLE_RATE);
        let error: Vec<f32> = middle.iter().zip(&expected[1000..]).map(|(a, b)| a - b).collect();
        assert!(rms(&error) < 0.01, "{}", rms(&error));

        let too_high = to_voice_rate(&tone(15_000.0, CLEANING_RATE)).unwrap();
        assert!(rms(&too_high[1000..SAMPLE_RATE as usize - 1000]) < 0.01);
    }

    #[test]
    fn only_quiet_hops_are_lifted_and_all_of_them_clear_the_threshold() {
        let hop = 480;
        let loud: Vec<f32> = (0..hop * 4).map(|i| 0.1 * (i as f32 * 0.07).sin()).collect();
        let mut unchanged = loud.clone();
        lift_quiet_hops(&mut unchanged, hop);
        assert_eq!(unchanged, loud);

        // Silence, a whisper just under the threshold, and a partial last hop.
        let mut quiet = loud.clone();
        quiet.extend(vec![0.0; hop]);
        quiet.extend((0..hop).map(|i| 3e-4 * (i as f32 * 0.3).sin()));
        quiet.extend(vec![0.0; hop / 3]);
        let before = quiet.clone();
        lift_quiet_hops(&mut quiet, hop);
        assert_eq!(quiet[..loud.len()], loud[..]);
        for frame in quiet.chunks(hop) {
            assert!(rms(frame).powi(2) >= QUIET_HOP_MEAN_SQUARE, "{}", rms(frame));
        }
        // Barely: about -67 dB.
        assert!(rms(&quiet[loud.len()..loud.len() + hop]) < 1e-3);
        // The same every run.
        let mut again = before;
        lift_quiet_hops(&mut again, hop);
        assert_eq!(again, quiet);
    }

    /// A sweep rather than a steady tone, so it lines up with itself at one offset only.
    fn sweep(seconds: f32) -> Vec<f32> {
        let length = (seconds * CLEANING_RATE as f32) as usize;
        let mut phase = 0f32;
        (0..length)
            .map(|i| {
                let hz = 200.0 + 2000.0 * i as f32 / length as f32;
                phase += std::f32::consts::TAU * hz / CLEANING_RATE as f32;
                0.3 * phase.sin()
            })
            .collect()
    }

    /// The offset, in samples, at which `output` best matches `input`.
    fn best_offset(input: &[f32], output: &[f32], range: i64) -> i64 {
        let score = |offset: i64| -> f32 {
            (2000..input.len() - 2000)
                .map(|i| input[i] * output.get((i as i64 + offset) as usize).copied().unwrap_or(0.0))
                .sum()
        };
        (-range..=range).max_by(|a, b| score(*a).total_cmp(&score(*b))).unwrap()
    }

    #[test]
    fn a_gated_recording_stays_in_time_after_a_silent_gap() {
        let (before, after) = (sweep(0.5), sweep(0.8));
        let mut recording = before.clone();
        recording.extend(vec![0.0; CLEANING_RATE as usize / 2]);
        let gap_end = recording.len();
        recording.extend(&after);
        // And ends in digital silence, which the model needs moving through too.
        recording.extend(vec![0.0; CLEANING_RATE as usize / 4]);

        let cleaned = denoise(&recording, DEFAULT_REDUCTION_DB).unwrap();
        assert_eq!(cleaned.len(), recording.len());
        let cleaned_after = &cleaned[gap_end..gap_end + after.len()];
        assert!(rms(cleaned_after) > 0.5 * rms(&after), "{} vs {}", rms(cleaned_after), rms(&after));
        // A stalled model would put it 1440 samples late.
        let offset = best_offset(&after, &cleaned[gap_end..], 2000);
        assert!(offset.abs() <= 2, "the sweep after the gap moved by {offset} samples");
    }
}
