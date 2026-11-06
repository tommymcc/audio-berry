//! Measurements for spotting bad takes, since the model varies between seeds.

const FRAME_SECONDS: f32 = 0.05;
const SILENCE_DB: f32 = -55.0;

/// A quiet-frame level above this means something is playing behind the voice.
/// Takes heard as clean measured about -91 to -96 dB, and noisy ones -34 to -48.
pub const MAX_FLOOR_DB: f32 = -55.0;

/// Overrides `MAX_FLOOR_DB`. A sample with faint room tone, such as a denoised
/// outdoor recording, passes a little of it to every take: one voice's takes
/// measured -51 to -57 dB and were heard as clean on a Cast speaker.
const NOISE_LIMIT_VAR: &str = "AUDIO_BERRY_NOISE_LIMIT_DB";

/// The noise limit in use, read once from the environment.
pub fn noise_limit_db() -> f32 {
    static LIMIT: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let value = std::env::var(NOISE_LIMIT_VAR).ok();
        let limit = parse_noise_limit(value.as_deref());
        match value {
            Some(value) if value.trim().parse::<f32>().ok() != Some(limit) => {
                eprintln!("{NOISE_LIMIT_VAR}={value} isn't a level from -100 to 0 dB, so using {limit} dB")
            }
            Some(_) => eprintln!("Noise limit: {limit} dB ({NOISE_LIMIT_VAR})"),
            None => {}
        }
        limit
    })
}

fn parse_noise_limit(value: Option<&str>) -> f32 {
    value
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|db| (-100.0..=0.0).contains(db))
        .unwrap_or(MAX_FLOOR_DB)
}

const MIN_SECONDS_FOR_FLOOR: f32 = 6.0;

#[derive(Clone, Copy, Debug)]
pub struct Quality {
    /// Level of the quietest 5% of frames.
    pub floor_db: f32,
    /// Level of the speech, ignoring silence.
    pub speech_db: f32,
    pub peak_db: f32,
    pub seconds: f32,
}

fn to_db(linear: f32) -> f32 {
    20.0 * linear.max(1e-5).log10()
}

fn frame_levels(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let size = ((sample_rate as f32 * FRAME_SECONDS) as usize).max(1);
    samples
        .chunks_exact(size)
        .map(|frame| (frame.iter().map(|s| s * s).sum::<f32>() / size as f32).sqrt())
        .collect()
}

pub fn measure(samples: &[f32], sample_rate: u32) -> Quality {
    let mut levels = frame_levels(samples, sample_rate);
    levels.sort_by(f32::total_cmp);
    let floor = levels.get(levels.len() / 20).copied().unwrap_or(0.0);
    let threshold = 10f32.powf(SILENCE_DB / 20.0);
    let active: Vec<f32> = levels.into_iter().filter(|level| *level > threshold).collect();
    let speech = if active.is_empty() {
        0.0
    } else {
        (active.iter().map(|level| level * level).sum::<f32>() / active.len() as f32).sqrt()
    };
    Quality {
        floor_db: to_db(floor),
        speech_db: to_db(speech),
        peak_db: to_db(samples.iter().fold(0f32, |peak, sample| peak.max(sample.abs()))),
        seconds: samples.len() as f32 / sample_rate as f32,
    }
}

impl Quality {
    /// Whether the pauses are quiet. Very short takes barely pause, so their
    /// quietest frames are speech and say nothing about the background.
    pub fn clean(&self) -> bool {
        self.seconds < MIN_SECONDS_FOR_FLOOR || self.floor_db <= noise_limit_db()
    }
}

/// Whether a take's length is plausible for its text. A take far outside this
/// has probably skipped words or run on. Tested speech ran at 16-19 characters
/// per second, and short headings can be slower.
pub fn plausible_length(characters: usize, seconds: f32) -> bool {
    let characters = characters as f32;
    seconds >= characters / 30.0 && seconds <= characters / 9.0 + 4.0
}

/// The same check for narration at a chosen speed.
pub fn plausible_length_at(characters: usize, seconds: f32, speed: f64) -> bool {
    plausible_length(characters, seconds * speed as f32)
}

/// Removes leading and trailing silence, so gaps between chunks are consistent.
pub fn trim_silence(samples: &[f32], sample_rate: u32) -> &[f32] {
    let size = ((sample_rate as f32 * FRAME_SECONDS) as usize).max(1);
    let threshold = 10f32.powf(SILENCE_DB / 20.0);
    let levels = frame_levels(samples, sample_rate);
    let Some(first) = levels.iter().position(|level| *level > threshold) else {
        return samples;
    };
    let last = levels.iter().rposition(|level| *level > threshold).unwrap_or(first);
    // Keep a frame either side so soft consonants aren't clipped.
    let start = first.saturating_sub(1) * size;
    let end = ((last + 2) * size).min(samples.len());
    &samples[start..end]
}

/// Gain that brings speech to `target_db`, limited to avoid amplifying a bad
/// take too far or clipping peaks.
pub fn gain_for(quality: &Quality, target_db: f32) -> f32 {
    let wanted = (target_db - quality.speech_db).clamp(-6.0, 6.0);
    let room = -1.0 - quality.peak_db;
    10f32.powf(wanted.min(room) / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plausible_length_follows_the_speed() {
        // 300 characters at the usual 18 per second takes about 16.7 seconds.
        assert!(!plausible_length_at(300, 8.3, 1.0));
        assert!(plausible_length_at(300, 8.3, 2.0));
        assert!(!plausible_length_at(300, 42.0, 1.0));
        assert!(plausible_length_at(300, 42.0, 0.5));
    }

    fn tone(seconds: f32, level: f32) -> Vec<f32> {
        (0..(24_000.0 * seconds) as usize)
            .map(|i| level * (i as f32 * 0.05).sin())
            .collect()
    }

    #[test]
    fn floor_separates_clean_pauses_from_background() {
        let mut clean = tone(3.0, 0.2);
        clean.extend(vec![0.0; 12_000]);
        let mut noisy = tone(3.0, 0.2);
        noisy.extend(tone(0.5, 0.02));
        assert!(measure(&clean, 24_000).floor_db < MAX_FLOOR_DB);
        assert!(measure(&noisy, 24_000).floor_db > MAX_FLOOR_DB);
    }

    #[test]
    fn noise_limit_comes_from_a_valid_level_or_the_default() {
        assert_eq!(parse_noise_limit(None), MAX_FLOOR_DB);
        assert_eq!(parse_noise_limit(Some("-50")), -50.0);
        assert_eq!(parse_noise_limit(Some(" -48.5 ")), -48.5);
        assert_eq!(parse_noise_limit(Some("loud")), MAX_FLOOR_DB);
        assert_eq!(parse_noise_limit(Some("5")), MAX_FLOOR_DB);
        assert_eq!(parse_noise_limit(Some("-120")), MAX_FLOOR_DB);
    }

    #[test]
    fn short_takes_are_not_judged_on_their_floor() {
        assert!(measure(&tone(1.5, 0.2), 24_000).clean());
        assert!(!measure(&tone(8.0, 0.2), 24_000).clean());
    }

    #[test]
    fn trimming_keeps_speech_and_a_margin() {
        let mut samples = vec![0.0; 24_000];
        samples.extend(tone(1.0, 0.2));
        samples.extend(vec![0.0; 24_000]);
        let trimmed = trim_silence(&samples, 24_000);
        assert!(trimmed.len() >= 24_000 && trimmed.len() < 28_000, "{}", trimmed.len());
    }

    #[test]
    fn gain_never_pushes_peaks_past_the_limit() {
        let quality = Quality { floor_db: -90.0, speech_db: -30.0, peak_db: -2.0, seconds: 1.0 };
        assert!(gain_for(&quality, -21.0) <= 10f32.powf(1.0 / 20.0) + 1e-6);
    }

    #[test]
    fn length_check_catches_skips_and_run_ons() {
        assert!(plausible_length(600, 34.0));
        assert!(!plausible_length(600, 5.0));
        assert!(!plausible_length(600, 120.0));
        assert!(plausible_length(30, 3.0));
    }
}
