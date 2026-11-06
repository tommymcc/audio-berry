//! Measurements for spotting bad takes, since the model varies between seeds.

/// Levels are measured over frames this long.
pub const FRAME_SECONDS: f32 = 0.05;
const SILENCE_DB: f32 = -55.0;

/// A quiet-frame level above this means something is playing behind the voice.
/// Takes heard as clean measured about -91 to -96 dB, and noisy ones -34 to -48.
pub const MAX_FLOOR_DB: f32 = -55.0;

/// Overrides `MAX_FLOOR_DB`, and is saved with the output directory. A sample
/// with faint room tone, such as a denoised outdoor recording, passes a little
/// of it to every take: one voice's takes measured -51 to -57 dB and were heard
/// as clean on a Cast speaker.
pub const NOISE_LIMIT_VAR: &str = "AUDIO_BERRY_NOISE_LIMIT_DB";

/// The noise limit given in the environment, if any. An invalid value is
/// reported and ignored.
pub fn noise_limit_from_env() -> Option<f32> {
    let value = std::env::var(NOISE_LIMIT_VAR).ok()?;
    let limit = valid_noise_limit(&value);
    if limit.is_none() {
        eprintln!("{NOISE_LIMIT_VAR}={value} isn't a level from -100 to 0 dB, so ignoring it");
    }
    limit
}

/// The noise limit given in the environment, without reporting an invalid value,
/// for when something else reports it: when a voice sample loads, `book` and
/// `retake` report it later.
pub fn noise_limit_from_env_quietly() -> Option<f32> {
    std::env::var(NOISE_LIMIT_VAR).ok().and_then(|value| valid_noise_limit(&value))
}

/// A noise limit from text, if it's a level from -100 to 0 dB.
pub fn valid_noise_limit(value: &str) -> Option<f32> {
    value.trim().parse::<f32>().ok().filter(|db| (-100.0..=0.0).contains(db))
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

fn from_db(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

fn frame_levels(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let size = ((sample_rate as f32 * FRAME_SECONDS) as usize).max(1);
    samples
        .chunks_exact(size)
        .map(|frame| (frame.iter().map(|s| s * s).sum::<f32>() / size as f32).sqrt())
        .collect()
}

/// The level of each `FRAME_SECONDS` frame, in dB.
pub fn frame_levels_db(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    frame_levels(samples, sample_rate).into_iter().map(to_db).collect()
}

pub fn measure(samples: &[f32], sample_rate: u32) -> Quality {
    let mut levels = frame_levels(samples, sample_rate);
    levels.sort_by(f32::total_cmp);
    let floor = levels.get(levels.len() / 20).copied().unwrap_or(0.0);
    let threshold = from_db(SILENCE_DB);
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
    pub fn clean(&self, noise_limit_db: f32) -> bool {
        self.seconds < MIN_SECONDS_FOR_FLOOR || self.floor_db <= noise_limit_db
    }
}

/// Whether a take's length is plausible for its text. A take far outside this
/// has probably skipped words or run on. Tested speech ran at 16-19 characters
/// per second, and short headings can be slower.
fn plausible_length(characters: usize, seconds: f32) -> bool {
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
    let threshold = from_db(SILENCE_DB);
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
    from_db(wanted.min(room))
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
    fn a_noise_limit_must_be_a_level_from_minus_100_to_0() {
        assert_eq!(valid_noise_limit("-50"), Some(-50.0));
        assert_eq!(valid_noise_limit(" -48.5 "), Some(-48.5));
        assert_eq!(valid_noise_limit("loud"), None);
        assert_eq!(valid_noise_limit("5"), None);
        assert_eq!(valid_noise_limit("-120"), None);
        assert_eq!(valid_noise_limit(""), None);
    }

    #[test]
    fn the_noise_limit_decides_whether_a_take_is_clean() {
        let mut take = tone(8.0, 0.2);
        take.extend(tone(1.0, 0.004));
        let quality = measure(&take, 24_000);
        assert!(quality.floor_db > -55.0 && quality.floor_db < -45.0, "{}", quality.floor_db);
        assert!(!quality.clean(-55.0));
        assert!(quality.clean(-45.0));
    }

    #[test]
    fn short_takes_are_not_judged_on_their_floor() {
        assert!(measure(&tone(1.5, 0.2), 24_000).clean(MAX_FLOOR_DB));
        assert!(!measure(&tone(8.0, 0.2), 24_000).clean(MAX_FLOOR_DB));
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
