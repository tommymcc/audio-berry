//! Measurements for spotting bad takes, since the model varies between seeds.

/// Levels are measured over frames this long.
pub const FRAME_SECONDS: f32 = 0.05;
const SILENCE_DB: f32 = -55.0;

/// The noise limit unless one is given: a floor above this means something is
/// playing behind the voice. Takes heard as clean measured about -91 to -96 dB,
/// and noisy ones -34 to -48.
pub const DEFAULT_NOISE_LIMIT_DB: f32 = -55.0;

/// Gives `--noise-limit`, which overrides `DEFAULT_NOISE_LIMIT_DB` and is saved
/// with the output directory. A sample with faint room tone, such as a denoised
/// outdoor recording, passes a little of it to every take: one voice's takes
/// measured -51 to -57 dB and were heard as clean on a Cast speaker.
pub const NOISE_LIMIT_VAR: &str = "AUDIO_BERRY_NOISE_LIMIT_DB";

/// A noise limit given on the command line.
pub fn parse_noise_limit(value: &str) -> Result<f32, String> {
    valid_noise_limit(value).ok_or_else(|| "must be a level from -100 to 0 dB".to_string())
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

/// How many samples a `FRAME_SECONDS` frame holds at `sample_rate`.
pub fn frame_size(sample_rate: u32) -> usize {
    ((sample_rate as f32 * FRAME_SECONDS) as usize).max(1)
}

/// The largest absolute sample, as a linear level.
pub fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0f32, |peak, sample| peak.max(sample.abs()))
}

/// The floor of frame levels sorted quietest first: the level of the quietest
/// 5%. Linear or in dB, it's the same frame.
pub fn floor_of_sorted(levels: &[f32]) -> Option<f32> {
    levels.get(levels.len() / 20).copied()
}

fn frame_levels(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let size = frame_size(sample_rate);
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
    let floor = floor_of_sorted(&levels).unwrap_or(0.0);
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
        peak_db: to_db(peak(samples)),
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

/// Trimming takes frames this far above the take's floor as speech. Room tone
/// before the first word measured up to 16 dB above the floor in a noisy voice's
/// takes, and first words rose 30 dB or more above it.
const SPEECH_ABOVE_FLOOR_DB: f32 = 20.0;
/// The most that trimming's levels can be, whatever the floor: a take with no
/// pause to measure a floor in has its quietest speech as its floor, and this
/// keeps that speech. First words in three voices peaked at -8 to -15 dB within
/// 0.3 s.
const SPEECH_CAP_DB: f32 = -30.0;
/// From the speech, trimming keeps frames this far above the floor, which catches
/// a word's soft start and fading end but not the room tone around them.
const EDGE_ABOVE_FLOOR_DB: f32 = 6.0;
/// Frames kept either side of that, for soft consonants quieter still: a first
/// word's "f" measured within the floor's 6 dB for up to 50 ms.
const EDGE_PAD_FRAMES: usize = 2;
/// Frames at or below this linear level are digital silence: `to_db` reads them
/// all as -100 dB.
const DIGITAL_SILENCE: f32 = 1e-5;

/// The floor of a take's frames that aren't digital silence. Takes start with
/// about 0.1 s of it, which in a short take is more than the quietest 5% of its
/// frames, so `measure`'s floor would be that rather than the room tone.
fn floor_ignoring_digital_silence(levels: &[f32]) -> Option<f32> {
    let mut sound: Vec<f32> = levels.iter().copied().filter(|level| *level > DIGITAL_SILENCE).collect();
    sound.sort_by(f32::total_cmp);
    floor_of_sorted(&sound).map(to_db)
}

/// The last frame of speech ending at `last`, given which frames are above the
/// edge level. A last word ending in a stop, as in "watch", goes quiet for its
/// closure and then releases, so straight after speech one frame below the edge
/// level is stepped over if the next is above it. Only straight after speech: a
/// fading end then reaches room tone, whose flicker this would step through.
/// Only at the end, too: a first stop's closure is silence before any sound.
fn speech_end(last: usize, above_edge: impl Fn(usize) -> bool) -> usize {
    let mut end = last;
    if !above_edge(end + 1) && above_edge(end + 2) {
        end += 2;
    }
    while above_edge(end + 1) {
        end += 1;
    }
    end
}

/// Removes leading and trailing silence, so gaps between chunks are consistent.
/// Speech is what rises `SPEECH_ABOVE_FLOOR_DB` above the take's floor, ignoring
/// digital silence, and the trim keeps what joins it above `EDGE_ABOVE_FLOOR_DB`,
/// plus `EDGE_PAD_FRAMES` either side. So a voice with room tone loses it too.
/// Both levels are at least `SILENCE_DB`, so for a clean take, with a floor below
/// about -75 dB, speech is what rises above it as before; the trim then keeps one
/// more frame either side than it used to, and soft edges down to the floor + 6
/// dB once the floor is above -61 dB.
pub fn trim_silence(samples: &[f32], sample_rate: u32) -> &[f32] {
    let size = frame_size(sample_rate);
    let levels = frame_levels(samples, sample_rate);
    let floor = floor_ignoring_digital_silence(&levels).unwrap_or(f32::NEG_INFINITY);
    // Capping the edge level as well as the speech level is deliberate: the edge
    // must stay below the speech it extends from.
    let above_floor = |margin: f32| from_db((floor + margin).clamp(SILENCE_DB, SPEECH_CAP_DB));
    let speech = above_floor(SPEECH_ABOVE_FLOOR_DB);
    let edge = above_floor(EDGE_ABOVE_FLOOR_DB);
    let Some(mut first) = levels.iter().position(|level| *level > speech) else {
        return samples;
    };
    let last = levels.iter().rposition(|level| *level > speech).unwrap_or(first);
    while first > 0 && levels[first - 1] > edge {
        first -= 1;
    }
    let last = speech_end(last, |frame| levels.get(frame).is_some_and(|level| *level > edge));
    let start = first.saturating_sub(EDGE_PAD_FRAMES) * size;
    let end = ((last + 1 + EDGE_PAD_FRAMES) * size).min(samples.len());
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
        assert!(measure(&clean, 24_000).floor_db < DEFAULT_NOISE_LIMIT_DB);
        assert!(measure(&noisy, 24_000).floor_db > DEFAULT_NOISE_LIMIT_DB);
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
        assert!(measure(&tone(1.5, 0.2), 24_000).clean(DEFAULT_NOISE_LIMIT_DB));
        assert!(!measure(&tone(8.0, 0.2), 24_000).clean(DEFAULT_NOISE_LIMIT_DB));
    }

    #[test]
    fn trimming_keeps_speech_and_a_margin() {
        let mut samples = vec![0.0; 24_000];
        samples.extend(tone(1.0, 0.2));
        samples.extend(vec![0.0; 24_000]);
        let trimmed = trim_silence(&samples, 24_000);
        // Two frames are kept either side, 4,800 samples; before, one was.
        assert!(trimmed.len() >= 24_000 && trimmed.len() < 30_000, "{}", trimmed.len());
    }

    /// Steady noise at `db`, the same every time.
    fn noise(seconds: f32, db: f32) -> Vec<f32> {
        // Uniform noise from -1 to 1 has an RMS of 1/sqrt(3).
        let scale = from_db(db) * 3f32.sqrt();
        // A linear congruential generator with the Numerical Recipes constants,
        // from a fixed seed, so every run gets the same noise.
        let mut state = 12_345u32;
        (0..(24_000.0 * seconds) as usize)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                scale * (state as f32 / u32::MAX as f32 * 2.0 - 1.0)
            })
            .collect()
    }

    /// Where in `samples` the trimmed part starts and ends, in frames.
    fn trimmed_frames(samples: &[f32]) -> (usize, usize) {
        let trimmed = trim_silence(samples, 24_000);
        let start = trimmed.as_ptr() as usize - samples.as_ptr() as usize;
        let start = start / std::mem::size_of::<f32>();
        let size = frame_size(24_000);
        (start / size, (start + trimmed.len()) / size)
    }

    /// A 900 Hz tone at `db`, like the squeak some takes start with.
    fn squeak(seconds: f32, db: f32) -> Vec<f32> {
        let amplitude = from_db(db) * 2f32.sqrt();
        (0..(24_000.0 * seconds) as usize)
            .map(|i| amplitude * (i as f32 * std::f32::consts::TAU * 900.0 / 24_000.0).sin())
            .collect()
    }

    #[test]
    fn trimming_removes_room_tone_up_to_the_speech() {
        // Laid out like a noisy voice's short take: digital silence, room tone at
        // -45 dB, speech, room tone, digital silence.
        let mut samples = vec![0.0; 2_400];
        samples.extend(noise(0.5, -45.0));
        samples.extend(tone(0.5, 0.2));
        samples.extend(noise(0.2, -45.0));
        samples.extend(vec![0.0; 2_400]);
        // Speech is frames 12 to 21, and two frames either side are kept.
        assert_eq!(trimmed_frames(&samples), (10, 24));
    }

    #[test]
    fn trimming_keeps_a_last_stop_released_after_a_quiet_frame() {
        let mut samples = noise(0.5, -45.0);
        samples.extend(tone(0.5, 0.2));
        samples.extend(noise(0.05, -45.0));
        samples.extend(noise(0.05, -33.0));
        samples.extend(noise(0.5, -45.0));
        // Speech is frames 10 to 19, and the release frame 21.
        assert_eq!(trimmed_frames(&samples), (8, 24));
    }

    #[test]
    fn trimming_stops_at_room_tone_after_a_fading_end() {
        let mut samples = noise(0.5, -45.0);
        samples.extend(tone(0.5, 0.2));
        samples.extend(noise(0.05, -36.0));
        samples.extend(noise(0.05, -45.0));
        samples.extend(noise(0.05, -37.0));
        samples.extend(noise(0.5, -45.0));
        // Speech is frames 10 to 19, fading in 20; the flicker in 22 is room tone.
        assert_eq!(trimmed_frames(&samples), (8, 23));
    }

    #[test]
    fn trimming_drops_a_squeak_before_the_room_tone() {
        let mut samples = vec![0.0; 2_400];
        samples.extend(noise(0.1, -45.0));
        samples.extend(squeak(0.2, -35.0));
        samples.extend(noise(0.4, -45.0));
        samples.extend(tone(0.5, 0.2));
        samples.extend(noise(0.5, -45.0));
        // The squeak is frames 4 to 7, 10 dB above the floor; the speech is 16 to 25.
        assert_eq!(trimmed_frames(&samples), (14, 28));
    }

    #[test]
    fn trimming_a_clean_take_keeps_what_rises_past_silence() {
        // With a -70 dB floor, a soft start at -52 dB is kept as it always was.
        let mut samples = noise(1.0, -70.0);
        samples.extend(noise(0.2, -52.0));
        samples.extend(tone(1.0, 0.2));
        samples.extend(noise(1.0, -70.0));
        // The soft start is frames 20 to 23, the speech 24 to 43.
        assert_eq!(trimmed_frames(&samples), (18, 46));
    }

    #[test]
    fn trimming_leaves_takes_without_speech_alone() {
        for samples in [vec![0.0; 48_000], noise(2.0, -60.0), noise(2.0, -45.0)] {
            assert_eq!(trim_silence(&samples, 24_000).len(), samples.len());
        }
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
