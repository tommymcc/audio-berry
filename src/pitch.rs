//! Measures how lively a take sounds, as the range of pitch its voice spans.
//! Flat takes of a lively voice sound unlike it even when every word is right.
//!
//! Pitch is tracked with pYIN, as librosa's `pyin` does it, so ranges agree with
//! the listening tests that chose the target: YIN's troughs at many thresholds
//! give each frame candidate pitches, and a hidden Markov model picks a smooth
//! path through them, deciding which frames are voiced.

use realfft::RealFftPlanner;
use realfft::num_complex::Complex;

/// The targets `--pitch-range` can ask for, in semitones. One voice's lines
/// spanned 7 to 16.
pub const PITCH_RANGE_LIMITS: std::ops::RangeInclusive<f32> = 2.0..=30.0;

/// A target pitch range given on the command line.
pub fn parse_pitch_range(value: &str) -> Result<f32, String> {
    let (lowest, highest) = (PITCH_RANGE_LIMITS.start(), PITCH_RANGE_LIMITS.end());
    valid_pitch_range(value).ok_or_else(|| format!("must be a range from {lowest} to {highest} semitones"))
}

/// A target pitch range from text, if it's within `PITCH_RANGE_LIMITS`.
pub fn valid_pitch_range(value: &str) -> Option<f32> {
    value.trim().parse::<f32>().ok().filter(|semitones| PITCH_RANGE_LIMITS.contains(semitones))
}

/// How near a take's pitch range is to the target, as the smaller of their
/// ratios: 1 when exact, and towards 0 the further off, whether flatter or livelier.
pub fn closeness(range: f32, target: f32) -> f32 {
    if range <= 0.0 || target <= 0.0 {
        return 0.0;
    }
    (range / target).min(target / range)
}

/// The pitch range of speech, in semitones: from the 5th to the 95th percentile
/// of its voiced pitch, measured from its median. `None` if fewer than
/// `MIN_VOICED_FRAMES` frames are voiced, such as in silence or noise.
pub fn range_semitones(samples: &[f32], sample_rate: u32) -> Option<f32> {
    let voiced: Vec<f64> = track(samples, sample_rate).into_iter().flatten().collect();
    if voiced.len() < MIN_VOICED_FRAMES {
        return None;
    }
    let median = percentile(&sorted(voiced.clone()), 50.0);
    let semitones = sorted(voiced.iter().map(|hz| 12.0 * (hz / median).log2()).collect());
    Some((percentile(&semitones, 95.0) - percentile(&semitones, 5.0)) as f32)
}

/// Fewer voiced frames than this is too little speech to measure.
const MIN_VOICED_FRAMES: usize = 20;
/// The pitch searched, which covers a man's speaking voice and most women's.
const LOWEST_HZ: f64 = 60.0;
const HIGHEST_HZ: f64 = 300.0;
/// Each frame is analysed over this much audio, centred on it.
const FRAME_SECONDS: f64 = 0.064;
const HOP_SECONDS: f64 = 0.01;
/// The pitch path's resolution.
const BINS_PER_SEMITONE: usize = 10;
/// How fast the pitch path can move, in semitones a second.
const MAX_TRANSITION_RATE: f64 = 35.92;
/// The chance of a frame switching between voiced and unvoiced.
const SWITCH_PROBABILITY: f64 = 0.01;
/// The chance given to the deepest trough when none is below a threshold.
const NO_TROUGH_PROBABILITY: f64 = 0.01;
/// YIN thresholds are tried from 0.01 to 1 in steps of this.
const THRESHOLDS: usize = 100;
/// The prior over thresholds is Beta(2, 18), so most weight is on thresholds near 0.1.
const BETA_B: i32 = 18;
/// Earlier troughs, at shorter periods, are favoured by this Boltzmann parameter.
const BOLTZMANN: f64 = 2.0;

/// Speech is analysed at this rate whatever its own. Tracking is sensitive to it:
/// a few takes measured 2 to 4 semitones apart at 24 and 16 kHz, so this is the
/// rate the listening tests measured at.
const ANALYSIS_RATE: u32 = 16_000;

/// Each frame's pitch in Hz every `HOP_SECONDS`, or `None` where unvoiced.
fn track(samples: &[f32], sample_rate: u32) -> Vec<Option<f64>> {
    let samples = resample(samples, sample_rate, ANALYSIS_RATE);
    let rate = f64::from(ANALYSIS_RATE);
    let frame = (FRAME_SECONDS * rate).round() as usize & !1;
    let hop = ((HOP_SECONDS * rate).round() as usize).max(1);
    let min_period = (rate / HIGHEST_HZ).floor() as usize;
    let max_period = ((rate / LOWEST_HZ).ceil() as usize).min(frame.saturating_sub(1));
    if samples.is_empty() || min_period < 1 || max_period <= min_period + 1 {
        return Vec::new();
    }
    let bins = (12.0 * BINS_PER_SEMITONE as f64 * (HIGHEST_HZ / LOWEST_HZ).log2()).floor() as usize + 1;
    let semitones_per_hop = (MAX_TRANSITION_RATE * 12.0 * HOP_SECONDS).round() as usize;
    let mut path = Path::new(bins, semitones_per_hop * BINS_PER_SEMITONE / 2);
    let mut differences = Differences::new(frame, min_period, max_period);
    let observations = Observations::new(bins, min_period, rate);

    // Frames are centred, so the signal is padded with half a frame of silence.
    let half = frame / 2;
    let mut padded = vec![0.0; samples.len() + frame];
    padded[half..half + samples.len()].copy_from_slice(&samples);
    let frames = 1 + samples.len() / hop;
    for index in 0..frames {
        let start = index * hop;
        let cmnd = differences.measure(&padded[start..start + frame]);
        path.step(&observations.of(cmnd));
    }
    path.best()
        .into_iter()
        .map(|state| (state < bins).then(|| LOWEST_HZ * 2f64.powf(state as f64 / (12 * BINS_PER_SEMITONE) as f64)))
        .collect()
}

/// Zero crossings of the resampling filter on each side, and its Kaiser window's
/// shape, which together keep aliasing near 80 dB down.
const RESAMPLING_CROSSINGS: f64 = 32.0;
const KAISER_BETA: f64 = 8.6;
/// The filter passes up to this fraction of the lower Nyquist frequency.
const ROLLOFF: f64 = 0.95;

/// `samples` at `to` Hz rather than `from`, by band-limited interpolation with a
/// Kaiser-windowed sinc. Output samples fall at as many positions between input
/// samples as `to` is over the rates' common factor, so each position's weights
/// are worked out once.
fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f64> {
    if from == 0 {
        return Vec::new();
    }
    if from == to || samples.is_empty() {
        return samples.iter().map(|&sample| f64::from(sample)).collect();
    }
    let (mut common, mut remainder) = (from, to);
    while remainder != 0 {
        (common, remainder) = (remainder, common % remainder);
    }
    let (from_steps, positions) = (u64::from(from / common), u64::from(to / common));
    // In cycles per input sample, below both rates' Nyquist frequencies.
    let cutoff = 0.5 * ROLLOFF * (f64::from(to) / f64::from(from)).min(1.0);
    let reach = RESAMPLING_CROSSINGS / (2.0 * cutoff);
    let window_scale = bessel_i0(KAISER_BETA).recip();
    // For each position, the offset of the first input sample weighted, and the weights.
    let kernels: Vec<(i64, Vec<f64>)> = (0..positions)
        .map(|position| {
            let fraction = position as f64 / positions as f64;
            let first = (fraction - reach).ceil() as i64;
            let last = (fraction + reach).floor() as i64;
            let weights = (first..=last)
                .map(|at| {
                    let offset = at as f64 - fraction;
                    let edge = offset / reach;
                    let window = bessel_i0(KAISER_BETA * (1.0 - edge * edge).max(0.0).sqrt()) * window_scale;
                    let phase = std::f64::consts::TAU * cutoff * offset;
                    let sinc = if phase == 0.0 { 1.0 } else { phase.sin() / phase };
                    2.0 * cutoff * sinc * window
                })
                .collect();
            (first, weights)
        })
        .collect();
    let length = samples.len() as u64 * positions / from_steps;
    (0..length)
        .map(|index| {
            let at = index * from_steps;
            let (base, position) = ((at / positions) as i64, (at % positions) as usize);
            let (first, weights) = &kernels[position];
            let start = base + first;
            weights
                .iter()
                .enumerate()
                .filter_map(|(tap, weight)| {
                    let sample = samples.get(usize::try_from(start + tap as i64).ok()?)?;
                    Some(f64::from(*sample) * weight)
                })
                .sum()
        })
        .collect()
}

/// The modified Bessel function of the first kind, order 0, for the Kaiser window.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
    while term > sum * 1e-12 {
        term *= (x / (2.0 * k)).powi(2);
        sum += term;
        k += 1.0;
    }
    sum
}

/// YIN's cumulative mean normalized difference function, for one frame at a time.
struct Differences {
    frame: usize,
    min_period: usize,
    max_period: usize,
    forward: std::sync::Arc<dyn realfft::RealToComplex<f64>>,
    inverse: std::sync::Arc<dyn realfft::ComplexToReal<f64>>,
    input: Vec<f64>,
    spectrum: Vec<Complex<f64>>,
    autocorrelation: Vec<f64>,
    cmnd: Vec<f64>,
}

impl Differences {
    fn new(frame: usize, min_period: usize, max_period: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(2 * frame);
        let inverse = planner.plan_fft_inverse(2 * frame);
        Differences {
            frame,
            min_period,
            max_period,
            input: forward.make_input_vec(),
            spectrum: forward.make_output_vec(),
            autocorrelation: inverse.make_output_vec(),
            cmnd: vec![0.0; max_period - min_period + 1],
            forward,
            inverse,
        }
    }

    /// The function at each period from `min_period` to `max_period`. Like
    /// librosa, the difference at period k is 2·(r(0) − r(k)) less the energy of
    /// the frame's first k samples, from the frame's autocorrelation r.
    fn measure(&mut self, samples: &[f64]) -> &[f64] {
        self.input[..self.frame].copy_from_slice(samples);
        self.input[self.frame..].fill(0.0);
        // Neither can fail: the buffers are the planned sizes, and the power
        // spectrum is real, as the inverse needs.
        self.forward.process(&mut self.input, &mut self.spectrum).expect("planned forward FFT");
        for bin in &mut self.spectrum {
            *bin = Complex::new(bin.norm_sqr(), 0.0);
        }
        self.inverse.process(&mut self.spectrum, &mut self.autocorrelation).expect("planned inverse FFT");
        let scale = 1.0 / (2 * self.frame) as f64;
        let r0 = self.autocorrelation[0] * scale;
        let (mut energy, mut cumulative) = (0.0, 0.0);
        for period in 1..=self.max_period {
            energy += samples[period - 1] * samples[period - 1];
            let difference = 2.0 * (r0 - self.autocorrelation[period] * scale) - energy;
            cumulative += difference;
            if period >= self.min_period {
                let mean = cumulative / period as f64;
                self.cmnd[period - self.min_period] = difference / (mean + f64::from(f32::MIN_POSITIVE));
            }
        }
        &self.cmnd
    }
}

/// Turns a frame's difference function into how likely each pitch state is:
/// the first `bins` states are voiced at each pitch, and the rest unvoiced.
struct Observations {
    bins: usize,
    min_period: usize,
    rate: f64,
    /// How likely each threshold is, from the Beta prior.
    threshold_weights: Vec<f64>,
}

impl Observations {
    fn new(bins: usize, min_period: usize, rate: f64) -> Self {
        // Beta(2, b)'s CDF has a closed form: 1 − (1 − x)^b · (1 + b·x).
        let cdf = |x: f64| 1.0 - (1.0 - x).powi(BETA_B) * (1.0 + f64::from(BETA_B) * x);
        let threshold = |index: usize| index as f64 / THRESHOLDS as f64;
        let threshold_weights =
            (0..THRESHOLDS).map(|index| cdf(threshold(index + 1)) - cdf(threshold(index))).collect();
        Observations { bins, min_period, rate, threshold_weights }
    }

    fn of(&self, cmnd: &[f64]) -> Vec<f64> {
        let mut probabilities = vec![0.0; 2 * self.bins];
        let last = cmnd.len() - 1;
        let troughs: Vec<usize> = (0..=last)
            .filter(|&index| match index {
                0 => cmnd[0] < cmnd[1],
                _ if index == last => cmnd[index] < cmnd[index - 1],
                _ => cmnd[index] < cmnd[index - 1] && cmnd[index] <= cmnd[index + 1],
            })
            .collect();
        let mut voiced = 0.0;
        if !troughs.is_empty() {
            let mut weights = vec![0.0; troughs.len()];
            // e^(−λ·n) for each count of troughs, for the Boltzmann distribution.
            let decay: Vec<f64> = (0..=troughs.len()).map(|count| (-BOLTZMANN * count as f64).exp()).collect();
            // At each threshold, the troughs below it share its weight, the
            // earliest most, by a Boltzmann distribution over their order.
            for (index, weight) in self.threshold_weights.iter().enumerate() {
                let threshold = (index + 1) as f64 / THRESHOLDS as f64;
                let below = troughs.iter().filter(|&&trough| cmnd[trough] < threshold).count();
                let normaliser = (1.0 - decay[1]) / (1.0 - decay[below]);
                let mut position = 0;
                for (slot, &trough) in troughs.iter().enumerate() {
                    if cmnd[trough] < threshold {
                        weights[slot] += normaliser * decay[position] * weight;
                        position += 1;
                    }
                }
            }
            // The deepest trough also takes the thresholds no trough is below.
            let deepest = first_largest(troughs.iter().map(|&trough| -cmnd[trough]));
            let depth = cmnd[troughs[deepest]];
            let unreached = (1..=THRESHOLDS).take_while(|&index| index as f64 / THRESHOLDS as f64 <= depth).count();
            weights[deepest] += NO_TROUGH_PROBABILITY * self.threshold_weights[..unreached].iter().sum::<f64>();

            for (&trough, &weight) in troughs.iter().zip(&weights) {
                if weight == 0.0 {
                    continue;
                }
                let period = (self.min_period + trough) as f64 + parabolic_shift(cmnd, trough);
                let hz = self.rate / period;
                // Rounded half to even, as numpy rounds.
                let bin = (12.0 * BINS_PER_SEMITONE as f64 * (hz / LOWEST_HZ).log2()).round_ties_even();
                let bin = bin.max(0.0) as usize;
                // Above the highest pitch, as librosa clips it into a state it then overwrites.
                if bin < self.bins {
                    probabilities[bin] = weight;
                }
            }
            voiced = probabilities[..self.bins].iter().sum::<f64>().clamp(0.0, 1.0);
        }
        probabilities[self.bins..].fill((1.0 - voiced) / self.bins as f64);
        probabilities
    }
}

/// Where between its neighbours a trough's parabola bottoms out, from −1 to 1,
/// or 0 at either end or where the parabola would put it further away.
fn parabolic_shift(values: &[f64], index: usize) -> f64 {
    if index == 0 || index + 1 >= values.len() {
        return 0.0;
    }
    let (before, at, after) = (values[index - 1], values[index], values[index + 1]);
    let curve = after + before - 2.0 * at;
    let slope = (after - before) / 2.0;
    if slope.abs() >= curve.abs() { 0.0 } else { -slope / curve }
}

/// The most likely sequence of states, by Viterbi decoding as frames arrive.
/// The pitch moves at most `reach` bins a frame, more likely the less it moves,
/// and voicing switches with `SWITCH_PROBABILITY`.
struct Path {
    bins: usize,
    reach: usize,
    /// Log-likelihood of the best path ending in each state.
    value: Vec<f64>,
    /// For each frame after the first, each state's best previous state.
    back: Vec<Vec<u16>>,
    /// `log_moves[to][from - lowest]`: how likely moving into bin `to` is from each
    /// bin in reach of it, from the lowest.
    log_moves: Vec<Vec<f64>>,
    started: bool,
}

impl Path {
    fn new(bins: usize, reach: usize) -> Self {
        let in_reach = |bin: usize| bin.saturating_sub(reach)..=(bin + reach).min(bins - 1);
        // Triangular around the current bin, normalised over the bins in its reach.
        let weight = |from: usize, to: usize| (reach + 1 - from.abs_diff(to)) as f64 / (reach + 1) as f64;
        let totals: Vec<f64> = (0..bins).map(|from| in_reach(from).map(|to| weight(from, to)).sum()).collect();
        let log_moves = (0..bins)
            .map(|to| in_reach(to).map(|from| (weight(from, to) / totals[from]).ln()).collect())
            .collect();
        Path { bins, reach, value: Vec::new(), back: Vec::new(), log_moves, started: false }
    }

    fn step(&mut self, observations: &[f64]) {
        let log = |probability: f64| (probability + f64::MIN_POSITIVE).ln();
        if !self.started {
            self.started = true;
            let initial = log(1.0 / observations.len() as f64);
            self.value = observations.iter().map(|&probability| log(probability) + initial).collect();
            return;
        }
        let stay = log(1.0 - SWITCH_PROBABILITY);
        let switch = log(SWITCH_PROBABILITY);
        let mut value = vec![0.0; 2 * self.bins];
        let mut back = vec![0u16; 2 * self.bins];
        for to in 0..self.bins {
            let lowest = to.saturating_sub(self.reach);
            let moves = &self.log_moves[to];
            // The best bin to come from, voiced and unvoiced, the lowest on a tie.
            let best = |voicing: usize| {
                let start = voicing * self.bins + lowest;
                let values = &self.value[start..start + moves.len()];
                let (mut best, mut best_from) = (f64::NEG_INFINITY, start);
                for (offset, (value, chance)) in values.iter().zip(moves).enumerate() {
                    if value + chance > best {
                        (best, best_from) = (value + chance, start + offset);
                    }
                }
                (best, best_from)
            };
            let (voiced, unvoiced) = (best(0), best(1));
            for to_voicing in 0..2 {
                let (from_voiced, from_unvoiced) = if to_voicing == 0 { (stay, switch) } else { (switch, stay) };
                let voiced = (voiced.0 + from_voiced, voiced.1);
                let unvoiced = (unvoiced.0 + from_unvoiced, unvoiced.1);
                // Ties go to the voiced state, as it comes first.
                let (best, best_from) = if unvoiced.0 > voiced.0 { unvoiced } else { voiced };
                let state = to_voicing * self.bins + to;
                value[state] = best + log(observations[state]);
                back[state] = best_from as u16;
            }
        }
        self.value = value;
        self.back.push(back);
    }

    /// The best path's states, one per frame.
    fn best(&self) -> Vec<usize> {
        if !self.started {
            return Vec::new();
        }
        let mut states = vec![first_largest(self.value.iter().copied())];
        for back in self.back.iter().rev() {
            states.push(usize::from(back[*states.last().unwrap_or(&0)]));
        }
        states.reverse();
        states
    }
}

/// The position of the largest value, the first if several are, as numpy's `argmax`.
fn first_largest(values: impl Iterator<Item = f64>) -> usize {
    let mut best = (0, f64::NEG_INFINITY);
    for (index, value) in values.enumerate() {
        if value > best.1 {
            best = (index, value);
        }
    }
    best.0
}

fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    values
}

/// A percentile of sorted values, interpolating between them as numpy does.
fn percentile(sorted: &[f64], percent: f64) -> f64 {
    let position = percent / 100.0 * (sorted.len() - 1) as f64;
    let (below, above) = (position.floor() as usize, position.ceil() as usize);
    sorted[below] + (sorted[above] - sorted[below]) * (position - below as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 24_000;

    /// A voiced sound: a fundamental following `hz` over time, with harmonics
    /// as a voice has, and a little noise.
    fn voice(seconds: f64, hz: impl Fn(f64) -> f64) -> Vec<f32> {
        let mut phase = 0.0;
        let mut noise = 0x2545_f491_u32;
        (0..(seconds * f64::from(RATE)) as usize)
            .map(|index| {
                let time = index as f64 / f64::from(RATE);
                phase += std::f64::consts::TAU * hz(time) / f64::from(RATE);
                let harmonics: f64 = (1..=5).map(|n| (n as f64 * phase).sin() / n as f64).sum();
                noise ^= noise << 13;
                noise ^= noise >> 17;
                noise ^= noise << 5;
                let hiss = (f64::from(noise) / f64::from(u32::MAX) - 0.5) * 0.01;
                (0.2 * harmonics + hiss) as f32
            })
            .collect()
    }

    /// A second of a pure tone at `hz`.
    fn sine(hz: f64, rate: u32) -> Vec<f32> {
        let rate = f64::from(rate);
        (0..rate as usize).map(|index| (std::f64::consts::TAU * hz * index as f64 / rate).sin() as f32).collect()
    }

    #[test]
    fn a_steady_tone_has_no_range() {
        let range = range_semitones(&voice(2.0, |_| 110.0), RATE).unwrap();
        assert!(range < 0.3, "{range}");
    }

    #[test]
    fn a_steady_tone_is_tracked_at_its_pitch() {
        let pitches: Vec<f64> = track(&voice(1.0, |_| 150.0), RATE).into_iter().flatten().collect();
        assert!(pitches.len() > 80, "{}", pitches.len());
        let median = percentile(&sorted(pitches), 50.0);
        assert!((median - 150.0).abs() < 1.5, "{median}");
    }

    #[test]
    fn a_glide_spans_its_interval_less_the_ends() {
        // An octave over 2 seconds, rising evenly in semitones, so the 5th to 95th
        // percentiles span 90% of it: 10.8 semitones.
        let glide = voice(2.0, |time| 80.0 * 2f64.powf(time / 2.0));
        let range = range_semitones(&glide, RATE).unwrap();
        assert!((range - 10.8).abs() < 0.5, "{range}");
    }

    #[test]
    fn a_glide_measures_the_same_at_any_sample_rate() {
        let glide = voice(2.0, |time| 100.0 * 2f64.powf(time / 4.0));
        let at_24 = range_semitones(&glide, RATE).unwrap();
        let halved: Vec<f32> = glide.iter().step_by(2).copied().collect();
        let at_12 = range_semitones(&halved, RATE / 2).unwrap();
        assert!((at_24 - at_12).abs() < 0.3, "{at_24} vs {at_12}");
    }

    #[test]
    fn resampling_keeps_a_tone_and_its_timing() {
        let resampled = resample(&sine(440.0, 24_000), 24_000, 16_000);
        let expected = sine(440.0, 16_000);
        assert_eq!(resampled.len(), expected.len());
        // Away from the ends, which the filter only half covers.
        for (got, wanted) in resampled.iter().zip(&expected).skip(200).take(15_600) {
            assert!((got - f64::from(*wanted)).abs() < 0.01, "{got} vs {wanted}");
        }
        assert_eq!(resample(&[0.5, -0.5], 16_000, 16_000), [0.5, -0.5]);
    }

    #[test]
    fn resampling_removes_what_the_new_rate_cannot_hold() {
        // 10 kHz is above 16 kHz's Nyquist frequency, so it would alias to 6 kHz.
        let resampled = resample(&sine(10_000.0, 24_000), 24_000, 16_000);
        let peak = resampled[200..15_800].iter().fold(0f64, |peak, sample| peak.max(sample.abs()));
        assert!(peak < 0.01, "{peak}");
    }

    #[test]
    fn silence_and_noise_have_no_range() {
        assert_eq!(range_semitones(&vec![0.0; RATE as usize * 2], RATE), None);
        assert_eq!(range_semitones(&[], RATE), None);
        let mut noise = 1u32;
        let hiss: Vec<f32> = (0..RATE * 2)
            .map(|_| {
                noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (f64::from(noise) / f64::from(u32::MAX) - 0.5) as f32 * 0.1
            })
            .collect();
        assert_eq!(range_semitones(&hiss, RATE), None);
    }

    #[test]
    fn too_little_speech_has_no_range() {
        assert_eq!(range_semitones(&voice(0.1, |_| 120.0), RATE), None);
    }

    #[test]
    fn closeness_is_1_at_the_target_and_falls_either_side() {
        assert_eq!(closeness(12.4, 12.4), 1.0);
        assert!((closeness(6.2, 12.4) - 0.5).abs() < 1e-6);
        assert!((closeness(24.8, 12.4) - 0.5).abs() < 1e-6);
        assert!(closeness(11.0, 12.4) > closeness(15.0, 12.4));
        assert_eq!(closeness(0.0, 12.4), 0.0);
    }

    #[test]
    fn a_target_must_be_from_2_to_30_semitones() {
        assert_eq!(parse_pitch_range("12.4"), Ok(12.4));
        assert_eq!(parse_pitch_range(" 2 "), Ok(2.0));
        assert_eq!(parse_pitch_range("30"), Ok(30.0));
        for bad in ["1.9", "31", "-12", "lively", ""] {
            assert_eq!(parse_pitch_range(bad), Err("must be a range from 2 to 30 semitones".to_string()), "{bad}");
        }
    }

    /// Prints the range of each WAV listed, one per line, in the file that
    /// `AUDIO_BERRY_PITCH_WAVS` names, to compare with another pitch tracker.
    #[test]
    #[ignore = "reads the WAVs listed in the file AUDIO_BERRY_PITCH_WAVS names"]
    fn print_pitch_ranges() {
        let list = std::env::var("AUDIO_BERRY_PITCH_WAVS").expect("AUDIO_BERRY_PITCH_WAVS names a list of WAVs");
        let paths = std::fs::read_to_string(list).unwrap();
        for path in paths.lines().filter(|line| !line.trim().is_empty()) {
            let audio = any_tts::AudioSamples::from_wav_file(path).unwrap();
            let started = std::time::Instant::now();
            let range = range_semitones(&audio.samples, audio.sample_rate);
            let range = range.map_or("none".to_string(), |range| format!("{range:.2}"));
            println!("{path}\t{range}\t{:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
        }
    }
}
