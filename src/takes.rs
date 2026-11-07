//! How many takes of a chunk to render, and which of them to keep.

use color_eyre::eyre::{Result, eyre};
use std::fmt;

/// The most takes `--takes` can ask for. Each costs a render and a transcription.
pub const TAKES_RANGE: std::ops::RangeInclusive<u64> = 1..=10;

/// How many takes of a chunk `book` and `retake` render, and which they keep.
/// The output directory saves it in `narration.txt`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TakePolicy {
    /// Renders takes until one passes, up to the command's own limit (5 for
    /// `book`, 6 for `retake`), keeping that one, or the best if none passes.
    #[default]
    FirstPassing,
    /// Renders exactly this many takes and keeps the best, from `--takes`.
    BestOf(u64),
}

/// A `--takes` count given on the command line.
pub fn parse_takes(value: &str) -> Result<TakePolicy, String> {
    let count = value.trim().parse::<u64>().ok().filter(|count| TAKES_RANGE.contains(count));
    count
        .map(TakePolicy::BestOf)
        .ok_or_else(|| format!("must be a whole number from {} to {}", TAKES_RANGE.start(), TAKES_RANGE.end()))
}

impl TakePolicy {
    /// How `narration.txt` saves it, such as `first-passing` or `best-of-3`.
    pub fn saved(self) -> String {
        match self {
            TakePolicy::FirstPassing => "first-passing".into(),
            TakePolicy::BestOf(count) => format!("best-of-{count}"),
        }
    }

    /// Reads what `saved` wrote, if it's a policy `--takes` could have given.
    pub fn from_saved(value: &str) -> Option<Self> {
        match value.trim() {
            "first-passing" => Some(TakePolicy::FirstPassing),
            other => parse_takes(other.strip_prefix("best-of-")?).ok(),
        }
    }

    /// Names the take kept, numbered from 1, for the progress output.
    pub fn describe(self, kept: u64) -> String {
        match self {
            TakePolicy::FirstPassing => format!("take {}", kept + 1),
            TakePolicy::BestOf(count) => format!("take {}, the best of {count}", kept + 1),
        }
    }

    /// Renders takes `first`, `first + 1` and so on with `render`, and keeps the
    /// best. `judge` says whether a take passes and how it ranks, and `limit` is
    /// how many `FirstPassing` tries. Ties keep the earlier take.
    pub fn record<T, R: PartialOrd>(
        self,
        first: u64,
        limit: u64,
        mut render: impl FnMut(u64) -> Result<T>,
        judge: impl Fn(&T) -> (bool, R),
    ) -> Result<Kept<T>> {
        let count = match self {
            TakePolicy::FirstPassing => limit,
            TakePolicy::BestOf(count) => count,
        };
        let mut best: Option<(Kept<T>, (bool, R))> = None;
        for take in first..first + count {
            let current = render(take)?;
            let judged = judge(&current);
            let passes = judged.0;
            let kept = Kept { take: current, number: take, rendered: take + 1 - first };
            if self == TakePolicy::FirstPassing && passes {
                return Ok(kept);
            }
            if best.as_ref().is_none_or(|(_, existing)| beats(&judged, existing)) {
                best = Some((kept, judged));
            }
        }
        let (mut kept, _) = best.ok_or_else(|| eyre!("no takes were rendered"))?;
        kept.rendered = count;
        Ok(kept)
    }
}

/// Whether one take is better than another, given whether each passes and how
/// each ranks: a passing take always beats a failing one, and otherwise the
/// higher rank wins.
pub fn beats<R: PartialOrd>(take: &(bool, R), other: &(bool, R)) -> bool {
    take > other
}

impl fmt::Display for TakePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TakePolicy::FirstPassing => write!(f, "the first that passes"),
            TakePolicy::BestOf(count) => write!(f, "the best of {count}"),
        }
    }
}

/// The take kept from those `record` rendered.
pub struct Kept<T> {
    pub take: T,
    /// Which take it is, counting from 0, which picks its seed.
    pub number: u64,
    /// How many takes were rendered, including this one.
    pub rendered: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Renders made-up takes, each whether it passes and its rank, and records
    /// which takes were rendered.
    fn record(policy: TakePolicy, first: u64, takes: &[(bool, f32)]) -> (Kept<(bool, f32)>, Vec<u64>) {
        let mut rendered = Vec::new();
        let kept = policy
            .record(first, 5, |take| {
                rendered.push(take);
                Ok(takes[(take - first) as usize])
            }, |take| *take)
            .unwrap();
        (kept, rendered)
    }

    #[test]
    fn the_default_keeps_the_first_take_that_passes() {
        let (kept, rendered) = record(TakePolicy::FirstPassing, 0, &[(false, 0.9), (true, 0.5), (true, 1.0)]);
        assert_eq!((kept.number, kept.rendered, kept.take), (1, 2, (true, 0.5)));
        assert_eq!(rendered, [0, 1]);
    }

    #[test]
    fn the_default_keeps_the_best_of_its_limit_when_none_passes() {
        let takes = [(false, 0.2), (false, 0.8), (false, 0.5), (false, 0.8), (false, 0.1), (true, 1.0)];
        let (kept, rendered) = record(TakePolicy::FirstPassing, 0, &takes);
        assert_eq!((kept.number, kept.rendered), (1, 5), "a tie keeps the earlier take");
        assert_eq!(rendered, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn best_of_renders_every_take_and_keeps_the_best_that_passes() {
        let takes = [(true, 0.5), (false, 1.0), (true, 0.9), (true, 0.7)];
        let (kept, rendered) = record(TakePolicy::BestOf(4), 0, &takes);
        assert_eq!((kept.number, kept.rendered, kept.take), (2, 4, (true, 0.9)));
        assert_eq!(rendered, [0, 1, 2, 3]);
    }

    #[test]
    fn best_of_keeps_the_best_failing_take_when_none_passes() {
        let (kept, _) = record(TakePolicy::BestOf(3), 0, &[(false, 0.5), (false, 0.7), (false, 0.6)]);
        assert_eq!((kept.number, kept.rendered), (1, 3));
    }

    #[test]
    fn best_of_one_renders_only_the_first_take() {
        let (kept, rendered) = record(TakePolicy::BestOf(1), 0, &[(false, 0.1), (true, 1.0)]);
        assert_eq!((kept.number, kept.rendered), (0, 1));
        assert_eq!(rendered, [0]);
    }

    #[test]
    fn takes_after_seeds_already_tried_are_numbered_on_from_them() {
        let (kept, rendered) = record(TakePolicy::BestOf(2), 7, &[(true, 0.4), (true, 0.6)]);
        assert_eq!((kept.number, kept.rendered), (8, 2));
        assert_eq!(rendered, [7, 8]);
    }

    #[test]
    fn a_passing_take_beats_any_failing_one() {
        assert!(beats(&(true, 0.1), &(false, 1.0)));
        assert!(!beats(&(false, 1.0), &(true, 0.1)));
        assert!(beats(&(true, 0.6), &(true, 0.5)));
        assert!(!beats(&(true, 0.5), &(true, 0.5)));
    }

    #[test]
    fn takes_must_be_from_1_to_10() {
        assert_eq!(parse_takes("3"), Ok(TakePolicy::BestOf(3)));
        assert_eq!(parse_takes("10"), Ok(TakePolicy::BestOf(10)));
        for bad in ["0", "11", "-1", "2.5", "many", ""] {
            assert_eq!(parse_takes(bad), Err("must be a whole number from 1 to 10".to_string()), "{bad}");
        }
    }

    #[test]
    fn a_policy_reads_back_as_saved() {
        for policy in [TakePolicy::FirstPassing, TakePolicy::BestOf(1), TakePolicy::BestOf(10)] {
            assert_eq!(TakePolicy::from_saved(&policy.saved()), Some(policy));
        }
        assert_eq!(TakePolicy::BestOf(3).saved(), "best-of-3");
        for bad in ["best-of-0", "best-of-11", "3", "best", ""] {
            assert_eq!(TakePolicy::from_saved(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_kept_take_is_named_from_1() {
        assert_eq!(TakePolicy::FirstPassing.describe(1), "take 2");
        assert_eq!(TakePolicy::BestOf(5).describe(2), "take 3, the best of 5");
    }
}
