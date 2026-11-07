//! How many takes of a chunk to render, and which of them to keep.

use color_eyre::eyre::{Result, eyre};
use std::fmt;

/// The counts `--takes` can ask for. Each take costs a render and a transcription.
pub const TAKES_RANGE: std::ops::RangeInclusive<u64> = 1..=10;

/// How many takes of a chunk `book` and `retake` render, and which they keep.
/// The output directory saves it in `narration.txt`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TakePolicy {
    /// Renders takes until one is good enough, up to the command's own limit,
    /// keeping the best rendered. For `book` that's a take that passes, of up to
    /// 5; for `retake` it's a flawless take with its phrases heard, of up to 6.
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

/// How a take compares with the others of its chunk: a take that passes beats
/// any that doesn't, and otherwise the higher rank wins.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Judged<R> {
    pub passes: bool,
    pub rank: R,
}

/// The take kept from those `render_best` rendered.
pub struct Kept<T> {
    pub take: T,
    /// Which take it is, counting from 0, which picks its seed.
    pub index: u64,
    /// The index of the first take rendered.
    pub first: u64,
    /// How many takes were rendered.
    pub rendered: u64,
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

    /// Names the take `kept`, numbered from 1, for the progress output.
    pub fn describe<T>(self, kept: &Kept<T>) -> String {
        let number = kept.index + 1;
        match self {
            TakePolicy::FirstPassing => format!("take {number}"),
            TakePolicy::BestOf(count) if kept.first == 0 => format!("take {number}, the best of {count}"),
            TakePolicy::BestOf(_) => {
                format!("take {number}, the best of takes {}–{}", kept.first + 1, kept.first + kept.rendered)
            }
        }
    }

    /// Renders takes `first`, `first + 1` and so on with `render`, and keeps the
    /// one `judge` puts highest, the earlier on a tie. `FirstPassing` stops after
    /// `limit` takes, or sooner at a take that's `done`.
    pub fn render_best<T, R: PartialOrd>(
        self,
        first: u64,
        limit: u64,
        mut render: impl FnMut(u64) -> Result<T>,
        judge: impl Fn(&T) -> Judged<R>,
        done: impl Fn(&T) -> bool,
    ) -> Result<Kept<T>> {
        let count = match self {
            TakePolicy::FirstPassing => limit,
            TakePolicy::BestOf(count) => count,
        };
        let mut best: Option<(T, u64, Judged<R>)> = None;
        let mut rendered = 0;
        for index in first..first + count {
            let current = render(index)?;
            rendered += 1;
            let stop = self == TakePolicy::FirstPassing && done(&current);
            let judged = judge(&current);
            if best.as_ref().is_none_or(|(_, _, existing)| judged > *existing) {
                best = Some((current, index, judged));
            }
            if stop {
                break;
            }
        }
        let (take, index, _) = best.ok_or_else(|| eyre!("no takes were rendered"))?;
        Ok(Kept { take, index, first, rendered })
    }
}

impl fmt::Display for TakePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TakePolicy::FirstPassing => write!(f, "the first good enough"),
            TakePolicy::BestOf(count) => write!(f, "the best of {count}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A made-up take: whether it passes, its rank, and whether it's done.
    type Fake = (bool, f32, bool);

    /// Renders made-up takes, and records which were rendered.
    fn render(policy: TakePolicy, first: u64, takes: &[Fake]) -> (Kept<Fake>, Vec<u64>) {
        let mut rendered = Vec::new();
        let render = |index: u64| {
            rendered.push(index);
            Ok(takes[(index - first) as usize])
        };
        let judge = |take: &Fake| Judged { passes: take.0, rank: take.1 };
        let kept = policy.render_best(first, 5, render, judge, |take| take.2).unwrap();
        (kept, rendered)
    }

    #[test]
    fn the_default_stops_at_the_first_take_that_is_done() {
        let takes = [(false, 0.9, false), (true, 0.5, true), (true, 1.0, true)];
        let (kept, rendered) = render(TakePolicy::FirstPassing, 0, &takes);
        assert_eq!((kept.index, kept.rendered, kept.take), (1, 2, (true, 0.5, true)));
        assert_eq!(rendered, [0, 1]);
    }

    #[test]
    fn the_default_keeps_a_better_take_than_the_one_it_stops_at() {
        // As `retake` does, where a take can rank above the one that stops it.
        let takes = [(false, 0.9, false), (false, 0.5, true)];
        let (kept, rendered) = render(TakePolicy::FirstPassing, 0, &takes);
        assert_eq!((kept.index, kept.rendered), (0, 2));
        assert_eq!(rendered, [0, 1]);
    }

    #[test]
    fn the_default_keeps_the_best_of_its_limit_when_none_is_done() {
        let failing = |rank| (false, rank, false);
        let takes = [failing(0.2), failing(0.8), failing(0.5), failing(0.8), failing(0.1), (true, 1.0, true)];
        let (kept, rendered) = render(TakePolicy::FirstPassing, 0, &takes);
        assert_eq!((kept.index, kept.rendered), (1, 5), "a tie keeps the earlier take");
        assert_eq!(rendered, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn best_of_renders_every_take_and_keeps_the_best_that_passes() {
        let takes = [(true, 0.5, true), (false, 1.0, false), (true, 0.9, true), (true, 0.7, true)];
        let (kept, rendered) = render(TakePolicy::BestOf(4), 0, &takes);
        assert_eq!((kept.index, kept.rendered, kept.take), (2, 4, (true, 0.9, true)));
        assert_eq!(rendered, [0, 1, 2, 3]);
    }

    #[test]
    fn best_of_keeps_the_best_failing_take_when_none_passes() {
        let (kept, _) = render(TakePolicy::BestOf(3), 0, &[(false, 0.5, false), (false, 0.7, false), (false, 0.6, false)]);
        assert_eq!((kept.index, kept.rendered), (1, 3));
    }

    #[test]
    fn best_of_one_renders_only_the_first_take() {
        let (kept, rendered) = render(TakePolicy::BestOf(1), 0, &[(false, 0.1, false), (true, 1.0, true)]);
        assert_eq!((kept.index, kept.rendered), (0, 1));
        assert_eq!(rendered, [0]);
    }

    #[test]
    fn takes_after_seeds_already_tried_are_numbered_on_from_them() {
        let (kept, rendered) = render(TakePolicy::BestOf(3), 7, &[(true, 0.4, true), (true, 0.6, true), (true, 0.5, true)]);
        assert_eq!((kept.index, kept.first, kept.rendered), (8, 7, 3));
        assert_eq!(rendered, [7, 8, 9]);
        assert_eq!(TakePolicy::BestOf(3).describe(&kept), "take 9, the best of takes 8–10");
    }

    #[test]
    fn a_passing_take_beats_any_failing_one() {
        let judged = |passes, rank| Judged { passes, rank };
        assert!(judged(true, 0.1) > judged(false, 1.0));
        assert!(judged(true, 0.6) > judged(true, 0.5));
        assert!(judged(false, 0.6) > judged(false, 0.5));
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
        let kept = Kept { take: (), index: 2, first: 0, rendered: 5 };
        assert_eq!(TakePolicy::FirstPassing.describe(&kept), "take 3");
        assert_eq!(TakePolicy::BestOf(5).describe(&kept), "take 3, the best of 5");
    }
}
