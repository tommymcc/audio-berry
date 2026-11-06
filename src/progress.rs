//! Shows progress through a run's chunks: a live bar on a terminal, or one plain
//! line per message when the output goes to a log, so logs read as they always have.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressState, ProgressStyle};
use std::io::{self, IsTerminal, Stdout, Write};
use std::time::{Duration, Instant};

/// Progress through `total` chunks, chosen once as a bar or plain lines.
pub struct Progress<W: Write = Stdout> {
    show: Show<W>,
}

enum Show<W> {
    Bar(ProgressBar),
    Lines(Lines<W>),
}

/// The plain lines written when the output isn't a terminal.
struct Lines<W> {
    out: W,
    total: usize,
    done: usize,
    started: Instant,
    /// Whether each finished chunk gets a line with its summary and the time left.
    timed: bool,
}

impl Progress {
    /// Progress where, without a terminal, each finished chunk prints its summary
    /// with the time elapsed and left.
    pub fn new(total: usize) -> Self {
        Self::on_stdout(total, true)
    }

    /// Progress where, without a terminal, a finished chunk prints nothing, for a
    /// caller that already prints its own lines per chunk.
    pub fn quiet(total: usize) -> Self {
        Self::on_stdout(total, false)
    }

    fn on_stdout(total: usize, timed: bool) -> Self {
        let out = io::stdout();
        if out.is_terminal() {
            Progress { show: Show::Bar(bar(total)) }
        } else {
            Self::lines(out, total, timed)
        }
    }
}

impl<W: Write> Progress<W> {
    fn lines(out: W, total: usize, timed: bool) -> Self {
        Progress { show: Show::Lines(Lines { out, total, done: 0, started: Instant::now(), timed }) }
    }

    /// Prints a message, above the bar if there is one.
    pub fn line(&mut self, message: &str) -> io::Result<()> {
        match &mut self.show {
            Show::Bar(bar) => {
                bar.println(message);
                Ok(())
            }
            Show::Lines(lines) => writeln!(lines.out, "{message}"),
        }
    }

    /// Counts a finished chunk, with a summary of it such as its take and recall.
    pub fn chunk_done(&mut self, summary: &str) -> io::Result<()> {
        match &mut self.show {
            Show::Bar(bar) => {
                bar.set_message(summary.to_string());
                bar.inc(1);
                Ok(())
            }
            Show::Lines(lines) => {
                lines.done += 1;
                if !lines.timed {
                    return Ok(());
                }
                let line = chunk_line(lines.done, lines.total, summary, lines.started.elapsed());
                writeln!(lines.out, "{line}")
            }
        }
    }

    /// Leaves the finished bar in place, so later lines print below it.
    pub fn finish(self) {
        if let Show::Bar(bar) = self.show {
            bar.finish();
        }
    }
}

/// The line printed for a finished chunk, such as
/// `[3/40] 22s of audio, take 1, recall 0.98, 2 min elapsed, about 25 min left`.
fn chunk_line(done: usize, total: usize, summary: &str, elapsed: Duration) -> String {
    let elapsed = elapsed.as_secs_f32();
    format!(
        "[{done}/{total}] {summary}, {:.0} min elapsed, about {:.0} min left",
        elapsed / 60.0,
        minutes_left(elapsed, done, total),
    )
}

/// Minutes left at this run's average time per chunk so far.
fn minutes_left(elapsed_secs: f32, done: usize, total: usize) -> f32 {
    elapsed_secs / done as f32 * total.saturating_sub(done) as f32 / 60.0
}

fn bar(total: usize) -> ProgressBar {
    let style = ProgressStyle::with_template(
        "[{pos}/{len}] {wide_bar} {percent}%, {minutes} min elapsed, {left}\n{wide_msg}",
    )
    .expect("the template is valid")
    .with_key("minutes", |state: &ProgressState, out: &mut dyn std::fmt::Write| {
        write!(out, "{:.0}", state.elapsed().as_secs_f32() / 60.0).ok();
    })
    .with_key("left", |state: &ProgressState, out: &mut dyn std::fmt::Write| {
        let done = state.pos() as usize;
        let total = state.len().unwrap_or(0) as usize;
        if done == 0 {
            write!(out, "time left unknown").ok();
        } else {
            let left = minutes_left(state.elapsed().as_secs_f32(), done, total);
            write!(out, "about {left:.0} min left").ok();
        }
    });
    let bar = ProgressBar::with_draw_target(Some(total as u64), ProgressDrawTarget::stdout()).with_style(style);
    // Chunks take many seconds each, so tick to keep the elapsed time moving.
    bar.enable_steady_tick(Duration::from_secs(1));
    bar
}

#[cfg(test)]
mod tests {
    use super::*;

    fn written(progress: Progress<Vec<u8>>) -> String {
        match progress.show {
            Show::Lines(lines) => String::from_utf8(lines.out).unwrap(),
            Show::Bar(_) => unreachable!(),
        }
    }

    #[test]
    fn a_chunk_line_matches_the_log_format() {
        let line = chunk_line(3, 40, "22s of audio, take 1, recall 0.98", Duration::from_secs(150));
        // 150s for 3 chunks is 50s each; 37 left is 1850s.
        assert_eq!(line, "[3/40] 22s of audio, take 1, recall 0.98, 2 min elapsed, about 31 min left");
    }

    #[test]
    fn the_last_chunk_has_no_time_left() {
        let line = chunk_line(5, 5, "9s of audio, take 2, recall 1.00", Duration::from_secs(600));
        assert_eq!(line, "[5/5] 9s of audio, take 2, recall 1.00, 10 min elapsed, about 0 min left");
    }

    #[test]
    fn time_left_is_the_average_per_chunk_times_those_left() {
        assert_eq!(minutes_left(120.0, 2, 12), 10.0);
        assert_eq!(minutes_left(60.0, 1, 1), 0.0);
    }

    #[test]
    fn plain_lines_print_messages_and_finished_chunks_in_order() {
        let mut progress = Progress::lines(Vec::new(), 2, true);
        progress.line("  take 1 rejected: floor -51 dB, 19.2s, recall 0.94, extra 0.04").unwrap();
        progress.chunk_done("19s of audio, take 2, recall 0.97").unwrap();
        progress.chunk_done("4s of audio, take 1, recall 1.00").unwrap();
        assert_eq!(
            written(progress),
            "  take 1 rejected: floor -51 dB, 19.2s, recall 0.94, extra 0.04\n\
             [1/2] 19s of audio, take 2, recall 0.97, 0 min elapsed, about 0 min left\n\
             [2/2] 4s of audio, take 1, recall 1.00, 0 min elapsed, about 0 min left\n"
        );
    }

    #[test]
    fn quiet_plain_lines_print_only_messages() {
        let mut progress = Progress::lines(Vec::new(), 1, false);
        progress.line("[1/1] chunk 3.12: Alice took a watch out").unwrap();
        progress.chunk_done("chunk 3.12 replaced").unwrap();
        assert_eq!(written(progress), "[1/1] chunk 3.12: Alice took a watch out\n");
    }

    /// Shows the bar with no model: run with
    /// `cargo test --release bar_demo -- --ignored --nocapture` in a terminal.
    #[test]
    #[ignore = "draws to the terminal and takes a few seconds"]
    fn bar_demo() {
        let mut progress = Progress::new(5);
        for chunk in 1..=5 {
            std::thread::sleep(Duration::from_millis(700));
            if chunk % 2 == 0 {
                progress.line(&format!("  take 1 rejected: floor -51 dB, {chunk}.0s, recall 0.94, extra 0.04")).unwrap();
            }
            progress.chunk_done(&format!("{chunk}s of audio, take 1, recall 0.98")).unwrap();
        }
        progress.finish();
        println!("Wrote demo.m4b");
    }
}
