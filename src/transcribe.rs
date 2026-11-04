//! Checks generated speech against its text with a local transcriber, because
//! the model sometimes produces fluent nonsense that sounds clean and runs to a
//! plausible length.

use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Passages the user approved by ear scored 0.98-1.00 recall with 0.00-0.02 extra.
const MIN_RECALL: f32 = 0.92;
const MAX_EXTRA: f32 = 0.12;

/// Fast enough to check every take, at the cost of some misheard words.
pub const CHECK_MODEL: &str = "base.en";
/// A voice's transcript is only written once and every take depends on it.
pub const VOICE_TRANSCRIPTION_MODEL: &str = "medium.en";

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 8] = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

pub struct Accuracy {
    /// Share of the text's words heard, in order.
    pub recall: f32,
    /// Words heard that aren't in the text, relative to the text's length.
    pub extra: f32,
    pub transcript: String,
}

impl Accuracy {
    pub fn acceptable(&self) -> bool {
        self.recall >= MIN_RECALL && self.extra <= MAX_EXTRA
    }

    pub fn score(&self) -> f32 {
        self.recall - self.extra
    }
}

/// Lowercases into words, spelling out small numbers so "1" matches "one".
fn words(text: &str) -> Vec<String> {
    let cleaned: String = text
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' { c } else { ' ' })
        .collect();
    let mut out = Vec::new();
    for word in cleaned.split_whitespace() {
        let word = word.trim_matches('\'');
        match word.parse::<usize>() {
            Ok(number) if number < 20 => out.push(ONES[number].to_string()),
            Ok(number) if number < 100 => {
                out.push(TENS[number / 10 - 2].to_string());
                if number % 10 != 0 {
                    out.push(ONES[number % 10].to_string());
                }
            }
            _ if !word.is_empty() => out.push(word.to_string()),
            _ => {}
        }
    }
    out
}

fn common_subsequence(a: &[String], b: &[String]) -> usize {
    let mut row = vec![0usize; b.len() + 1];
    for word in a {
        let mut diagonal = 0;
        for (j, other) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = if word == other {
                diagonal + 1
            } else {
                above.max(row[j])
            };
            diagonal = above;
        }
    }
    row[b.len()]
}

pub fn compare(text: &str, transcript: &str) -> Accuracy {
    let expected = words(text);
    let heard = words(transcript);
    let matched = common_subsequence(&expected, &heard);
    let length = expected.len().max(1) as f32;
    Accuracy {
        recall: matched as f32 / length,
        extra: heard.len().saturating_sub(matched) as f32 / length,
        transcript: transcript.to_string(),
    }
}

/// Whether `text` has the phrase's words in a row, ignoring case and punctuation.
pub fn contains_phrase(text: &str, phrase: &str) -> bool {
    let phrase = words(phrase);
    !phrase.is_empty() && words(text).windows(phrase.len()).any(|window| window == phrase)
}

/// A long-running transcriber process, so its model only loads once.
pub struct Transcriber {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Transcriber {
    pub fn start(model: &str) -> Result<Self, Box<dyn Error>> {
        let script = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/transcribe.py");
        let mut child = Command::new("uv")
            .args(["run", "--no-project", "--with", "faster-whisper", "python", script, model])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("couldn't start the transcriber with uv: {error}"))?;
        let stdin = child.stdin.take().ok_or("transcriber has no stdin")?;
        let stdout = child.stdout.take().ok_or("transcriber has no stdout")?;
        let mut transcriber = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let ready = transcriber.read_line()?;
        if ready != "ready" {
            return Err(format!("transcriber didn't start: {ready}").into());
        }
        Ok(transcriber)
    }

    fn read_line(&mut self) -> Result<String, Box<dyn Error>> {
        let mut line = String::new();
        if self.stdout.read_line(&mut line)? == 0 {
            return Err("the transcriber stopped unexpectedly".into());
        }
        Ok(line.trim_end().to_string())
    }

    pub fn transcribe(&mut self, path: &Path) -> Result<String, Box<dyn Error>> {
        writeln!(self.stdin, "{}", path.display())?;
        self.stdin.flush()?;
        self.read_line()
    }
}

impl Drop for Transcriber {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_ignores_case_punctuation_and_digit_style() {
        let accuracy = compare("Chapter 12. Down the Rabbit-Hole!", "chapter twelve, down the rabbit hole");
        assert_eq!(accuracy.recall, 1.0);
        assert_eq!(accuracy.extra, 0.0);
        assert!(accuracy.acceptable());
    }

    #[test]
    fn phrases_match_whole_words_in_order() {
        assert!(contains_phrase("It took a Watch, out of its pocket", "took a watch out"));
        assert!(!contains_phrase("took a watt out", "took a watch"));
        assert!(!contains_phrase("watchman", "watch"));
    }

    #[test]
    fn nonsense_and_repeats_are_rejected() {
        let text = "the quick brown fox jumps over the lazy dog";
        assert!(!compare(text, "in what trade setups and cost free workers").acceptable());
        let repeated = format!("{text} {text}");
        let accuracy = compare(text, &repeated);
        assert_eq!(accuracy.recall, 1.0);
        assert!(!accuracy.acceptable());
    }
}
