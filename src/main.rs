mod accuracy;
mod audio;
mod audition;
mod book;
mod cast;
mod checks;
mod clean;
mod epub;
mod narrator;
mod quality;
mod sample_check;
mod sections;
mod store;
mod text;
mod transcribe;
mod voice;

use std::error::Error;
use std::path::{Path, PathBuf};

/// Beyond this, the length checks and Whisper haven't been tried.
const SPEED_RANGE: std::ops::RangeInclusive<f64> = 0.5..=2.0;

const USAGE: &str = "usage:
  audio-berry speak <voice-sample> <text> <output.wav> [--speed <factor>]
  audio-berry chapters <book.epub>
  audio-berry book <voice-sample> <book.epub> <output-dir> [chapter numbers, e.g. 2 3 7] [--speed <factor>]
  audio-berry retake <voice-sample> <book.epub> <output-dir> <words from the book, or a chunk number like 3.12>...
  audio-berry speakers
  audio-berry play <speaker> <file>...
  audio-berry audition <voice-sample> [speaker] [--speed <factor>]
  audio-berry sections <recording> [count] [--reduction <dB>]
  audio-berry clean <recording> <start> <end> <output.wav> [--reduction <dB>]";

/// Rust ignores SIGPIPE, so printing to a reader that has gone, such as `head`, panics.
/// Restoring the default makes the program stop quietly like other Unix tools.
#[cfg(unix)]
fn stop_quietly_when_output_closes() {
    // SAFETY: runs first in main, before any other threads exist.
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };
}

#[cfg(not(unix))]
fn stop_quietly_when_output_closes() {}

fn main() -> Result<(), Box<dyn Error>> {
    stop_quietly_when_output_closes();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let speed = SPEED.take(&mut args)?.map(|value| parse_speed(&value)).transpose()?;
    let reduction = REDUCTION.take(&mut args)?.map(|value| clean::parse_reduction(&value)).transpose()?;
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["speak", voice, text, output] => narrator::speak(Path::new(voice), text, Path::new(output), speed),
        ["chapters", book] => book::list_chapters(Path::new(book)),
        ["book", voice, book, output, chapters @ ..] => {
            let chapters = if chapters.is_empty() {
                None
            } else {
                let numbers = chapters.iter().map(|number| number.parse::<usize>());
                Some(numbers.collect::<Result<Vec<_>, _>>().map_err(|_| USAGE)?)
            };
            book::narrate(Path::new(voice), Path::new(book), Path::new(output), chapters, speed)
        }
        ["retake", voice, book, output, targets @ ..]
            if !targets.is_empty() && !targets.iter().any(|target| target.starts_with("--")) =>
        {
            book::retake(Path::new(voice), Path::new(book), Path::new(output), targets)
        }
        ["speakers"] => {
            for speaker in cast::discover()? {
                println!("{} ({})", speaker.name, speaker.ip);
            }
            Ok(())
        }
        ["play", speaker, files @ ..] if !files.is_empty() => {
            let files: Vec<PathBuf> = files.iter().map(PathBuf::from).collect();
            if let Some(missing) = files.iter().find(|file| !file.is_file()) {
                return Err(format!("no such file: {}", missing.display()).into());
            }
            cast::play(&cast::find(speaker)?, files)
        }
        ["audition", voice] => audition::audition(Path::new(voice), None, speed),
        ["audition", voice, speaker] => audition::audition(Path::new(voice), Some(speaker), speed),
        ["sections", recording] => sections::suggest(Path::new(recording), sections::DEFAULT_COUNT, reduction),
        ["sections", recording, count] => {
            let count = count.parse::<usize>().ok().filter(|count| *count > 0);
            let count = count.ok_or("the count of sections must be a whole number from 1")?;
            sections::suggest(Path::new(recording), count, reduction)
        }
        ["clean", recording, start, end, output] => {
            let (start, end) = (clean::parse_time(start)?, clean::parse_time(end)?);
            let reduction = reduction.unwrap_or(clean::DEFAULT_REDUCTION_DB);
            clean::clean(Path::new(recording), start, end, Path::new(output), reduction)
        }
        _ => Err(USAGE.into()),
    }
}

/// An option some commands take, given as `<name> <value>` anywhere in the arguments.
struct Flag {
    name: &'static str,
    /// What the value is, for when it's missing.
    value: &'static str,
    commands: &'static [&'static str],
    /// Added when another command is given it.
    refused_note: &'static str,
}

const SPEED: Flag = Flag {
    name: "--speed",
    value: "a factor, such as 0.9",
    commands: &["speak", "book", "audition"],
    refused_note: "; retake uses the speed saved in the output directory",
};

const REDUCTION: Flag = Flag {
    name: "--reduction",
    value: "a level in dB, such as 15",
    commands: &["clean", "sections"],
    refused_note: "",
};

impl Flag {
    fn accepted_by(&self, command: Option<&str>) -> bool {
        command.is_some_and(|command| self.commands.contains(&command))
    }

    /// Takes the flag and its value out of the arguments, if given, refusing it
    /// for a command that doesn't take it.
    fn take(&self, args: &mut Vec<String>) -> Result<Option<String>, Box<dyn Error>> {
        let Some(position) = args.iter().position(|arg| arg == self.name) else {
            return Ok(None);
        };
        let value = args.get(position + 1).cloned();
        args.drain(position..(position + 2).min(args.len()));
        if !self.accepted_by(args.first().map(String::as_str)) {
            let commands = self.commands.join(", ");
            return Err(format!("{} only works with {commands}{}", self.name, self.refused_note).into());
        }
        Ok(Some(value.ok_or_else(|| format!("{} needs {}", self.name, self.value))?))
    }
}

fn parse_speed(value: &str) -> Result<f64, String> {
    let speed = value.parse::<f64>().ok().filter(|speed| SPEED_RANGE.contains(speed));
    speed.ok_or_else(|| format!("--speed must be from {} to {}, not {value}", SPEED_RANGE.start(), SPEED_RANGE.end()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| word.to_string()).collect()
    }

    /// Takes a flag, then parses its value as main does.
    fn take<T>(flag: &Flag, words: &[&str], parse: fn(&str) -> Result<T, String>) -> Result<(Vec<String>, Option<T>), Box<dyn Error>> {
        let mut rest = args(words);
        let value = flag.take(&mut rest)?.map(|value| parse(&value)).transpose()?;
        Ok((rest, value))
    }

    #[test]
    fn speed_is_taken_from_any_position() {
        let (rest, speed) = take(&SPEED, &["book", "--speed", "0.9", "v", "b", "o", "3"], parse_speed).unwrap();
        assert_eq!(rest, args(&["book", "v", "b", "o", "3"]));
        assert_eq!(speed, Some(0.9));
        let (rest, speed) = take(&SPEED, &["book", "v", "b", "o", "--speed", "1.2"], parse_speed).unwrap();
        assert_eq!(rest, args(&["book", "v", "b", "o"]));
        assert_eq!(speed, Some(1.2));
    }

    #[test]
    fn speed_is_for_commands_that_render_at_a_chosen_speed() {
        for command in ["speak", "book", "audition"] {
            assert!(SPEED.accepted_by(Some(command)), "{command}");
        }
        for command in ["retake", "play", "chapters", "speakers", "clean"] {
            assert!(!SPEED.accepted_by(Some(command)), "{command}");
        }
        assert!(!SPEED.accepted_by(None));
    }

    #[test]
    fn no_flag_means_no_value() {
        assert_eq!(take(&SPEED, &["chapters", "b"], parse_speed).unwrap().1, None);
        assert_eq!(take(&REDUCTION, &["clean", "r", "1", "2", "o"], clean::parse_reduction).unwrap().1, None);
    }

    #[test]
    fn a_missing_or_silly_speed_is_an_error() {
        for bad in ["0", "-1", "0.4", "2.1", "fast", "NaN"] {
            assert!(take(&SPEED, &["book", "--speed", bad], parse_speed).is_err(), "{bad}");
        }
        assert!(take(&SPEED, &["book", "--speed"], parse_speed).is_err());
    }

    #[test]
    fn reduction_is_taken_from_any_position() {
        let words = ["clean", "--reduction", "20", "r", "1", "2", "o"];
        let (rest, reduction) = take(&REDUCTION, &words, clean::parse_reduction).unwrap();
        assert_eq!(rest, args(&["clean", "r", "1", "2", "o"]));
        assert_eq!(reduction, Some(20.0));
    }

    #[test]
    fn reduction_is_for_commands_that_clean() {
        let (rest, reduction) = take(&REDUCTION, &["sections", "r", "--reduction", "20"], clean::parse_reduction).unwrap();
        assert_eq!(rest, args(&["sections", "r"]));
        assert_eq!(reduction, Some(20.0));
        for command in ["speak", "book", "audition", "retake"] {
            assert!(!REDUCTION.accepted_by(Some(command)), "{command}");
        }
    }

    #[test]
    fn a_missing_or_silly_reduction_is_an_error() {
        for bad in ["0", "-15", "101", "lots"] {
            assert!(take(&REDUCTION, &["clean", "--reduction", bad], clean::parse_reduction).is_err(), "{bad}");
        }
        assert!(take(&REDUCTION, &["clean", "--reduction"], clean::parse_reduction).is_err());
    }

    #[test]
    fn a_flag_for_another_command_is_refused_by_name_whatever_its_value() {
        for (flag, words) in [(&REDUCTION, ["book", "--reduction", "lots"]), (&SPEED, ["clean", "--speed", "fast"])] {
            let error = flag.take(&mut args(&words)).unwrap_err().to_string();
            assert!(error.starts_with(&format!("{} only works with", flag.name)), "{error}");
        }
    }
}
