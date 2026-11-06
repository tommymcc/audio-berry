mod audio;
mod audition;
mod book;
mod cast;
mod checks;
mod epub;
mod narrator;
mod quality;
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
  audio-berry audition <voice-sample> [speaker] [--speed <factor>]";

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
    let (args, speed) = take_speed(std::env::args().skip(1).collect())?;
    if speed.is_some() && !takes_speed(args.first().map(String::as_str)) {
        return Err(format!(
            "--speed only works with {}; retake uses the speed saved in the output directory",
            SPEED_COMMANDS.join(", ")
        )
        .into());
    }
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
        _ => Err(USAGE.into()),
    }
}

/// The commands that accept `--speed`. `retake` doesn't, as it reuses the saved speed.
const SPEED_COMMANDS: [&str; 3] = ["speak", "book", "audition"];

fn takes_speed(command: Option<&str>) -> bool {
    command.is_some_and(|command| SPEED_COMMANDS.contains(&command))
}

/// Takes `--speed <factor>` out of the arguments, wherever it appears.
fn take_speed(mut args: Vec<String>) -> Result<(Vec<String>, Option<f64>), Box<dyn Error>> {
    let Some(flag) = args.iter().position(|arg| arg == "--speed") else {
        return Ok((args, None));
    };
    let value = args.get(flag + 1).ok_or("--speed needs a factor, such as 0.9")?;
    let speed = value.parse::<f64>().ok().filter(|speed| SPEED_RANGE.contains(speed));
    let speed = speed.ok_or_else(|| {
        format!("--speed must be from {} to {}, not {value}", SPEED_RANGE.start(), SPEED_RANGE.end())
    })?;
    args.drain(flag..=flag + 1);
    Ok((args, Some(speed)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| word.to_string()).collect()
    }

    #[test]
    fn speed_is_taken_from_any_position() {
        let (rest, speed) = take_speed(args(&["book", "--speed", "0.9", "v", "b", "o", "3"])).unwrap();
        assert_eq!(rest, args(&["book", "v", "b", "o", "3"]));
        assert_eq!(speed, Some(0.9));
        let (rest, speed) = take_speed(args(&["book", "v", "b", "o", "--speed", "1.2"])).unwrap();
        assert_eq!(rest, args(&["book", "v", "b", "o"]));
        assert_eq!(speed, Some(1.2));
    }

    #[test]
    fn speed_is_for_commands_that_render_at_a_chosen_speed() {
        for command in SPEED_COMMANDS {
            assert!(takes_speed(Some(command)), "{command}");
        }
        for command in ["retake", "play", "chapters", "speakers"] {
            assert!(!takes_speed(Some(command)), "{command}");
        }
        assert!(!takes_speed(None));
    }

    #[test]
    fn no_flag_means_no_speed() {
        assert_eq!(take_speed(args(&["chapters", "b"])).unwrap().1, None);
    }

    #[test]
    fn a_missing_or_silly_speed_is_an_error() {
        for bad in ["0", "-1", "0.4", "2.1", "fast", "NaN"] {
            assert!(take_speed(args(&["book", "--speed", bad])).is_err(), "{bad}");
        }
        assert!(take_speed(args(&["book", "--speed"])).is_err());
    }
}
