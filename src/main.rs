mod accuracy;
mod assemble;
mod audio;
mod audition;
mod book;
mod cast;
mod checks;
mod clean;
mod epub;
mod files;
mod mishearings;
mod narrator;
mod progress;
mod quality;
mod sample_check;
mod sections;
mod shell;
mod store;
#[cfg(test)]
mod test_dir;
mod text;
mod transcribe;
mod voice;

use clap::{Parser, Subcommand};
use color_eyre::config::{HookBuilder, Theme};
use color_eyre::eyre::{Result, bail};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

/// Beyond this, the length checks and Whisper haven't been tried.
const SPEED_RANGE: std::ops::RangeInclusive<f64> = 0.5..=2.0;

/// Narrates DRM-free EPUB books locally, in a voice cloned from a short sample.
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Says some text in the voice of a sample
    Speak {
        /// A voice sample, such as voices/me.wav
        #[arg(value_name = "voice-sample")]
        voice_sample: PathBuf,
        /// What to say
        #[arg(value_name = "text", allow_hyphen_values = true)]
        text: String,
        /// Where to save the speech
        #[arg(value_name = "output.wav")]
        output: PathBuf,
        #[command(flatten)]
        say: SayArgs,
    },
    /// Lists a book's chapters, numbered for `book`
    Chapters {
        /// A DRM-free EPUB
        #[arg(value_name = "book.epub")]
        book: PathBuf,
    },
    /// Narrates a book, or only the chapters given
    ///
    /// Rerunning the same command resumes an interrupted book.
    Book {
        /// A voice sample, such as voices/me.wav
        #[arg(value_name = "voice-sample")]
        voice_sample: PathBuf,
        /// A DRM-free EPUB
        #[arg(value_name = "book.epub")]
        book: PathBuf,
        /// The folder holding this book's narration
        #[arg(value_name = "output-dir")]
        output_dir: PathBuf,
        /// Chapter numbers from `chapters`, such as 2 3 7; all of them if none are given
        #[arg(value_name = "chapter")]
        chapters: Vec<usize>,
        #[command(flatten)]
        speed: Speed,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Records new takes of chunks of a narrated book
    ///
    /// Uses the speed saved in the output directory.
    Retake {
        /// A voice sample, such as voices/me.wav
        #[arg(value_name = "voice-sample")]
        voice_sample: PathBuf,
        /// A DRM-free EPUB
        #[arg(value_name = "book.epub")]
        book: PathBuf,
        /// The folder holding this book's narration
        #[arg(value_name = "output-dir")]
        output_dir: PathBuf,
        /// Words from the book, or a chunk number such as 3.12; after `--` if they start with -
        #[arg(value_name = "target", required = true)]
        targets: Vec<String>,
        /// Refused with the reason, rather than as an unknown option
        #[arg(long = "speed", value_name = "factor", hide = true, value_parser = refuse_retake_speed)]
        refused_speed: Option<String>,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Lists the words most often heard as other words in an output directory
    Mishearings {
        /// A DRM-free EPUB
        #[arg(value_name = "book.epub")]
        book: PathBuf,
        /// The folder holding this book's narration
        #[arg(value_name = "output-dir")]
        output_dir: PathBuf,
        /// Marks the pairs this voice already allows
        #[arg(value_name = "voice-sample")]
        voice_sample: Option<PathBuf>,
    },
    /// Lists the Cast speakers on the network
    Speakers,
    /// Plays files on a Cast speaker
    Play {
        /// Part of the speaker's name, ignoring case
        #[arg(value_name = "speaker")]
        speaker: String,
        /// Audio files, such as a book's M4B or chapter WAVs, played in turn
        #[arg(value_name = "file", required = true)]
        files: Vec<PathBuf>,
    },
    /// Plays a voice sample and its clone saying the same words
    Audition {
        /// A voice sample, such as voices/me.wav
        #[arg(value_name = "voice-sample")]
        voice_sample: PathBuf,
        /// Plays on the Cast speaker with this in its name, ignoring case, rather than here
        #[arg(long, value_name = "name")]
        speaker: Option<String>,
        #[command(flatten)]
        say: SayArgs,
    },
    /// Suggests the best stretches of a recording to use as a voice sample
    Sections {
        /// A recording of the voice, in any format ffmpeg reads
        #[arg(value_name = "recording")]
        recording: PathBuf,
        /// How many stretches to suggest
        #[arg(value_name = "count", default_value_t = sections::DEFAULT_COUNT, value_parser = parse_count)]
        count: usize,
        #[command(flatten)]
        reduction: Reduction,
    },
    /// Makes a voice sample from part of a noisy recording
    Clean {
        /// A recording of the voice, in any format ffmpeg reads
        #[arg(value_name = "recording")]
        recording: PathBuf,
        /// Seconds, such as 20.1, 20.1s or 1500ms, or minutes and seconds, such as 1:05.5
        #[arg(value_name = "start", value_parser = clean::parse_time)]
        start: f64,
        /// As for start
        #[arg(value_name = "end", value_parser = clean::parse_time)]
        end: f64,
        /// Where to save the voice sample, such as voices/me.wav
        #[arg(value_name = "output.wav")]
        output: PathBuf,
        #[command(flatten)]
        reduction: Reduction,
    },
}

#[derive(clap::Args)]
struct Speed {
    /// Speaking speed from 0.5 to 2.0, where 0.8 is slower and 1.25 faster
    #[arg(long = "speed", value_name = "factor", value_parser = parse_speed)]
    factor: Option<f64>,
}

/// Options for the commands that load the model and a voice sample.
#[derive(clap::Args)]
struct ModelArgs {
    #[arg(
        long,
        value_name = "dB",
        env = quality::NOISE_LIMIT_VAR,
        allow_negative_numbers = true,
        value_parser = quality::parse_noise_limit,
        help = format!(
            "The most background a take can have, from -100 to 0 dB [default: {}]; book and retake \
             retry takes above it and save it in the output directory",
            quality::DEFAULT_NOISE_LIMIT_DB
        ),
    )]
    noise_limit: Option<f32>,
    /// The model's precision; the default suits the device
    #[arg(
        long,
        value_name = "precision",
        env = narrator::PRECISION_VAR,
        value_parser = ["f32", "f16", "bf16"],
        ignore_case = true,
    )]
    precision: Option<String>,
}

impl ModelArgs {
    /// Passes the precision to the model, which only reads it from the environment.
    fn use_precision(&self) {
        if let Some(precision) = &self.precision {
            // SAFETY: called from main before the model loads, while no other
            // threads exist that could read the environment.
            unsafe { std::env::set_var(narrator::PRECISION_VAR, precision.to_ascii_lowercase()) };
        }
    }
}

/// Options for `speak` and `audition`, which each say one take.
#[derive(clap::Args)]
struct SayArgs {
    #[command(flatten)]
    speed: Speed,
    /// Picks a repeatable take; without one, each run differs
    #[arg(long, value_name = "n", env = narrator::SEED_VAR)]
    seed: Option<u64>,
    #[command(flatten)]
    model: ModelArgs,
}

impl SayArgs {
    fn settings(&self) -> narrator::SpeakSettings {
        narrator::SpeakSettings { speed: self.speed.factor, seed: self.seed }
    }
}

impl Command {
    /// The model options, for a command that loads the model.
    fn model_args(&self) -> Option<&ModelArgs> {
        match self {
            Command::Speak { say, .. } | Command::Audition { say, .. } => Some(&say.model),
            Command::Book { model, .. } | Command::Retake { model, .. } => Some(model),
            _ => None,
        }
    }

    /// The files the command reads, which must all exist.
    fn input_files(&self) -> Vec<&Path> {
        match self {
            Command::Speak { voice_sample, .. } | Command::Audition { voice_sample, .. } => vec![voice_sample],
            Command::Chapters { book } => vec![book],
            Command::Book { voice_sample, book, .. } | Command::Retake { voice_sample, book, .. } => vec![voice_sample, book],
            Command::Mishearings { book, voice_sample, .. } => {
                std::iter::once(book).chain(voice_sample).map(PathBuf::as_path).collect()
            }
            Command::Play { files, .. } => files.iter().map(PathBuf::as_path).collect(),
            Command::Sections { recording, .. } | Command::Clean { recording, .. } => vec![recording],
            Command::Speakers => vec![],
        }
    }
}

#[derive(clap::Args)]
struct Reduction {
    #[arg(
        long = "reduction",
        value_name = "dB",
        value_parser = clean::parse_reduction,
        help = format!("The most noise to remove, from 1 to 100 dB [default: {}]", clean::DEFAULT_REDUCTION_DB),
    )]
    db: Option<f32>,
}

/// Rust ignores SIGPIPE, so printing to a reader that has gone, such as `head`, panics.
/// Restoring the default makes the program stop quietly like other Unix tools.
#[cfg(unix)]
fn stop_quietly_when_output_closes() {
    // SAFETY: runs first in main, before any other threads exist.
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };
}

#[cfg(not(unix))]
fn stop_quietly_when_output_closes() {}

/// Shows errors readably: in colour only on a terminal, and without a backtrace
/// unless RUST_LIB_BACKTRACE asks for one, since most errors are about the files
/// given rather than the code. Panics still follow RUST_BACKTRACE.
fn install_error_reports() -> Result<()> {
    if std::env::var_os("RUST_LIB_BACKTRACE").is_none() {
        // SAFETY: called from main before any other threads exist.
        unsafe { std::env::set_var("RUST_LIB_BACKTRACE", "0") };
    }
    let theme = if std::io::stderr().is_terminal() { Theme::dark() } else { Theme::new() };
    HookBuilder::default().theme(theme).display_env_section(false).install()
}

fn main() -> Result<()> {
    stop_quietly_when_output_closes();
    install_error_reports()?;
    let command = Cli::parse().command;
    // Before loading anything, which can take a while.
    if let Some(missing) = command.input_files().into_iter().find(|file| !file.is_file()) {
        bail!("no such file: {}", missing.display());
    }
    if let Some(model) = command.model_args() {
        model.use_precision();
    }
    match command {
        Command::Speak { voice_sample, text, output, say } => {
            narrator::speak(&voice_sample, &text, &output, &say.settings(), say.model.noise_limit)
        }
        Command::Chapters { book } => book::list_chapters(&book),
        Command::Book { voice_sample, book, output_dir, chapters, speed, model } => {
            let chapters = (!chapters.is_empty()).then_some(chapters);
            book::narrate(&voice_sample, &book, &output_dir, chapters, speed.factor, model.noise_limit)
        }
        Command::Retake { voice_sample, book, output_dir, targets, model, .. } => {
            let targets: Vec<&str> = targets.iter().map(String::as_str).collect();
            book::retake(&voice_sample, &book, &output_dir, &targets, model.noise_limit)
        }
        Command::Mishearings { book, output_dir, voice_sample } => mishearings::list(&book, &output_dir, voice_sample.as_deref()),
        Command::Speakers => {
            for speaker in cast::discover()? {
                println!("{} ({})", speaker.name, speaker.ip);
            }
            Ok(())
        }
        Command::Play { speaker, files } => cast::play(&cast::find(&speaker)?, files),
        Command::Audition { voice_sample, speaker, say } => {
            audition::audition(&voice_sample, speaker.as_deref(), &say.settings(), say.model.noise_limit)
        }
        Command::Sections { recording, count, reduction } => sections::suggest(&recording, count, reduction.db),
        Command::Clean { recording, start, end, output, reduction } => {
            let reduction = reduction.db.unwrap_or(clean::DEFAULT_REDUCTION_DB);
            clean::clean(&recording, start, end, &output, reduction)
        }
    }
}

fn parse_speed(value: &str) -> Result<f64, String> {
    let speed = value.parse::<f64>().ok().filter(|speed| SPEED_RANGE.contains(speed));
    speed.ok_or_else(|| format!("must be from {:.1} to {:.1}", SPEED_RANGE.start(), SPEED_RANGE.end()))
}

fn refuse_retake_speed(_: &str) -> Result<String, String> {
    Err("retake uses the speed saved in the output directory".to_string())
}

fn parse_count(value: &str) -> Result<usize, String> {
    let count = value.parse::<usize>().ok().filter(|count| *count > 0);
    count.ok_or_else(|| "must be a whole number from 1".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse(words: &[&str]) -> Result<Command, clap::Error> {
        Cli::try_parse_from(std::iter::once("audio-berry").chain(words.iter().copied())).map(|cli| cli.command)
    }

    /// Each command's usage line, as `audio-berry <command> --help` shows it.
    fn usage_lines() -> Vec<String> {
        let mut cli = Cli::command();
        cli.build();
        let commands = cli.get_subcommands_mut().filter(|command| command.get_name() != "help");
        let usage = commands.map(|command| command.render_usage().to_string());
        usage.map(|line| line.trim_start_matches("Usage: ").to_string()).collect()
    }

    #[test]
    fn the_cli_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn the_readme_lists_the_commands_as_usage_does() {
        let readme = include_str!("../README.md");
        let start = readme.find("## Commands\n\n```text\n").expect("README has a Commands list") + "## Commands\n\n```text\n".len();
        let listed = &readme[start..start + readme[start..].find("```").unwrap()];
        assert_eq!(listed.lines().collect::<Vec<_>>(), usage_lines());
    }

    #[test]
    fn speed_is_taken_from_any_position() {
        for words in [["book", "--speed", "0.9", "v", "b", "o", "3"], ["book", "v", "b", "o", "3", "--speed", "0.9"]] {
            let Ok(Command::Book { chapters, speed, .. }) = parse(&words) else { panic!("{words:?}") };
            assert_eq!(chapters, [3]);
            assert_eq!(speed.factor, Some(0.9));
        }
    }

    #[test]
    fn speed_is_for_commands_that_render_at_a_chosen_speed() {
        assert!(parse(&["speak", "v", "t", "o", "--speed", "1.2"]).is_ok());
        assert!(parse(&["audition", "v", "--speed", "1.2"]).is_ok());
        for words in [&["retake", "v", "b", "o", "3.12", "--speed", "1"][..], &["clean", "r", "1", "2", "o", "--speed", "1"]] {
            assert!(parse(words).is_err(), "{words:?}");
        }
    }

    #[test]
    fn no_flag_means_no_value() {
        let Ok(Command::Speak { say, .. }) = parse(&["speak", "v", "t", "o"]) else { panic!() };
        assert_eq!(say.speed.factor, None);
        let Ok(Command::Clean { reduction, .. }) = parse(&["clean", "r", "1", "2", "o"]) else { panic!() };
        assert_eq!(reduction.db, None);
    }

    #[test]
    fn a_missing_or_silly_speed_is_an_error() {
        for bad in ["0", "-1", "0.4", "2.1", "fast", "NaN"] {
            assert!(parse(&["book", "v", "b", "o", "--speed", bad]).is_err(), "{bad}");
        }
        assert!(parse(&["book", "v", "b", "o", "--speed"]).is_err());
    }

    #[test]
    fn reduction_is_for_commands_that_clean() {
        let Ok(Command::Clean { reduction, .. }) = parse(&["clean", "--reduction", "20", "r", "1", "2", "o"]) else { panic!() };
        assert_eq!(reduction.db, Some(20.0));
        let Ok(Command::Sections { count, reduction, .. }) = parse(&["sections", "r", "--reduction", "20"]) else { panic!() };
        assert_eq!((count, reduction.db), (sections::DEFAULT_COUNT, Some(20.0)));
        for command in ["speak", "book", "audition", "retake"] {
            assert!(parse(&[command, "v", "b", "o", "3", "--reduction", "20"]).is_err(), "{command}");
        }
    }

    #[test]
    fn a_missing_or_silly_reduction_is_an_error() {
        for bad in ["0", "-15", "101", "lots"] {
            assert!(parse(&["clean", "r", "1", "2", "o", "--reduction", bad]).is_err(), "{bad}");
        }
        assert!(parse(&["clean", "r", "1", "2", "o", "--reduction"]).is_err());
    }

    #[test]
    fn the_counts_and_times_are_checked() {
        assert!(parse(&["sections", "r", "0"]).is_err());
        assert!(parse(&["clean", "r", "soon", "2", "o"]).is_err());
        assert!(parse(&["book", "v", "b", "o", "three"]).is_err());
    }

    #[test]
    fn a_noise_limit_is_negative_and_for_commands_that_load_a_voice() {
        let Ok(Command::Book { model, .. }) = parse(&["book", "v", "b", "o", "--noise-limit", "-50"]) else { panic!() };
        assert_eq!(model.noise_limit, Some(-50.0));
        let Ok(Command::Retake { model, .. }) = parse(&["retake", "v", "b", "o", "3.12", "--noise-limit", "-48"]) else { panic!() };
        assert_eq!(model.noise_limit, Some(-48.0));
        for bad in ["5", "-101", "lots"] {
            assert!(parse(&["book", "v", "b", "o", "--noise-limit", bad]).is_err(), "{bad}");
        }
        assert!(parse(&["clean", "r", "1", "2", "o", "--noise-limit", "-50"]).is_err());
    }

    #[test]
    fn precision_is_one_the_model_knows() {
        let Ok(Command::Retake { model, .. }) = parse(&["retake", "v", "b", "o", "3", "--precision", "F16"]) else { panic!() };
        assert_eq!(model.precision.map(|p| p.to_ascii_lowercase()).as_deref(), Some("f16"));
        assert!(parse(&["book", "v", "b", "o", "--precision", "f8"]).is_err());
    }

    #[test]
    fn a_seed_is_for_commands_that_say_one_take() {
        let Ok(Command::Audition { say, .. }) = parse(&["audition", "v", "--seed", "2"]) else { panic!() };
        assert_eq!(say.seed, Some(2));
        assert!(parse(&["speak", "v", "t", "o", "--seed", "two"]).is_err());
        assert!(parse(&["book", "v", "b", "o", "--seed", "2"]).is_err());
    }

    #[test]
    fn text_and_targets_can_start_with_a_hyphen() {
        let Ok(Command::Speak { text, say, .. }) = parse(&["speak", "v", "-hello there", "o", "--speed", "0.9"]) else { panic!() };
        assert_eq!((text.as_str(), say.speed.factor), ("-hello there", Some(0.9)));
        let Ok(Command::Speak { text, say, .. }) = parse(&["speak", "v", "--speed", "0.9", "t", "o"]) else { panic!() };
        assert_eq!((text.as_str(), say.speed.factor), ("t", Some(0.9)));
        let words = ["retake", "v", "b", "o", "3.12", "--noise-limit", "-50", "--", "-said she"];
        let Ok(Command::Retake { targets, model, .. }) = parse(&words) else { panic!() };
        assert_eq!((targets, model.noise_limit), (vec!["3.12".to_string(), "-said she".to_string()], Some(-50.0)));
    }

    #[test]
    fn retake_says_why_it_refuses_a_speed() {
        let error = parse(&["retake", "v", "b", "o", "3.12", "--speed", "1"]).err().unwrap().to_string();
        assert!(error.contains("retake uses the speed saved in the output directory"), "{error}");
    }

    #[test]
    fn commands_that_need_a_list_refuse_an_empty_one() {
        assert!(parse(&["retake", "v", "b", "o"]).is_err());
        assert!(parse(&["play", "kitchen"]).is_err());
    }

    #[test]
    fn audition_names_a_speaker_with_an_option() {
        let Ok(Command::Audition { speaker, .. }) = parse(&["audition", "v", "--speaker", "kitchen"]) else { panic!() };
        assert_eq!(speaker.as_deref(), Some("kitchen"));
        let Ok(Command::Audition { speaker, .. }) = parse(&["audition", "v"]) else { panic!() };
        assert_eq!(speaker, None);
        assert!(parse(&["audition", "v", "kitchen"]).is_err());
    }

    #[test]
    fn the_files_each_command_reads_are_checked() {
        let inputs = |words: &[&str]| -> Vec<PathBuf> {
            parse(words).unwrap().input_files().into_iter().map(Path::to_path_buf).collect()
        };
        let paths = |names: &[&str]| -> Vec<PathBuf> { names.iter().map(PathBuf::from).collect() };
        assert_eq!(inputs(&["speak", "v.wav", "hi", "o.wav"]), paths(&["v.wav"]));
        assert_eq!(inputs(&["audition", "v.wav", "--speaker", "kitchen"]), paths(&["v.wav"]));
        assert_eq!(inputs(&["chapters", "b.epub"]), paths(&["b.epub"]));
        assert_eq!(inputs(&["book", "v.wav", "b.epub", "out"]), paths(&["v.wav", "b.epub"]));
        assert_eq!(inputs(&["retake", "v.wav", "b.epub", "out", "3.1"]), paths(&["v.wav", "b.epub"]));
        assert_eq!(inputs(&["mishearings", "b.epub", "out"]), paths(&["b.epub"]));
        assert_eq!(inputs(&["mishearings", "b.epub", "out", "v.wav"]), paths(&["b.epub", "v.wav"]));
        assert_eq!(inputs(&["play", "kitchen", "a.wav", "b.wav"]), paths(&["a.wav", "b.wav"]));
        assert_eq!(inputs(&["sections", "r.mp3"]), paths(&["r.mp3"]));
        assert_eq!(inputs(&["clean", "r.mp3", "1", "2", "o.wav"]), paths(&["r.mp3"]));
        assert!(inputs(&["speakers"]).is_empty());
    }
}
