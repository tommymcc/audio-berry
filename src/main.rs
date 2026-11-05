mod book;
mod cast;
mod epub;
mod quality;
mod store;
mod text;
mod transcribe;

use transcribe::{Transcriber, VOICE_TRANSCRIPTION_MODEL};
use any_tts::models::omnivoice::{preferred_runtime_choice, trim_long_audio};
use any_tts::{ModelType, ReferenceAudio, SynthesisRequest, TtsConfig, TtsModel, load_model};
use any_tts::AudioSamples;
use std::error::Error;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const SAMPLE_RATE: u32 = 24_000;
/// OmniVoice recommends 3-10 seconds of reference and warns above 20, so longer
/// samples are cut to 15 at a pause.
const MAX_VOICE_SECONDS: f32 = 15.0;
const MIN_VOICE_SECONDS: f32 = 3.0;
const TRIM_VOICE_ABOVE_SECONDS: f32 = 20.0;

/// Decodes any audio file ffmpeg understands into mono f32 PCM.
pub fn decode_audio(path: &Path, sample_rate: u32) -> Result<Vec<f32>, Box<dyn Error>> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-ac", "1", "-ar", &sample_rate.to_string(), "-f", "f32le", "-"])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Loads a voice sample and what's said in it. The text lives in a `.txt` file
/// beside the sample, written by transcribing it the first time, and can be
/// corrected by hand.
pub fn load_reference(path: &Path) -> Result<ReferenceAudio, Box<dyn Error>> {
    let samples = trim_long_audio(
        &decode_audio(path, SAMPLE_RATE)?,
        SAMPLE_RATE,
        MAX_VOICE_SECONDS,
        MIN_VOICE_SECONDS,
        TRIM_VOICE_ABOVE_SECONDS,
    );
    let reference = ReferenceAudio::new(samples, SAMPLE_RATE);
    println!("Voice sample: using {:.1}s", reference.duration_secs());

    let transcript_path = path.with_extension("txt");
    let existing = match fs::read_to_string(&transcript_path) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => String::new(),
        Err(error) => return Err(error.into()),
    };
    let existing = existing.trim();
    if !existing.is_empty() {
        return Ok(reference.with_transcript(existing));
    }

    let clip = transcript_path.with_extension("transcribing.wav");
    AudioSamples::new(reference.samples.clone(), SAMPLE_RATE).save_wav(&clip)?;
    let transcript = Transcriber::start(VOICE_TRANSCRIPTION_MODEL)
        .and_then(|mut transcriber| transcriber.transcribe(&clip));
    fs::remove_file(&clip)?;
    let transcript = transcript?;
    if transcript.is_empty() {
        return Err(format!("heard no speech in {}", path.display()).into());
    }
    fs::write(&transcript_path, format!("{transcript}\n"))?;
    println!("Heard in the voice sample: {transcript}");
    println!("Saved to {}. Correct any mistakes there.", transcript_path.display());
    Ok(reference.with_transcript(transcript))
}

/// Beyond this, the length checks and Whisper haven't been tried.
const SPEED_RANGE: std::ops::RangeInclusive<f64> = 0.5..=2.0;

const USAGE: &str = "usage:
  audio-berry speak <voice-sample> <text> <output.wav> [--speed <factor>]
  audio-berry chapters <book.epub>
  audio-berry book <voice-sample> <book.epub> <output-dir> [chapter numbers, e.g. 2 3 7] [--speed <factor>]
  audio-berry retake <voice-sample> <book.epub> <output-dir> <words from the book, or a chunk number like 3.12>...
  audio-berry speakers
  audio-berry play <speaker> <file>...";

pub fn load_narrator() -> Result<Box<dyn TtsModel>, Box<dyn Error>> {
    let started = Instant::now();
    let model = load_model(TtsConfig::new(ModelType::OmniVoice).with_preferred_runtime())?;
    println!("Model loaded in {:.0}s on {}", started.elapsed().as_secs_f32(), preferred_runtime_choice().device.label());
    Ok(model)
}

fn main() -> Result<(), Box<dyn Error>> {
    let (args, speed) = take_speed(std::env::args().skip(1).collect())?;
    if speed.is_some() && !matches!(args.first().map(String::as_str), Some("speak" | "book")) {
        return Err("--speed only works with speak and book; retake uses the speed saved in the output directory".into());
    }
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["speak", voice, text, output] => speak(voice, text, output, speed),
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
        _ => Err(USAGE.into()),
    }
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

fn speak(voice: &str, text: &str, output: &str, speed: Option<f64>) -> Result<(), Box<dyn Error>> {
    let reference = load_reference(Path::new(voice))?;

    let model = load_narrator()?;

    let started = Instant::now();
    let mut request = SynthesisRequest::new(text).with_reference_audio(reference);
    if let Some(speed) = speed {
        request = request.with_speed(speed);
    }
    let audio = model.synthesize(&request)?;
    let elapsed = started.elapsed().as_secs_f32();

    audio.save_wav(output)?;
    println!(
        "Wrote {output}: {:.1}s of audio in {elapsed:.0}s ({:.2}x realtime)",
        audio.duration_secs(),
        audio.duration_secs() / elapsed
    );
    Ok(())
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
