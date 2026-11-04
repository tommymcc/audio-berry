mod book;
mod cast;
mod epub;
mod quality;
mod text;
mod transcribe;

use any_tts::{ModelType, ReferenceAudio, SynthesisRequest, TtsConfig, TtsModel, load_model};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const SAMPLE_RATE: u32 = 24_000;
/// Guidance strength. 2.0 followed the text more closely than 1.3 in testing.
pub const CFG_SCALE: f64 = 2.0;

/// Decodes any audio file ffmpeg understands into mono f32 PCM.
pub fn load_reference(path: &Path) -> Result<ReferenceAudio, Box<dyn Error>> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-ac", "1", "-ar", &SAMPLE_RATE.to_string(), "-f", "f32le", "-"])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let samples = output
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(ReferenceAudio::new(samples, SAMPLE_RATE))
}

const USAGE: &str = "usage:
  audio-berry speak <voice-sample> <text> <output.wav>
  audio-berry chapters <book.epub>
  audio-berry book <voice-sample> <book.epub> <output-dir> [chapter numbers, e.g. 2 3 7]
  audio-berry retake <voice-sample> <book.epub> <output-dir> <words from the book, or a chunk number>...
  audio-berry speakers
  audio-berry play <speaker> <file>...";

pub fn load_narrator() -> Result<Box<dyn TtsModel>, Box<dyn Error>> {
    let started = Instant::now();
    let model = load_model(TtsConfig::new(ModelType::VibeVoice).with_preferred_runtime())?;
    println!("Model loaded in {:.0}s", started.elapsed().as_secs_f32());
    Ok(model)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["speak", voice, text, output] => speak(voice, text, output),
        ["chapters", book] => book::list_chapters(Path::new(book)),
        ["book", voice, book, output, chapters @ ..] => {
            let chapters = if chapters.is_empty() {
                None
            } else {
                let numbers = chapters.iter().map(|number| number.parse::<usize>());
                Some(numbers.collect::<Result<Vec<_>, _>>().map_err(|_| USAGE)?)
            };
            book::narrate(Path::new(voice), Path::new(book), Path::new(output), chapters)
        }
        ["retake", voice, book, output, targets @ ..] if !targets.is_empty() => {
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

fn speak(voice: &str, text: &str, output: &str) -> Result<(), Box<dyn Error>> {
    let reference = load_reference(Path::new(voice))?;
    println!("Reference: {:.1}s", reference.duration_secs());

    let model = load_narrator()?;

    let started = Instant::now();
    let request = SynthesisRequest::new(text)
        .with_reference_audio(reference)
        .with_cfg_scale(CFG_SCALE);
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
