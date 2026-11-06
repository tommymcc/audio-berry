//! Reads audio files of any format.

use any_tts::AudioSamples;
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// Decodes any audio file ffmpeg understands into mono f32 PCM.
pub fn decode_audio(path: &Path, sample_rate: u32) -> Result<Vec<f32>, Box<dyn Error>> {
    decode(path, sample_rate, None)
}

/// Decodes at most the first `seconds` of an audio file, so a long one can be
/// refused without holding all of it.
pub fn decode_audio_up_to(path: &Path, sample_rate: u32, seconds: f64) -> Result<Vec<f32>, Box<dyn Error>> {
    decode(path, sample_rate, Some(seconds))
}

fn decode(path: &Path, sample_rate: u32, seconds: Option<f64>) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg.args(["-v", "error"]);
    if let Some(seconds) = seconds {
        ffmpeg.args(["-t", &seconds.to_string()]);
    }
    ffmpeg.arg("-i").arg(path);
    run(ffmpeg, None, sample_rate, &format!("ffmpeg failed on {}", path.display()))
}

/// Converts audio to mono f32 PCM at `sample_rate`, exactly as `decode_audio`
/// would read it back from a saved WAV file, without saving one.
pub fn resample_as_saved(audio: &AudioSamples, sample_rate: u32) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg.args(["-v", "error", "-f", "wav", "-i", "pipe:0"]);
    run(ffmpeg, Some(audio.get_wav()), sample_rate, "ffmpeg failed")
}

/// Runs ffmpeg, given its input, feeding it `stdin` if any, and reads the mono
/// f32 PCM it writes. A failure is reported after `failed`.
fn run(mut ffmpeg: Command, stdin: Option<Vec<u8>>, sample_rate: u32, failed: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    ffmpeg
        .args(["-ac", "1", "-ar", &sample_rate.to_string(), "-f", "f32le", "-"])
        .stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = ffmpeg.spawn()?;
    // Written from another thread, so ffmpeg's output can't fill up and stall it.
    let writer = match (stdin, child.stdin.take()) {
        (Some(bytes), Some(mut pipe)) => Some(std::thread::spawn(move || pipe.write_all(&bytes))),
        _ => None,
    };
    let output = child.wait_with_output()?;
    let written = match writer {
        Some(writer) => writer.join().map_err(|_| "couldn't feed ffmpeg")?,
        None => Ok(()),
    };
    // ffmpeg stopping early closes the pipe, and its own error says why.
    if !output.status.success() {
        return Err(format!("{failed}: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    written?;
    Ok(output
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_resampled_in_memory_matches_the_saved_file_read_back() {
        // Transcripts are kept, so a take must be heard the same either way.
        let samples = (0..24_000).map(|i| (i as f32 * 0.031).sin() * 0.4 + (i as f32 * 0.17).cos() * 0.1).collect();
        let audio = AudioSamples::new(samples, 24_000);
        let path = std::env::temp_dir().join(format!("audio-berry-resample-{}.wav", std::process::id()));
        audio.save_wav(&path).unwrap();
        let from_file = decode_audio(&path, 16_000);
        std::fs::remove_file(&path).unwrap();
        let from_file = from_file.unwrap();
        assert_eq!(from_file.len(), 16_000);
        assert_eq!(resample_as_saved(&audio, 16_000).unwrap(), from_file);
    }
}
