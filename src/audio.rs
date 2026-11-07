//! Reads audio files of any format, and plays them here.

use any_tts::AudioSamples;
use color_eyre::eyre::{Report, Result, WrapErr, eyre};
use color_eyre::{Section, SectionExt};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// Decodes any audio file ffmpeg understands into mono f32 PCM.
pub fn decode_audio(path: &Path, sample_rate: u32) -> Result<Vec<f32>> {
    decode(path, sample_rate, None)
}

/// Decodes at most the first `seconds` of an audio file, so a long one can be
/// refused without holding all of it.
pub fn decode_audio_up_to(path: &Path, sample_rate: u32, seconds: f64) -> Result<Vec<f32>> {
    decode(path, sample_rate, Some(seconds))
}

fn decode(path: &Path, sample_rate: u32, seconds: Option<f64>) -> Result<Vec<f32>> {
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg.args(["-v", "error"]);
    if let Some(seconds) = seconds {
        ffmpeg.args(["-t", &seconds.to_string()]);
    }
    ffmpeg.arg("-i").arg(path);
    run(ffmpeg, None, sample_rate, &format!("read {}", path.display()))
}

/// Converts audio to mono f32 PCM at `sample_rate`, exactly as `decode_audio`
/// would read it back from a saved WAV file, without saving one.
pub fn resample_as_saved(audio: &AudioSamples, sample_rate: u32) -> Result<Vec<f32>> {
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg.args(["-v", "error", "-f", "wav", "-i", "pipe:0"]);
    run(ffmpeg, Some(audio.get_wav()), sample_rate, "resample audio")
}

/// Runs ffmpeg, given its input, feeding it `stdin` if any, and reads the mono
/// f32 PCM it writes. A failure says ffmpeg couldn't do `doing`.
fn run(mut ffmpeg: Command, stdin: Option<Vec<u8>>, sample_rate: u32, doing: &str) -> Result<Vec<f32>> {
    ffmpeg
        .args(["-ac", "1", "-ar", &sample_rate.to_string(), "-f", "f32le", "-"])
        .stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = ffmpeg.spawn().wrap_err(COULDNT_RUN)?;
    // Written from another thread, so ffmpeg's output can't fill up and stall it.
    let writer = match (stdin, child.stdin.take()) {
        (Some(bytes), Some(mut pipe)) => Some(std::thread::spawn(move || pipe.write_all(&bytes))),
        _ => None,
    };
    let output = child.wait_with_output()?;
    let written = match writer {
        Some(writer) => writer.join().map_err(|_| eyre!("couldn't feed ffmpeg"))?,
        None => Ok(()),
    };
    // ffmpeg stopping early closes the pipe, and its own error says why.
    if !output.status.success() {
        return Err(ffmpeg_failed(doing, &output.stderr));
    }
    written?;
    Ok(output
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Why ffmpeg failed to start, which is usually that it isn't installed.
pub const COULDNT_RUN: &str = "couldn't run ffmpeg; is it installed?";

/// An error saying ffmpeg couldn't do what it was asked, such as "read
/// voices/me.wav", with what it said about it below, without the internal
/// addresses it starts its lines with.
pub fn ffmpeg_failed(doing: &str, stderr: &[u8]) -> Report {
    tool_failed("ffmpeg", doing, stderr)
}

/// As `ffmpeg_failed`, for `ffmpeg` or `ffplay`.
fn tool_failed(tool: &str, doing: &str, stderr: &[u8]) -> Report {
    let failed = eyre!("{tool} couldn't {doing}");
    match ffmpeg_said(stderr) {
        Some(said) => failed.section(said.header(format!("{tool} said:"))),
        None => failed,
    }
}

/// Why ffplay failed to start. It comes with ffmpeg, but some packages leave it out.
const COULDNT_RUN_FFPLAY: &str = "couldn't run ffplay, which plays audio here; is it installed with ffmpeg?";

/// Plays an audio file on this computer, waiting until it ends.
pub fn play_here(path: &Path) -> Result<()> {
    let output = Command::new("ffplay")
        .args(["-nodisp", "-autoexit", "-loglevel", "error"])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .wrap_err(COULDNT_RUN_FFPLAY)?;
    // ffplay exits successfully even when it can't read the file, but at this
    // log level it only writes errors.
    if !output.status.success() || !output.stderr.trim_ascii().is_empty() {
        return Err(tool_failed("ffplay", &format!("play {}", path.display()), &output.stderr));
    }
    Ok(())
}

/// ffmpeg's error output without its addresses or blank lines, if it said anything.
fn ffmpeg_said(stderr: &[u8]) -> Option<String> {
    let stderr = String::from_utf8_lossy(stderr);
    let said: Vec<&str> = stderr.lines().map(without_address).filter(|line| !line.is_empty()).collect();
    (!said.is_empty()).then(|| said.join("\n"))
}

/// A line of ffmpeg's output without a leading `[in#0 @ 0x5bfd60e7dc40] `.
fn without_address(line: &str) -> &str {
    let line = line.trim();
    match line.strip_prefix('[').and_then(|rest| rest.split_once("] ")) {
        Some((inside, rest)) if inside.contains(" @ 0x") => rest.trim(),
        _ => line,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_dir::TempDir;

    #[test]
    fn ffmpeg_errors_say_what_failed_then_what_ffmpeg_said_without_addresses() {
        let stderr = b"[in#0 @ 0x5bfd60e7dc40] Error opening input: No such file or directory\n\
                       Error opening input file voices/me.wav.\n\n";
        let report = ffmpeg_failed("read voices/me.wav", stderr);
        assert_eq!(report.to_string(), "ffmpeg couldn't read voices/me.wav");
        assert_eq!(
            ffmpeg_said(stderr).unwrap(),
            "Error opening input: No such file or directory\nError opening input file voices/me.wav."
        );
        assert_eq!(ffmpeg_said(b"\n  \n"), None);
    }

    #[test]
    fn a_file_that_isnt_audio_fails_to_play_here_and_says_so() {
        // Fails on reading the file, before any sound could be made.
        let dir = TempDir::new("play-here");
        let path = dir.join("not-audio.wav");
        std::fs::write(&path, "not audio").unwrap();
        let error = play_here(&path).unwrap_err().to_string();
        assert!(error.contains("ffplay"), "{error}");
    }

    #[test]
    fn brackets_without_an_address_are_kept() {
        assert_eq!(without_address("[mp3 @ 0x1f] bad frame"), "bad frame");
        assert_eq!(without_address("[sic] as written"), "[sic] as written");
        assert_eq!(without_address("  plain  "), "plain");
    }

    #[test]
    fn audio_resampled_in_memory_matches_the_saved_file_read_back() {
        // Transcripts are kept, so a take must be heard the same either way.
        let samples = (0..24_000).map(|i| (i as f32 * 0.031).sin() * 0.4 + (i as f32 * 0.17).cos() * 0.1).collect();
        let audio = AudioSamples::new(samples, 24_000);
        let dir = TempDir::new("resample");
        let path = dir.join("take.wav");
        audio.save_wav(&path).unwrap();
        let from_file = decode_audio(&path, 16_000).unwrap();
        assert_eq!(from_file.len(), 16_000);
        assert_eq!(resample_as_saved(&audio, 16_000).unwrap(), from_file);
    }
}
