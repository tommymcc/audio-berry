//! Reads audio files of any format.

use std::error::Error;
use std::path::Path;
use std::process::Command;

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
