//! Says a voice sample's own words in its cloned voice, so the clone and the
//! original can be compared by ear, saying the same thing.

use crate::cast;
use crate::narrator::Narrator;
use crate::voice::Voice;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// Where auditions are saved, so they can be played again with `play`.
const AUDITION_DIR: &str = "output/audition";

/// The pair of files an audition writes for one voice sample.
#[derive(Debug, PartialEq)]
struct Pair {
    /// The part of the sample the model hears, which its transcript describes.
    original: PathBuf,
    /// The transcript said in the cloned voice.
    cloned: PathBuf,
}

/// Names the pair after the sample, so `voices/me.wav` gives `me-original.wav`
/// and `me-clone.wav`. Speakers with a screen show these names.
fn pair_for(voice: &Path, dir: &Path) -> Result<Pair, Box<dyn Error>> {
    let stem = voice
        .file_stem()
        .ok_or_else(|| format!("no file name in {}", voice.display()))?
        .to_string_lossy();
    Ok(Pair {
        original: dir.join(format!("{stem}-original.wav")),
        cloned: dir.join(format!("{stem}-clone.wav")),
    })
}

/// Saves the sample as the model hears it and its transcript in the cloned voice,
/// then plays the two in turn on `speaker`, if one is given.
pub fn audition(voice_sample: &Path, speaker: Option<&str>, speed: Option<f64>) -> Result<(), Box<dyn Error>> {
    // Finding the speaker first means a wrong name fails before the slow part.
    let speaker = speaker.map(cast::find).transpose()?;
    let voice = Voice::load(voice_sample)?;
    let pair = pair_for(voice_sample, Path::new(AUDITION_DIR))?;
    fs::create_dir_all(AUDITION_DIR)?;

    // What plays as the original is the part of the sample its transcript covers.
    voice.save_heard(&pair.original)?;
    println!("Wrote {}: the sample as the model hears it", pair.original.display());
    println!("Saying: {}", voice.transcript());
    let narrator = Narrator::load()?;
    narrator.say_to_file(&voice, voice.transcript(), &pair.cloned, speed.unwrap_or(1.0))?;

    match speaker {
        Some(speaker) => cast::play(&speaker, vec![pair.original, pair.cloned]),
        None => {
            println!(
                "To hear them: audio-berry play <speaker> {} {}",
                pair.original.display(),
                pair.cloned.display()
            );
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pair_is_named_after_the_sample() {
        let pair = pair_for(Path::new("voices/me.wav"), Path::new("output/audition")).unwrap();
        assert_eq!(
            pair,
            Pair {
                original: PathBuf::from("output/audition/me-original.wav"),
                cloned: PathBuf::from("output/audition/me-clone.wav"),
            }
        );
    }

    #[test]
    fn a_sample_in_any_format_gives_wav_files() {
        let pair = pair_for(Path::new("/home/x/dad.full.mp3"), Path::new("out")).unwrap();
        assert_eq!(pair.original, PathBuf::from("out/dad.full-original.wav"));
        assert_eq!(pair.cloned, PathBuf::from("out/dad.full-clone.wav"));
    }

    #[test]
    fn a_sample_needs_a_file_name() {
        assert!(pair_for(Path::new("/"), Path::new("out")).is_err());
    }
}
