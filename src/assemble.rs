//! Joins an output directory's complete chapters into a WAV each and a
//! chaptered M4B, tagged with the book's title, author and cover.

use crate::audio::ffmpeg_failed;
use crate::epub::{Book, image_type};
use crate::quality;
use crate::store::NarrationStore;
use crate::voice::SAMPLE_RATE;
use any_tts::AudioSamples;
use color_eyre::eyre::{Result, eyre};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SPEECH_TARGET_DB: f32 = -21.0;
const CHUNK_GAP_SECONDS: f32 = 0.35;
const CHAPTER_GAP_SECONDS: f32 = 1.2;

fn safe_name(title: &str) -> String {
    let name: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    name.trim_matches('_').chars().take(60).collect()
}

fn metadata_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        if "=;#\\\n".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Writes the book's cover beside the M4B, where `play` finds it for the chapter
/// WAVs, which can't hold a picture.
fn save_cover(book: &Book, out: &Path) -> Result<Option<PathBuf>> {
    let Some(cover) = &book.cover else {
        return Ok(None);
    };
    let (_, extension) = image_type(cover).ok_or_else(|| eyre!("the cover is not a JPEG or PNG"))?;
    let path = out.join(format!("cover.{extension}"));
    fs::write(&path, cover)?;
    Ok(Some(path))
}

/// Copies a chapter's WAV to `tagged` with its title, the book and the author,
/// which `play` sends to the speaker's display.
fn tag_chapter(book: &Book, title: &str, untagged: &Path, tagged: &Path) -> Result<()> {
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg.args(["-v", "error", "-y", "-i"]).arg(untagged).args(["-c", "copy"]);
    ffmpeg.arg("-metadata").arg(format!("title={title}"));
    ffmpeg.arg("-metadata").arg(format!("album={}", book.title));
    if let Some(author) = &book.author {
        ffmpeg.arg("-metadata").arg(format!("artist={author}"));
    }
    let output = ffmpeg.arg(tagged).output()?;
    if !output.status.success() {
        return Err(ffmpeg_failed(&format!("ffmpeg couldn't tag {}", tagged.display()), &output.stderr));
    }
    fs::remove_file(untagged)?;
    Ok(())
}

/// Joins every complete chapter into one WAV each and a chaptered M4B.
pub fn assemble(book: &Book, store: &NarrationStore, out: &Path) -> Result<PathBuf> {
    let chapter_dir = out.join("chapters");
    fs::create_dir_all(&chapter_dir)?;
    let mut sample_rate = SAMPLE_RATE;
    let mut all = Vec::new();
    let mut metadata = format!(";FFMETADATA1\ntitle={}\n", metadata_escape(&book.title));
    if let Some(author) = &book.author {
        writeln!(metadata, "artist={}", metadata_escape(author))?;
    }

    let chapters = store.complete_chapters();
    for (position, &chapter) in chapters.iter().enumerate() {
        let mut samples = Vec::new();
        for item in store.chunks().iter().filter(|item| item.chapter == chapter) {
            let audio = AudioSamples::from_wav_file(store.audio_path(item))?;
            sample_rate = audio.sample_rate;
            let trimmed = quality::trim_silence(&audio.samples, sample_rate);
            let gain = quality::gain_for(&quality::measure(trimmed, sample_rate), SPEECH_TARGET_DB);
            if !samples.is_empty() {
                samples.resize(samples.len() + (CHUNK_GAP_SECONDS * sample_rate as f32) as usize, 0.0);
            }
            samples.extend(trimmed.iter().map(|sample| sample * gain));
        }
        let title = &book.chapters[chapter].title;
        let name = format!("{:02}-{}.wav", chapter + 1, safe_name(title));
        let untagged = chapter_dir.join("untagged.wav");
        AudioSamples::new(samples.clone(), sample_rate).save_wav(&untagged)?;
        tag_chapter(book, title, &untagged, &chapter_dir.join(name))?;

        let start = all.len() as u64 * 1000 / u64::from(sample_rate);
        all.extend_from_slice(&samples);
        let end = all.len() as u64 * 1000 / u64::from(sample_rate);
        write!(
            metadata,
            "[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle={}\n",
            metadata_escape(title)
        )?;
        if position + 1 < chapters.len() {
            all.resize(all.len() + (CHAPTER_GAP_SECONDS * sample_rate as f32) as usize, 0.0);
        }
    }

    let joined = out.join("joined.wav");
    AudioSamples::new(all, sample_rate).save_wav(&joined)?;
    let metadata_path = out.join("chapters.txt");
    fs::write(&metadata_path, metadata)?;
    let m4b = out.join(format!("{}.m4b", safe_name(&book.title)));
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg.args(["-v", "error", "-y", "-i"]).arg(&joined).arg("-i").arg(&metadata_path);
    let cover = save_cover(book, out)?;
    if let Some(cover) = &cover {
        ffmpeg.arg("-i").arg(cover);
    }
    ffmpeg.args(["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]);
    if cover.is_some() {
        ffmpeg.args(["-map", "2:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]);
    }
    ffmpeg.args(["-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", "-f", "mp4"]);
    let output = ffmpeg.arg(&m4b).output()?;
    if !output.status.success() {
        return Err(ffmpeg_failed(&format!("ffmpeg couldn't make {}", m4b.display()), &output.stderr));
    }
    fs::remove_file(&joined)?;
    Ok(m4b)
}
