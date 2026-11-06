//! Narrates a whole book: generates checked chunks, then assembles the audio.

use crate::accuracy;
use crate::checks::{Standard, Verdict};
use crate::epub::{Book, image_type, read_book};
use crate::narrator::Narrator;
use crate::quality;
use crate::store::{Chunk, NarrationStore, RenderSettings, fnv};
use crate::text::{MAX_CHUNK_CHARS, chunk};
use crate::transcribe::{self, Transcriber};
use crate::voice::{SAMPLE_RATE, Voice};
use any_tts::AudioSamples;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// Takes tried per chunk by `book`.
pub const MAX_TAKES: u64 = 5;
/// New takes tried per chunk by `retake`.
const MAX_RETAKES: u64 = 6;
const SPEECH_TARGET_DB: f32 = -21.0;
const CHUNK_GAP_SECONDS: f32 = 0.35;
const CHAPTER_GAP_SECONDS: f32 = 1.2;

fn open_store(book: &Book, voice: &Voice, out: &Path, speed: Option<f64>) -> Result<NarrationStore, Box<dyn Error>> {
    let chapters: Vec<Vec<String>> = book
        .chapters
        .iter()
        .map(|chapter| chunk(&chapter.paragraphs, MAX_CHUNK_CHARS))
        .collect();
    NarrationStore::open(
        out,
        &book.title,
        &chapters,
        voice.hash(),
        &RenderSettings::from_env(speed),
        quality::noise_limit_from_env(),
    )
}

/// Prints each chapter's number, title and size.
pub fn list_chapters(path: &Path) -> Result<(), Box<dyn Error>> {
    let book = read_book(path)?;
    println!("{} by {}", book.title, book.author.as_deref().unwrap_or("unknown"));
    for (index, chapter) in book.chapters.iter().enumerate() {
        let chunks = chunk(&chapter.paragraphs, MAX_CHUNK_CHARS);
        let characters: usize = chunks.iter().map(|chunk| chunk.chars().count()).sum();
        println!(
            "{:>3}  {}  ({characters} characters, {} chunks)",
            index + 1,
            chapter.title,
            chunks.len()
        );
    }
    Ok(())
}

struct Take {
    audio: AudioSamples,
    verdict: Verdict,
}

/// Renders takes of chunks in one voice and checks each against one standard.
struct Recorder<'a> {
    narrator: Narrator,
    transcriber: Transcriber,
    voice: &'a Voice,
    speed: f64,
    standard: &'a Standard,
}

impl<'a> Recorder<'a> {
    /// Loads the narrator and the transcriber that checks it.
    fn load(voice: &'a Voice, store: &NarrationStore, standard: &'a Standard) -> Result<Self, Box<dyn Error>> {
        let narrator = Narrator::load()?;
        let transcriber = Transcriber::start(transcribe::CHECK_MODEL)?;
        Ok(Recorder { narrator, transcriber, voice, speed: store.speed(), standard })
    }

    /// Renders and checks one take. Seeds follow from the text, so `take` picks a
    /// repeatable take that differs from the others.
    fn take(&mut self, item: &Chunk, take: u64) -> Result<Take, Box<dyn Error>> {
        let base_seed = fnv(item.text.as_bytes(), 1) >> 1;
        let audio = self.narrator.say(self.voice, &item.text, self.speed, Some(base_seed + take))?;
        let transcript = self.transcriber.transcribe(&audio)?;
        let verdict = self.standard.judge(&item.text, &audio, &transcript);
        Ok(Take { audio, verdict })
    }

    /// Renders takes until one passes, or returns the best of them with its take count.
    fn best(&mut self, item: &Chunk) -> Result<(Take, u64), Box<dyn Error>> {
        let mut best: Option<Take> = None;
        for take in 0..MAX_TAKES {
            let current = self.take(item, take)?;
            if current.verdict.passes() {
                return Ok((current, take + 1));
            }
            let Verdict { quality, accuracy, .. } = &current.verdict;
            println!(
                "  take {} rejected: floor {:.0} dB, {:.1}s, recall {:.2}, extra {:.2}",
                take + 1,
                quality.floor_db,
                quality.seconds,
                accuracy.recall,
                accuracy.extra,
            );
            let replace = best
                .as_ref()
                .is_none_or(|existing| current.verdict.rank() > existing.verdict.rank());
            if replace {
                best = Some(current);
            }
        }
        let best = best.ok_or("no takes were rendered")?;
        Ok((best, MAX_TAKES))
    }
}

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
fn save_cover(book: &Book, out: &Path) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let Some(cover) = &book.cover else {
        return Ok(None);
    };
    let (_, extension) = image_type(cover).ok_or("the cover is not a JPEG or PNG")?;
    let path = out.join(format!("cover.{extension}"));
    fs::write(&path, cover)?;
    Ok(Some(path))
}

/// Copies a chapter's WAV to `tagged` with its title, the book and the author,
/// which `play` sends to the speaker's display.
fn tag_chapter(book: &Book, title: &str, untagged: &Path, tagged: &Path) -> Result<(), Box<dyn Error>> {
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg.args(["-v", "error", "-y", "-i"]).arg(untagged).args(["-c", "copy"]);
    ffmpeg.arg("-metadata").arg(format!("title={title}"));
    ffmpeg.arg("-metadata").arg(format!("album={}", book.title));
    if let Some(author) = &book.author {
        ffmpeg.arg("-metadata").arg(format!("artist={author}"));
    }
    let output = ffmpeg.arg(tagged).output()?;
    if !output.status.success() {
        return Err(format!("ffmpeg failed tagging {}: {}", tagged.display(), String::from_utf8_lossy(&output.stderr)).into());
    }
    fs::remove_file(untagged)?;
    Ok(())
}

/// Joins every complete chapter into one WAV each and a chaptered M4B.
fn assemble(book: &Book, store: &NarrationStore, out: &Path) -> Result<PathBuf, Box<dyn Error>> {
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
        return Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    fs::remove_file(&joined)?;
    Ok(m4b)
}

/// Writes one row per chunk with audio, so weak spots can be found and listened to.
fn write_report(book: &Book, store: &NarrationStore, standard: &Standard, out: &Path) -> Result<usize, Box<dyn Error>> {
    let mut report = String::from(
        "chapter\tchunk\tfile\ttakes\tfloor_db\tspeech_db\tseconds\tchars_per_s\trecall\textra\tflag\ttext\ttranscript\n",
    );
    let mut flagged = 0;
    for item in store.chunks().iter().filter(|item| store.has_audio(item)) {
        let path = store.audio_path(item);
        let audio = AudioSamples::from_wav_file(&path)?;
        let check = store.check(item)?;
        let verdict = standard.judge(&item.text, &audio, &check.transcript);
        let pass = verdict.passes();
        let Verdict { quality, accuracy, .. } = &verdict;
        let characters = item.text.chars().count();
        flagged += usize::from(!pass);
        writeln!(
            report,
            "{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.2}\t{:.2}\t{}\t{}\t{}",
            book.chapters[item.chapter].title,
            item.number(),
            path.file_name().unwrap_or_default().to_string_lossy(),
            check.takes,
            quality.floor_db,
            quality.speech_db,
            quality.seconds,
            characters as f32 / quality.seconds.max(0.01),
            accuracy.recall,
            accuracy.extra,
            if pass { "" } else { "CHECK" },
            item.text,
            check.transcript,
        )?;
    }
    fs::write(out.join("report.tsv"), report)?;
    Ok(flagged)
}

/// Narrates the selected chapters (or all of them) into `out`. Finished chunks
/// are cached, so an interrupted run picks up where it stopped. The report and
/// M4B cover every chapter rendered there so far.
pub fn narrate(
    voice_sample: &Path,
    epub: &Path,
    out: &Path,
    chapters: Option<Vec<usize>>,
    speed: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let book = read_book(epub)?;
    let selected: Vec<usize> = match chapters {
        Some(numbers) => {
            if let Some(bad) = numbers.iter().find(|&&n| n == 0 || n > book.chapters.len()) {
                return Err(format!("no chapter {bad}; this book has {}", book.chapters.len()).into());
            }
            numbers.iter().map(|number| number - 1).collect()
        }
        None => (0..book.chapters.len()).collect(),
    };

    let voice = Voice::load(voice_sample)?;
    let store = open_store(&book, &voice, out, speed)?;
    let standard = Standard::of(&store);

    let text_dir = out.join("text");
    fs::create_dir_all(&text_dir)?;
    for &chapter in &selected {
        let text = store
            .chunks()
            .iter()
            .filter(|item| item.chapter == chapter)
            .map(|item| item.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(text_dir.join(format!("{:02}.txt", chapter + 1)), text)?;
    }

    let pending = store.pending(&selected);
    let total = store.chunks().iter().filter(|item| selected.contains(&item.chapter)).count();
    println!("{total} chunks, {} to generate", pending.len());
    if !pending.is_empty() {
        let mut recorder = Recorder::load(&voice, &store, &standard)?;
        let started = Instant::now();
        let mut done = 0;
        for item in &pending {
            let (take, count) = recorder.best(item)?;
            store.store(item, &take.audio, &take.verdict.accuracy.transcript, count)?;
            done += 1;
            let elapsed = started.elapsed().as_secs_f32();
            println!(
                "[{done}/{}] {:.0}s of audio, take {count}, recall {:.2}, {:.0} min elapsed, about {:.0} min left",
                pending.len(),
                take.audio.duration_secs(),
                take.verdict.accuracy.recall,
                elapsed / 60.0,
                elapsed / done as f32 * (pending.len() - done) as f32 / 60.0,
            );
        }
    }

    let flagged = write_report(&book, &store, &standard, out)?;
    let m4b = assemble(&book, &store, out)?;
    println!("Wrote {}", m4b.display());
    println!("{flagged} chunks need checking; see {}", out.join("report.tsv").display());
    print_missing(&store);
    Ok(())
}

/// The `book` command that rebuilds the M4B without rendering anything new, or
/// `None` if no chapter is complete. Naming only complete chapters is what stops
/// it rendering whatever else is missing.
fn rebuild_command(voice: &Path, epub: &Path, out: &Path, complete: &[usize], missing: usize) -> Option<String> {
    if complete.is_empty() {
        return None;
    }
    let mut command = format!("audio-berry book {} {} {}", voice.display(), epub.display(), out.display());
    if missing > 0 {
        for chapter in complete {
            write!(command, " {}", chapter + 1).ok()?;
        }
    }
    Some(command)
}

fn print_missing(store: &NarrationStore) {
    let missing = store.missing_count();
    if missing > 0 {
        println!(
            "{missing} chunks in {} chapters have no audio, so those chapters aren't in the M4B",
            store.incomplete_chapter_count()
        );
    }
    let partly: Vec<String> = store.partly_rendered_chapters().iter().map(|chapter| (chapter + 1).to_string()).collect();
    if !partly.is_empty() {
        println!("Only partly rendered: chapter {}", partly.join(", "));
    }
}

/// Records new takes of only the chunks that `targets` name, keeping a new take
/// only if it beats the current one. The old audio moves to `replaced/`. This leaves
/// the M4B alone, so it's safe during playback; `book` rebuilds it afterwards.
pub fn retake(voice_sample: &Path, epub: &Path, out: &Path, targets: &[&str]) -> Result<(), Box<dyn Error>> {
    let book = read_book(epub)?;
    let voice = Voice::load(voice_sample)?;
    let store = open_store(&book, &voice, out, None)?;
    let standard = Standard::of(&store);
    let wanted = store.select(targets)?;

    let mut recorder = Recorder::load(&voice, &store, &standard)?;
    for (position, selected) in wanted.iter().enumerate() {
        let item = selected.chunk;
        let phrases = &selected.phrases;
        let path = store.audio_path(item);
        let heard_all = |transcript: &str| {
            phrases.iter().all(|phrase| accuracy::contains_phrase(transcript, phrase))
        };
        // The check file holds the take count, so seeds already tried are skipped.
        let (first_take, mut best_rank) = if store.has_audio(item) {
            let check = store.check(item)?;
            let audio = AudioSamples::from_wav_file(&path)?;
            let (plausible, clean, score) = standard.judge(&item.text, &audio, &check.transcript).rank();
            (check.takes, Some((plausible, clean, heard_all(&check.transcript), score)))
        } else {
            (0, None)
        };
        println!(
            "[{}/{}] chunk {}: {}",
            position + 1,
            wanted.len(),
            selected.numbers.join(", "),
            item.text.chars().take(60).collect::<String>()
        );

        let mut best: Option<Take> = None;
        let mut count = first_take;
        for take in first_take..first_take + MAX_RETAKES {
            let current = recorder.take(item, take)?;
            count = take + 1;
            let Verdict { quality, accuracy, .. } = &current.verdict;
            let (plausible, clean, score) = current.verdict.rank();
            let heard = heard_all(&accuracy.transcript);
            let rank = (plausible, clean, heard, score);
            let done = plausible && clean && heard && accuracy.exact();
            println!(
                "  take {count}: recall {:.2}, extra {:.2}, phrase heard {heard}, floor {:.0} dB",
                accuracy.recall, accuracy.extra, quality.floor_db,
            );
            println!("    heard: {}", accuracy.transcript);
            if best_rank.is_none_or(|existing| rank > existing) {
                best_rank = Some(rank);
                best = Some(current);
            }
            if done {
                break;
            }
        }
        match best {
            Some(take) => {
                if store.has_audio(item) {
                    store.retire(item)?;
                }
                store.store(item, &take.audio, &take.verdict.accuracy.transcript, count)?;
                println!("  replaced: {}", take.verdict.accuracy.transcript);
            }
            None => {
                // Still counts the tried seeds, so the next retake tries new ones.
                store.record_tried(item, count)?;
                println!("  kept the current take; none was better");
            }
        }
    }

    let flagged = write_report(&book, &store, &standard, out)?;
    println!("{flagged} chunks need checking; see {}", out.join("report.tsv").display());
    print_missing(&store);
    match rebuild_command(voice_sample, epub, out, &store.complete_chapters(), store.missing_count()) {
        Some(command) => println!("To rebuild the M4B and chapter WAVs, run:\n  {command}"),
        None => println!("No chapter is complete yet, so there's no M4B to rebuild."),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(complete: &[usize], missing: usize) -> Option<String> {
        rebuild_command(Path::new("v.wav"), Path::new("b.epub"), Path::new("out"), complete, missing)
    }

    #[test]
    fn a_partly_rendered_folder_rebuilds_by_naming_complete_chapters() {
        assert_eq!(command(&[2, 3], 40).unwrap(), "audio-berry book v.wav b.epub out 3 4");
    }

    #[test]
    fn a_finished_book_rebuilds_without_naming_chapters() {
        assert_eq!(command(&[0, 1], 0).unwrap(), "audio-berry book v.wav b.epub out");
    }

    #[test]
    fn nothing_rebuilds_with_no_complete_chapter() {
        assert_eq!(command(&[], 40), None);
    }
}
