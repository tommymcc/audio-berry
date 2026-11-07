//! Narrates a whole book, and retakes chosen chunks: renders checked takes,
//! reports on them, then assembles the audio.

use crate::accuracy::Allowances;
use crate::assemble::assemble;
use crate::checks::{Standard, Verdict};
use crate::epub::{Book, read_book};
use crate::narrator::Narrator;
use crate::progress::Progress;
use crate::store::{Chunk, NarrationStore, RenderSettings, fnv};
use crate::transcribe::{self, Transcriber};
use crate::voice::{Voice, allowances_path, read_allowances};
use any_tts::AudioSamples;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

/// Takes tried per chunk by `book`.
pub const MAX_TAKES: u64 = 5;
/// New takes tried per chunk by `retake`.
const MAX_RETAKES: u64 = 6;

/// Reads the voice's allowances, and says how many there are, since they change
/// which takes pass.
fn read_allowances_saying_so(voice_sample: &Path) -> Result<Allowances, Box<dyn Error>> {
    let allowances = read_allowances(voice_sample)?;
    if !allowances.is_empty() {
        println!("Allowances: {} from {}", allowances.len(), allowances_path(voice_sample).display());
    }
    Ok(allowances)
}

fn open_store(
    book: &Book,
    voice: &Voice,
    out: &Path,
    speed: Option<f64>,
    noise_limit_db: Option<f32>,
) -> Result<NarrationStore, Box<dyn Error>> {
    NarrationStore::open(out, &book.title, &book.chunks(), voice.hash(), &RenderSettings::from_env(speed), noise_limit_db)
}

/// Prints each chapter's number, title and size.
pub fn list_chapters(path: &Path) -> Result<(), Box<dyn Error>> {
    let book = read_book(path)?;
    println!("{} by {}", book.title, book.author.as_deref().unwrap_or("unknown"));
    for (index, chapter) in book.chapters.iter().enumerate() {
        let chunks = chapter.chunks();
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
    /// Each rejected take is reported through `progress`.
    fn best(&mut self, item: &Chunk, progress: &mut Progress) -> Result<(Take, u64), Box<dyn Error>> {
        let mut best: Option<Take> = None;
        for take in 0..MAX_TAKES {
            let current = self.take(item, take)?;
            if current.verdict.passes() {
                return Ok((current, take + 1));
            }
            let Verdict { quality, accuracy, .. } = &current.verdict;
            progress.line(&format!(
                "  take {} rejected: floor {:.0} dB, {:.1}s, recall {:.2}, extra {:.2}",
                take + 1,
                quality.floor_db,
                quality.seconds,
                accuracy.recall,
                accuracy.extra,
            ))?;
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
    noise_limit_db: Option<f32>,
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

    let allowances = read_allowances_saying_so(voice_sample)?;
    let voice = Voice::load(voice_sample, noise_limit_db)?;
    let store = open_store(&book, &voice, out, speed, noise_limit_db)?;
    let standard = Standard::of(&store, allowances);

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
        let mut progress = Progress::new(total, total - pending.len());
        for item in &pending {
            let (take, count) = recorder.best(item, &mut progress)?;
            store.store(item, &take.audio, &take.verdict.accuracy.transcript, count)?;
            progress.chunk_done(&format!(
                "{:.0}s of audio, take {count}, recall {:.2}",
                take.audio.duration_secs(),
                take.verdict.accuracy.recall,
            ))?;
        }
        progress.finish();
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
pub fn retake(
    voice_sample: &Path,
    epub: &Path,
    out: &Path,
    targets: &[&str],
    noise_limit_db: Option<f32>,
) -> Result<(), Box<dyn Error>> {
    let book = read_book(epub)?;
    let allowances = read_allowances_saying_so(voice_sample)?;
    let voice = Voice::load(voice_sample, noise_limit_db)?;
    let store = open_store(&book, &voice, out, None, noise_limit_db)?;
    let standard = Standard::of(&store, allowances);
    let wanted = store.select(targets)?;

    let mut recorder = Recorder::load(&voice, &store, &standard)?;
    let mut progress = Progress::untimed(wanted.len());
    for (position, selected) in wanted.iter().enumerate() {
        let item = selected.chunk;
        let phrases = &selected.phrases;
        let path = store.audio_path(item);
        let heard_all = |transcript: &str| {
            phrases.iter().all(|phrase| standard.heard(phrase, transcript))
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
        let numbers = selected.numbers.join(", ");
        progress.line(&format!(
            "[{}/{}] chunk {numbers}: {}",
            position + 1,
            wanted.len(),
            item.text.chars().take(60).collect::<String>()
        ))?;

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
            progress.line(&format!(
                "  take {count}: recall {:.2}, extra {:.2}, phrase heard {heard}, floor {:.0} dB",
                accuracy.recall, accuracy.extra, quality.floor_db,
            ))?;
            progress.line(&format!("    heard: {}", accuracy.transcript))?;
            if best_rank.is_none_or(|existing| rank > existing) {
                best_rank = Some(rank);
                best = Some(current);
            }
            if done {
                break;
            }
        }
        let outcome = match best {
            Some(take) => {
                if store.has_audio(item) {
                    store.retire(item)?;
                }
                store.store(item, &take.audio, &take.verdict.accuracy.transcript, count)?;
                progress.line(&format!("  replaced: {}", take.verdict.accuracy.transcript))?;
                "replaced"
            }
            None => {
                // Still counts the tried seeds, so the next retake tries new ones.
                store.record_tried(item, count)?;
                progress.line("  kept the current take; none was better")?;
                "kept the current take"
            }
        };
        progress.chunk_done(&format!("chunk {numbers}: {outcome}"))?;
    }
    progress.finish();

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
