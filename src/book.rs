//! Narrates a whole book, and retakes chosen chunks: renders checked takes,
//! reports on them, then assembles the audio.

use crate::accuracy::{Accuracy, Allowances};
use crate::assemble::assemble;
use crate::checker::{Checker, Transcript};
use crate::checks::{Goal, Standard, Verdict};
use crate::epub::{Book, read_book};
use crate::estimate::{Pace, split_above_seconds, warn_if_split};
use crate::narrator::Narrator;
use crate::progress::Progress;
use crate::store::{self, Chunk, KeepSettings, NarrationStore, RenderSettings, fnv};
use crate::takes::{Kept, TakePolicy};
use crate::text::ChunkLimit;
use crate::voice::{Voice, allowances_path, read_allowances};
use any_tts::AudioSamples;
use color_eyre::eyre::{Result, bail};
use std::fmt::Write as _;
use fs_err as fs;
use std::path::Path;

/// Reads the voice's allowances, and says how many there are, since they change
/// which takes pass.
fn read_allowances_saying_so(voice_sample: &Path) -> Result<Allowances> {
    let allowances = read_allowances(voice_sample)?;
    if !allowances.is_empty() {
        println!("Allowances: {} from {}", allowances.len(), allowances_path(voice_sample).display());
    }
    Ok(allowances)
}

/// A book opened for rendering in an output directory, in one voice, with the
/// standard its takes are judged by and the take policy that picks them.
struct Session {
    book: Book,
    voice: Voice,
    store: NarrationStore,
    standard: Standard,
    takes: TakePolicy,
    /// The voice's pace at the session's speed.
    pace: Pace,
}

impl Session {
    /// Settles what decides which take is kept in `out` from `given` and what's
    /// saved there, loads the voice sample judged against the settled noise limit,
    /// and opens `out` for `book` in that voice at `speed`, or the speed it saved,
    /// in chunks kept under the length any-tts splits.
    fn open(voice_sample: &Path, book: Book, out: &Path, speed: Option<f64>, given: KeepSettings) -> Result<Session> {
        let allowances = read_allowances_saying_so(voice_sample)?;
        let settled = store::settle(out, given)?;
        let rules = *settled.rules();
        let voice = Voice::load(voice_sample, settled.sample_noise_limit())?;
        let render = RenderSettings::from_env(speed);
        let pace = voice.pace(settled.speed(&render));
        let limit = ChunkLimit::under_split(&pace, split_above_seconds());
        let store = settled.open(&book.title, &book.chunks(&limit), &limit, voice.hash(), &render)?;
        let standard = Standard::of(&rules, store.speed(), allowances);
        Ok(Session { book, voice, store, standard, takes: rules.takes, pace })
    }

    /// Warns of each of `chunks` that any-tts will split.
    fn warn_of_splits<'a>(&self, chunks: impl IntoIterator<Item = &'a Chunk>) {
        for item in chunks {
            warn_if_split(&self.pace, &item.text, &format!("chunk {}", item.number()));
        }
    }
}

/// The take's pitch range for the progress output, when ranked by it.
fn pitch_range_note(standard: &Standard, verdict: &Verdict) -> String {
    match (standard.pitch_range_target(), verdict.pitch_range) {
        (None, _) => String::new(),
        (Some(_), Some(range)) => format!(", pitch range {range:.1} st"),
        (Some(_), None) => ", pitch range unmeasured".into(),
    }
}

/// The phrases heard in the wrong place, for the progress output.
fn moved_suffix(accuracy: &Accuracy) -> String {
    if accuracy.in_order() { String::new() } else { format!(", moved {}", accuracy.moved_list()) }
}

/// The mid-phrase pauses found, for the progress output.
fn mid_phrase_pauses_suffix(verdict: &Verdict) -> String {
    if verdict.no_mid_phrase_pause() {
        String::new()
    } else {
        format!(", paused {}", verdict.mid_phrase_pause_list())
    }
}

/// Prints each chapter's number, title and size.
pub fn list_chapters(path: &Path) -> Result<()> {
    let book = read_book(path)?;
    println!("{} by {}", book.title, book.author.as_deref().unwrap_or("unknown"));
    for (index, chapter) in book.chapters.iter().enumerate() {
        println!("{:>3}  {}  ({} characters)", index + 1, chapter.title, chapter.spoken_chars());
    }
    Ok(())
}

struct Take {
    audio: AudioSamples,
    heard: Transcript,
    verdict: Verdict,
}

/// Renders takes of chunks in one voice and checks each against one standard.
struct Recorder<'a> {
    narrator: Narrator,
    checker: Checker,
    voice: &'a Voice,
    speed: f64,
    standard: &'a Standard,
}

impl<'a> Recorder<'a> {
    /// Loads the narrator and the checker that hears its takes, for the session's
    /// voice, speed and standard.
    fn load(session: &'a Session) -> Result<Self> {
        let narrator = Narrator::load()?;
        let checker = Checker::start()?;
        let Session { voice, store, standard, .. } = session;
        Ok(Recorder { narrator, checker, voice, speed: store.speed(), standard })
    }

    /// Renders and checks one take for `goal`. Seeds follow from the text, so
    /// `take` picks a repeatable take that differs from the others.
    fn take(&mut self, item: &Chunk, goal: &Goal, take: u64) -> Result<Take> {
        let base_seed = fnv(item.text.as_bytes(), 1) >> 1;
        let audio = self.narrator.say(self.voice, &item.text, self.speed, Some(base_seed + take))?;
        let heard = self.checker.transcribe(&audio)?;
        let verdict = self.standard.judge(goal, &item.text, &audio, &heard.heard());
        Ok(Take { audio, heard, verdict })
    }

    /// Renders takes from `first` on as `policy` says, for `goal`, and keeps the
    /// best. Takes are reported through `progress`: for `book` each one that isn't
    /// good enough, and for `retake` every one, with what was heard.
    fn best(
        &mut self,
        item: &Chunk,
        goal: &Goal,
        first: u64,
        policy: TakePolicy,
        progress: &mut Progress,
    ) -> Result<Kept<Take>> {
        let standard = self.standard;
        let render = |take| {
            let current = self.take(item, goal, take)?;
            let Verdict { quality, accuracy, .. } = &current.verdict;
            match goal {
                Goal::Book if !current.verdict.good_enough() => progress.line(&format!(
                    "  take {} rejected: floor {:.0} dB, {:.1}s, recall {:.2}, extra {:.2}{}{}",
                    take + 1,
                    quality.floor_db,
                    quality.seconds,
                    accuracy.recall,
                    accuracy.extra,
                    moved_suffix(accuracy),
                    mid_phrase_pauses_suffix(&current.verdict),
                ))?,
                Goal::Book => {}
                Goal::Retake { .. } => {
                    progress.line(&format!(
                        "  take {}: recall {:.2}, extra {:.2}{}{}, phrase heard {}, floor {:.0} dB{}",
                        take + 1,
                        accuracy.recall,
                        accuracy.extra,
                        moved_suffix(accuracy),
                        mid_phrase_pauses_suffix(&current.verdict),
                        current.verdict.heard(),
                        quality.floor_db,
                        pitch_range_note(standard, &current.verdict),
                    ))?;
                    progress.line(&format!("    heard: {}", accuracy.transcript))?;
                }
            }
            Ok(current)
        };
        let judge = |take: &Take| take.verdict.judged();
        policy.render_best(first, goal.most_takes(), render, judge, |take| take.verdict.good_enough())
    }
}

/// Writes one row per chunk with audio, so weak spots can be found and listened to.
fn write_report(book: &Book, store: &NarrationStore, standard: &Standard, out: &Path) -> Result<usize> {
    // New columns go last, so anything reading columns by position still works.
    let mut report = String::from(
        "chapter\tchunk\tfile\ttakes\tfloor_db\tspeech_db\tseconds\tchars_per_s\trecall\textra\tflag\ttext\ttranscript\t\
         pitch_range_st\tchecker\tmoved\tpauses\n",
    );
    let mut flagged = 0;
    for item in store.chunks().iter().filter(|item| store.has_audio(item)) {
        let path = store.audio_path(item);
        let audio = AudioSamples::from_wav_file(&path)?;
        let check = store.check(item)?;
        let verdict = standard.judge(&Goal::Book, &item.text, &audio, &check.heard());
        let pass = verdict.good_enough();
        let Verdict { quality, accuracy, .. } = &verdict;
        let characters = item.text.chars().count();
        flagged += usize::from(!pass);
        writeln!(
            report,
            "{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.2}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
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
            verdict.pitch_range.map(|range| format!("{range:.1}")).unwrap_or_default(),
            check.checker.as_deref().unwrap_or_default(),
            accuracy.moved_list(),
            verdict.pauses_column(),
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
    keep: KeepSettings,
) -> Result<()> {
    let book = read_book(epub)?;
    let selected: Vec<usize> = match chapters {
        Some(numbers) => {
            if let Some(bad) = numbers.iter().find(|&&n| n == 0 || n > book.chapters.len()) {
                bail!("no chapter {bad}; this book has {}", book.chapters.len());
            }
            numbers.iter().map(|number| number - 1).collect()
        }
        None => (0..book.chapters.len()).collect(),
    };

    let session = Session::open(voice_sample, book, out, speed, keep)?;
    let Session { book, store, standard, takes, .. } = &session;

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
    session.warn_of_splits(pending.iter().copied());
    if !pending.is_empty() {
        let mut recorder = Recorder::load(&session)?;
        let mut progress = Progress::new(total, total - pending.len());
        for item in &pending {
            let kept = recorder.best(item, &Goal::Book, 0, *takes, &mut progress)?;
            let take = &kept.take;
            store.store(item, &take.audio, &take.heard, kept.tried())?;
            progress.chunk_done(&format!(
                "{:.0}s of audio, {}, recall {:.2}{}",
                take.audio.duration_secs(),
                takes.describe(&kept),
                take.verdict.accuracy.recall,
                pitch_range_note(standard, &take.verdict),
            ))?;
        }
        progress.finish();
    }

    let flagged = write_report(book, store, standard, out)?;
    let m4b = assemble(book, store, out)?;
    println!("Wrote {}", m4b.display());
    println!("{flagged} chunks need checking; see {}", out.join("report.tsv").display());
    print_missing(store);
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
    keep: KeepSettings,
) -> Result<()> {
    let session = Session::open(voice_sample, read_book(epub)?, out, None, keep)?;
    let Session { book, store, standard, takes, .. } = &session;
    let wanted = store.select(targets)?;
    session.warn_of_splits(wanted.iter().map(|selected| selected.chunk));

    let mut recorder = Recorder::load(&session)?;
    let mut progress = Progress::untimed(wanted.len());
    for (position, selected) in wanted.iter().enumerate() {
        let item = selected.chunk;
        let goal = Goal::Retake { phrases: &selected.phrases };
        let path = store.audio_path(item);
        // The check file holds the take count, so seeds already tried are skipped.
        let (first_take, current) = if store.has_audio(item) {
            let check = store.check(item)?;
            let audio = AudioSamples::from_wav_file(&path)?;
            let verdict = standard.judge(&goal, &item.text, &audio, &check.heard());
            (check.takes, Some(verdict.judged()))
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

        let kept = recorder.best(item, &goal, first_take, *takes, &mut progress)?;
        let better = current.is_none_or(|current| kept.take.verdict.judged() > current);
        let outcome = if better {
            let transcript = &kept.take.heard.text;
            store.replace(item, &kept.take.audio, &kept.take.heard, kept.tried())?;
            let pitch_range = pitch_range_note(standard, &kept.take.verdict);
            progress.line(&format!("  replaced with {}{pitch_range}: {transcript}", takes.describe(&kept)))?;
            "replaced"
        } else {
            // Still counts the tried seeds, so the next retake tries new ones.
            store.record_tried(item, kept.tried())?;
            progress.line("  kept the current take; none was better")?;
            "kept the current take"
        };
        progress.chunk_done(&format!("chunk {numbers}: {outcome}"))?;
    }
    progress.finish();

    let flagged = write_report(book, store, standard, out)?;
    println!("{flagged} chunks need checking; see {}", out.join("report.tsv").display());
    print_missing(store);
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
