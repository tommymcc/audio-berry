//! Narrates a whole book: generates checked chunks, then assembles the audio.

use crate::epub::{Book, read_book};
use crate::quality::{self, Quality};
use crate::text::{MAX_CHUNK_CHARS, chunk};
use crate::transcribe::{self, Accuracy, Transcriber};
use any_tts::{AudioSamples, ReferenceAudio, SynthesisRequest, TtsModel};
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const CACHE_VERSION: &str = "2";
const MAX_TAKES: u64 = 5;
/// New takes tried per chunk by `retake`.
const MAX_RETAKES: u64 = 6;
/// Audio frames generated per second of speech.
const TOKENS_PER_SECOND: f32 = 7.5;
const SPEECH_TARGET_DB: f32 = -21.0;
const CHUNK_GAP_SECONDS: f32 = 0.35;
const CHAPTER_GAP_SECONDS: f32 = 1.2;

struct Planned {
    chapter: usize,
    text: String,
    hash: String,
}

fn fnv(bytes: &[u8], seed: u64) -> u64 {
    bytes.iter().fold(seed ^ 0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn plan(book: &Book, selected: &[usize], voice_hash: u64) -> Vec<Planned> {
    let mut planned = Vec::new();
    for &chapter in selected {
        for text in chunk(&book.chapters[chapter].paragraphs, MAX_CHUNK_CHARS) {
            let key = format!("{CACHE_VERSION}\n{voice_hash:x}\n{text}");
            planned.push(Planned {
                chapter,
                hash: format!("{:016x}", fnv(key.as_bytes(), 0)),
                text,
            });
        }
    }
    planned
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
    quality: Quality,
    accuracy: Accuracy,
}

impl Take {
    fn plausible(&self, characters: usize) -> bool {
        quality::plausible_length(characters, self.quality.seconds)
    }

    fn clean(&self) -> bool {
        self.quality.clean()
    }

    fn acceptable(&self, characters: usize) -> bool {
        self.plausible(characters) && self.clean() && self.accuracy.acceptable()
    }

    /// Plausible length matters most, then clean audio, then accuracy.
    fn rank(&self, characters: usize) -> (bool, bool, f32) {
        (self.plausible(characters), self.clean(), self.accuracy.score())
    }
}

/// Renders and checks one take. Seeds follow from the text, so `take` picks a
/// repeatable take that differs from the others.
fn render_take(
    model: &dyn TtsModel,
    transcriber: &mut Transcriber,
    reference: &ReferenceAudio,
    planned: &Planned,
    out: &Path,
    take: u64,
) -> Result<Take, Box<dyn Error>> {
    let characters = planned.text.chars().count();
    let base_seed = fnv(planned.text.as_bytes(), 1) >> 1;
    // Stops runaway takes soon after they can no longer pass the length check.
    let max_seconds = (characters as f32 / 9.0 + 4.0) * 1.25;
    let max_tokens = (max_seconds * TOKENS_PER_SECOND) as usize;
    let candidate = out.join("chunks").join(format!("{}.take.wav", planned.hash));
    // SAFETY: nothing else reads or writes the environment during this
    // call, as the model is only used from this thread between requests.
    unsafe { std::env::set_var("VIBEVOICE_SEED", (base_seed + take).to_string()) };
    let request = SynthesisRequest::new(planned.text.as_str())
        .with_reference_audio(reference.clone())
        .with_cfg_scale(crate::CFG_SCALE)
        .with_max_tokens(max_tokens);
    let audio = model.synthesize(&request)?;
    audio.save_wav(&candidate)?;
    let transcript = transcriber.transcribe(&candidate)?;
    fs::remove_file(&candidate)?;
    Ok(Take {
        quality: quality::measure(&audio.samples, audio.sample_rate),
        accuracy: transcribe::compare(&planned.text, &transcript),
        audio,
    })
}

/// Renders takes until one passes, or returns the best of them with its take count.
fn render(
    model: &dyn TtsModel,
    transcriber: &mut Transcriber,
    reference: &ReferenceAudio,
    planned: &Planned,
    out: &Path,
) -> Result<(Take, u64), Box<dyn Error>> {
    let characters = planned.text.chars().count();
    let mut best: Option<Take> = None;
    for take in 0..MAX_TAKES {
        let current = render_take(model, transcriber, reference, planned, out, take)?;
        if current.acceptable(characters) {
            return Ok((current, take + 1));
        }
        println!(
            "  take {} rejected: floor {:.0} dB, {:.1}s, recall {:.2}, extra {:.2}",
            take + 1,
            current.quality.floor_db,
            current.quality.seconds,
            current.accuracy.recall,
            current.accuracy.extra,
        );
        let replace = best
            .as_ref()
            .is_none_or(|existing| current.rank(characters) > existing.rank(characters));
        if replace {
            best = Some(current);
        }
    }
    let best = best.ok_or("no takes were rendered")?;
    Ok((best, MAX_TAKES))
}

/// Saves a chunk and its check file, replacing the audio in one step.
fn store_take(path: &Path, take: &Take, count: u64) -> Result<(), Box<dyn Error>> {
    let transcript = take.accuracy.transcript.replace(['\n', '\t'], " ");
    fs::write(path.with_extension("check"), format!("{count}\n{transcript}"))?;
    let temporary = path.with_extension("tmp");
    take.audio.save_wav(&temporary)?;
    fs::rename(&temporary, path)?;
    Ok(())
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

/// Joins chunks into one WAV per chapter and a chaptered M4B.
fn assemble(book: &Book, planned: &[Planned], selected: &[usize], out: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let chapter_dir = out.join("chapters");
    fs::create_dir_all(&chapter_dir)?;
    let mut sample_rate = 24_000;
    let mut all = Vec::new();
    let mut metadata = format!(";FFMETADATA1\ntitle={}\n", metadata_escape(&book.title));
    if let Some(author) = &book.author {
        writeln!(metadata, "artist={}", metadata_escape(author))?;
    }

    for (position, &chapter) in selected.iter().enumerate() {
        let mut samples = Vec::new();
        for item in planned.iter().filter(|item| item.chapter == chapter) {
            let audio = AudioSamples::from_wav_file(chunk_path(out, &item.hash))?;
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
        AudioSamples::new(samples.clone(), sample_rate).save_wav(chapter_dir.join(name))?;

        let start = all.len() as u64 * 1000 / u64::from(sample_rate);
        all.extend_from_slice(&samples);
        let end = all.len() as u64 * 1000 / u64::from(sample_rate);
        write!(
            metadata,
            "[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle={}\n",
            metadata_escape(title)
        )?;
        if position + 1 < selected.len() {
            all.resize(all.len() + (CHAPTER_GAP_SECONDS * sample_rate as f32) as usize, 0.0);
        }
    }

    let joined = out.join("joined.wav");
    AudioSamples::new(all, sample_rate).save_wav(&joined)?;
    let metadata_path = out.join("chapters.txt");
    fs::write(&metadata_path, metadata)?;
    let m4b = out.join(format!("{}.m4b", safe_name(&book.title)));
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(&joined)
        .arg("-i")
        .arg(&metadata_path)
        .args(["-map_metadata", "1", "-map_chapters", "1", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", "-f", "mp4"])
        .arg(&m4b)
        .output()?;
    if !output.status.success() {
        return Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    fs::remove_file(&joined)?;
    Ok(m4b)
}

fn chunk_path(out: &Path, hash: &str) -> PathBuf {
    out.join("chunks").join(format!("{hash}.wav"))
}

/// Writes one row per chunk, so weak spots can be found and listened to.
fn write_report(book: &Book, planned: &[Planned], out: &Path) -> Result<usize, Box<dyn Error>> {
    let mut report = String::from(
        "chapter\tchunk\tfile\ttakes\tfloor_db\tspeech_db\tseconds\tchars_per_s\trecall\textra\tflag\ttext\ttranscript\n",
    );
    let mut flagged = 0;
    for (index, item) in planned.iter().enumerate() {
        let audio = AudioSamples::from_wav_file(chunk_path(out, &item.hash))?;
        let quality = quality::measure(&audio.samples, audio.sample_rate);
        let sidecar = fs::read_to_string(chunk_path(out, &item.hash).with_extension("check"))?;
        let (takes, transcript) = sidecar.split_once('\n').ok_or("malformed chunk check file")?;
        let accuracy = transcribe::compare(&item.text, transcript);
        let characters = item.text.chars().count();
        let pass = quality.clean()
            && quality::plausible_length(characters, quality.seconds)
            && accuracy.acceptable();
        flagged += usize::from(!pass);
        writeln!(
            report,
            "{}\t{index}\t{}.wav\t{takes}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.2}\t{:.2}\t{}\t{}\t{transcript}",
            book.chapters[item.chapter].title,
            item.hash,
            quality.floor_db,
            quality.speech_db,
            quality.seconds,
            characters as f32 / quality.seconds.max(0.01),
            accuracy.recall,
            accuracy.extra,
            if pass { "" } else { "CHECK" },
            item.text,
        )?;
    }
    fs::write(out.join("report.tsv"), report)?;
    Ok(flagged)
}

/// Narrates the selected chapters (or all of them) into `out`. Finished chunks
/// are cached, so an interrupted run picks up where it stopped.
pub fn narrate(
    voice: &Path,
    epub: &Path,
    out: &Path,
    chapters: Option<Vec<usize>>,
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

    let reference = crate::load_reference(voice)?;
    let voice_bytes: Vec<u8> = reference.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let planned = plan(&book, &selected, fnv(&voice_bytes, 0));

    fs::create_dir_all(out.join("chunks"))?;
    let text_dir = out.join("text");
    fs::create_dir_all(&text_dir)?;
    for &chapter in &selected {
        let text = planned
            .iter()
            .filter(|item| item.chapter == chapter)
            .map(|item| item.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(text_dir.join(format!("{:02}.txt", chapter + 1)), text)?;
    }

    let pending = planned.iter().filter(|item| !chunk_path(out, &item.hash).exists()).count();
    println!("{} chunks, {pending} to generate", planned.len());
    if pending > 0 {
        let model = crate::load_narrator()?;
        let mut transcriber = Transcriber::start()?;
        let started = Instant::now();
        let mut done = 0;
        for item in &planned {
            let path = chunk_path(out, &item.hash);
            if path.exists() {
                continue;
            }
            let (take, count) = render(model.as_ref(), &mut transcriber, &reference, item, out)?;
            store_take(&path, &take, count)?;
            done += 1;
            let elapsed = started.elapsed().as_secs_f32();
            println!(
                "[{done}/{pending}] {:.0}s of audio, take {count}, recall {:.2}, {:.0} min elapsed, about {:.0} min left",
                take.audio.duration_secs(),
                take.accuracy.recall,
                elapsed / 60.0,
                elapsed / done as f32 * (pending - done) as f32 / 60.0,
            );
        }
    }

    let flagged = write_report(&book, &planned, out)?;
    let m4b = assemble(&book, &planned, &selected, out)?;
    println!("Wrote {}", m4b.display());
    println!("{flagged} chunks need checking; see {}", out.join("report.tsv").display());
    Ok(())
}

/// Finds the chunks a target names: a report chunk number, or words from the book.
fn matching_chunks(planned: &[Planned], target: &str) -> Vec<usize> {
    if let Ok(number) = target.parse::<usize>() {
        return if number < planned.len() { vec![number] } else { Vec::new() };
    }
    planned
        .iter()
        .enumerate()
        .filter(|(_, item)| transcribe::contains_phrase(&item.text, target))
        .map(|(index, _)| index)
        .collect()
}

/// Records new takes of the chunks that `targets` name, keeping a new take only
/// if it beats the current one. The old audio moves to `replaced/`. This leaves
/// the M4B alone, so it's safe during playback; `book` rebuilds it afterwards.
pub fn retake(voice: &Path, epub: &Path, out: &Path, targets: &[&str]) -> Result<(), Box<dyn Error>> {
    let book = read_book(epub)?;
    let selected: Vec<usize> = (0..book.chapters.len()).collect();
    let reference = crate::load_reference(voice)?;
    let voice_bytes: Vec<u8> = reference.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let planned = plan(&book, &selected, fnv(&voice_bytes, 0));

    // A phrase must be heard in the new take, or it hasn't fixed what was asked.
    let mut wanted: Vec<(usize, Vec<&str>)> = Vec::new();
    for target in targets {
        let found = matching_chunks(&planned, target);
        if found.is_empty() {
            return Err(format!("nothing in the book matches \"{target}\"").into());
        }
        let phrase = target.parse::<usize>().is_err().then_some(*target);
        for index in found {
            match wanted.iter_mut().find(|(existing, _)| *existing == index) {
                Some((_, phrases)) => phrases.extend(phrase),
                None => wanted.push((index, phrase.into_iter().collect())),
            }
        }
    }
    // Text changes can leave chunks with no audio at all, which need recording too.
    for (index, item) in planned.iter().enumerate() {
        if !chunk_path(out, &item.hash).exists() && !wanted.iter().any(|(existing, _)| *existing == index) {
            wanted.push((index, Vec::new()));
        }
    }
    wanted.sort_by_key(|(index, _)| *index);

    let model = crate::load_narrator()?;
    let mut transcriber = Transcriber::start()?;
    fs::create_dir_all(out.join("replaced"))?;
    for (position, (index, phrases)) in wanted.iter().enumerate() {
        let item = &planned[*index];
        let path = chunk_path(out, &item.hash);
        let characters = item.text.chars().count();
        let heard_all = |transcript: &str| {
            phrases.iter().all(|phrase| transcribe::contains_phrase(transcript, phrase))
        };
        // The check file holds the take count, so seeds already tried are skipped.
        let (first_take, mut best_rank) = if path.exists() {
            let sidecar = fs::read_to_string(path.with_extension("check"))?;
            let (takes, transcript) = sidecar.split_once('\n').ok_or("malformed chunk check file")?;
            let audio = AudioSamples::from_wav_file(&path)?;
            let current = Take {
                quality: quality::measure(&audio.samples, audio.sample_rate),
                accuracy: transcribe::compare(&item.text, transcript),
                audio,
            };
            let (plausible, clean, score) = current.rank(characters);
            (takes.parse::<u64>()?, Some((plausible, clean, heard_all(transcript), score)))
        } else {
            (0, None)
        };
        println!("[{}/{}] chunk {index}: {}", position + 1, wanted.len(), item.text.chars().take(60).collect::<String>());

        let mut best: Option<Take> = None;
        let mut count = first_take;
        for take in first_take..first_take + MAX_RETAKES {
            let current = render_take(model.as_ref(), &mut transcriber, &reference, item, out, take)?;
            count = take + 1;
            let (plausible, clean, score) = current.rank(characters);
            let heard = heard_all(&current.accuracy.transcript);
            let rank = (plausible, clean, heard, score);
            let exact = current.accuracy.recall >= 0.99 && current.accuracy.extra <= 0.01;
            println!(
                "  take {count}: recall {:.2}, extra {:.2}, phrase heard {heard}, floor {:.0} dB",
                current.accuracy.recall, current.accuracy.extra, current.quality.floor_db,
            );
            println!("    heard: {}", current.accuracy.transcript);
            if best_rank.is_none_or(|existing| rank > existing) {
                best_rank = Some(rank);
                best = Some(current);
            }
            if plausible && clean && heard && exact {
                break;
            }
        }
        match best {
            Some(take) => {
                if path.exists() {
                    let old = out.join("replaced").join(format!("{}-{first_take}.wav", item.hash));
                    fs::rename(&path, &old)?;
                    fs::rename(path.with_extension("check"), old.with_extension("check"))?;
                }
                store_take(&path, &take, count)?;
                println!("  replaced: {}", take.accuracy.transcript);
            }
            None => {
                // Still counts the tried seeds, so the next retake tries new ones.
                let sidecar = fs::read_to_string(path.with_extension("check"))?;
                let transcript = sidecar.split_once('\n').map_or("", |(_, transcript)| transcript);
                fs::write(path.with_extension("check"), format!("{count}\n{transcript}"))?;
                println!("  kept the current take; none was better");
            }
        }
    }

    let flagged = write_report(&book, &planned, out)?;
    println!("{flagged} chunks need checking; see {}", out.join("report.tsv").display());
    println!("Run `book` with the same arguments to rebuild the M4B.");
    Ok(())
}
