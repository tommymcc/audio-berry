//! An output directory: one narration of one book, in one voice, with one set
//! of render settings. Owns chunk numbering, the cache key and the files.

use crate::quality::{MAX_FLOOR_DB, valid_noise_limit};
use crate::accuracy::contains_phrase;
use any_tts::{AudioSamples, DeviceSelection};
use any_tts::models::omnivoice::{OmniVoiceRuntimeChoice, preferred_runtime_choice};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

const CACHE_VERSION: &str = "3";
const DEFAULT_PRECISION: &str = "f16";
const NORMAL_SPEED: f64 = 1.0;
const MANIFEST: &str = "narration.txt";

pub fn fnv(bytes: &[u8], seed: u64) -> u64 {
    bytes.iter().fold(seed ^ 0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Settings that change a chunk's audio.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderSettings {
    pub precision: String,
    /// `None` uses what the folder saved, or normal speed in a new folder.
    pub speed: Option<f64>,
}

impl RenderSettings {
    pub fn from_env(speed: Option<f64>) -> Self {
        let dtype = std::env::var("OMNIVOICE_DTYPE").ok();
        Self { speed, ..Self::from_dtype(dtype.as_deref(), preferred_runtime_choice()) }
    }

    /// Mirrors the model, which ignores values it doesn't recognise and can't run
    /// BF16 on the CPU or Metal.
    fn from_dtype(value: Option<&str>, runtime: OmniVoiceRuntimeChoice) -> Self {
        let value = value.unwrap_or_default().trim().to_ascii_lowercase();
        let precision = match (value.as_str(), runtime.device) {
            ("bf16", DeviceSelection::Cpu | DeviceSelection::Metal(_)) => "f32".into(),
            ("f32" | "f16" | "bf16", _) => value,
            _ => runtime.dtype_label().into(),
        };
        Self { precision, speed: None }
    }
}

/// A piece of a chapter's text, narrated as one piece of audio.
pub struct Chunk {
    /// Zero-based position of the chapter in the book.
    pub chapter: usize,
    position: usize,
    pub text: String,
    hash: String,
}

impl Chunk {
    /// The chapter number and position in the chapter, such as `3.12`.
    pub fn number(&self) -> String {
        format!("{}.{}", self.chapter + 1, self.position + 1)
    }
}

/// What's known about a stored take without loading its audio.
#[derive(Debug, PartialEq, Eq)]
pub struct Check {
    /// Seeds tried so far, so the next retake tries new ones.
    pub takes: u64,
    pub transcript: String,
}

fn read_manifest(out: &Path) -> Result<Option<Manifest>, Box<dyn Error>> {
    match fs::read_to_string(out.join(MANIFEST)) {
        Ok(text) => Ok(Some(Manifest::parse(&text)?)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Reads a chunk number such as `3.12` as its chapter and position.
fn parse_number(target: &str) -> Option<(usize, usize)> {
    let (chapter, position) = target.split_once('.')?;
    Some((chapter.parse().ok()?, position.parse().ok()?))
}

/// A chunk that a retake should record.
pub struct Selected<'a> {
    pub chunk: &'a Chunk,
    /// Words that must be heard in a new take.
    pub phrases: Vec<String>,
    /// Every chunk number that shares this audio.
    pub numbers: Vec<String>,
}

/// What a folder narrates, saved on first use.
#[derive(Clone)]
struct Manifest {
    book: String,
    voice: String,
    precision: String,
    speed: f64,
    /// Not a render setting: it only decides which take is kept, so it can
    /// change without touching the audio. `None` in output directories saved
    /// before it was, until a limit is given.
    noise_limit_db: Option<f32>,
}

impl Manifest {
    fn render(&self) -> String {
        let mut text = format!(
            "book={}\nvoice={}\nprecision={}\nspeed={}\n",
            self.book, self.voice, self.precision, self.speed
        );
        if let Some(limit) = self.noise_limit_db {
            text.push_str(&format!("noise_limit_db={limit}\n"));
        }
        text
    }

    fn parse(text: &str) -> Result<Self, Box<dyn Error>> {
        let field = |name: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(name)?.strip_prefix('='))
                .map(str::to_string)
                .ok_or_else(|| format!("{MANIFEST} has no {name}"))
        };
        Ok(Self {
            book: field("book")?,
            voice: field("voice")?,
            precision: field("precision")?,
            speed: field("speed").map_or(Ok(NORMAL_SPEED), |speed| speed.parse())?,
            noise_limit_db: match field("noise_limit_db") {
                Ok(value) => Some(valid_noise_limit(&value).ok_or_else(|| {
                    format!("{MANIFEST} has noise_limit_db={value}, which isn't a level from -100 to 0 dB. Correct or delete that line")
                })?),
                Err(_) => None,
            },
        })
    }

    fn require(&self, wanted: &Self, out: &Path) -> Result<(), Box<dyn Error>> {
        let fields = [
            ("book", &self.book, &wanted.book),
            ("voice", &self.voice, &wanted.voice),
            ("precision", &self.precision, &wanted.precision),
            ("speed", &self.speed.to_string(), &wanted.speed.to_string()),
        ];
        for (name, saved, asked) in fields {
            if saved != asked {
                return Err(format!(
                    "{} was rendered with {name} {saved}, not {asked}. Use a new output directory.",
                    out.display()
                )
                .into());
            }
        }
        Ok(())
    }
}

pub struct NarrationStore {
    out: PathBuf,
    chunks: Vec<Chunk>,
    speed: f64,
    noise_limit_db: f32,
    manifest: Manifest,
}

/// The noise limit a run uses, and what that means for `narration.txt`.
struct SettledNoiseLimit {
    limit_db: f32,
    /// What `narration.txt` should hold. `None` leaves an older output
    /// directory's file without a limit.
    saved: Option<f32>,
    /// Whether to rewrite an existing `narration.txt` now. A new output
    /// directory saves its file with the first take instead.
    save_now: bool,
    note: Option<String>,
}

/// A given limit wins and is saved; otherwise the saved one is used. An output
/// directory saved before limits were gets the default without it being written,
/// because it may have been rendered with a limit that wasn't saved.
fn settle_noise_limit(saved: Option<&Manifest>, given: Option<f32>, out: &Path) -> SettledNoiseLimit {
    let path = out.join(MANIFEST);
    let Some(saved) = saved else {
        let limit_db = given.unwrap_or(MAX_FLOOR_DB);
        let note = given.map(|limit| format!("Noise limit: {limit} dB"));
        return SettledNoiseLimit { limit_db, saved: Some(limit_db), save_now: false, note };
    };
    match (saved.noise_limit_db, given) {
        (Some(old), Some(new)) if old != new => SettledNoiseLimit {
            limit_db: new,
            saved: Some(new),
            save_now: true,
            note: Some(format!("Noise limit: {new} dB, replacing {old} dB in {}", path.display())),
        },
        (None, Some(new)) => SettledNoiseLimit {
            limit_db: new,
            saved: Some(new),
            save_now: true,
            note: Some(format!("Noise limit: {new} dB, now saved in {}", path.display())),
        },
        (Some(old), _) => SettledNoiseLimit {
            limit_db: old,
            saved: Some(old),
            save_now: false,
            note: (given.is_some() || old != MAX_FLOOR_DB).then(|| format!("Noise limit: {old} dB")),
        },
        (None, None) => SettledNoiseLimit {
            limit_db: MAX_FLOOR_DB,
            saved: None,
            save_now: false,
            note: Some(format!(
                "Noise limit: {MAX_FLOOR_DB} dB. {} has no saved limit, so if this output directory \
                 was rendered with {}, give it once more to save it.",
                path.display(),
                crate::quality::NOISE_LIMIT_VAR,
            )),
        },
    }
}

/// Numbers the chunks and keys each by its text, voice and speed.
fn chunks_of(chapters: &[Vec<String>], voice: &str, speed: f64) -> Vec<Chunk> {
    // Normal speed keeps the original key, so older chunks stay valid.
    let speed_key = if speed == NORMAL_SPEED { String::new() } else { format!("speed={speed}\n") };
    let mut chunks = Vec::new();
    for (chapter, texts) in chapters.iter().enumerate() {
        for (position, text) in texts.iter().enumerate() {
            let key = format!("{CACHE_VERSION}\n{voice}\n{speed_key}{text}");
            chunks.push(Chunk {
                chapter,
                position,
                hash: format!("{:016x}", fnv(key.as_bytes(), 0)),
                text: text.clone(),
            });
        }
    }
    chunks
}

impl NarrationStore {
    /// Opens an output directory only to read what has been rendered there, in
    /// the voice and settings it saved, changing nothing.
    pub fn read(out: &Path, book_title: &str, chapters: &[Vec<String>]) -> Result<Self, Box<dyn Error>> {
        let manifest = read_manifest(out)?
            .ok_or_else(|| format!("{} has no {MANIFEST}, so nothing has been rendered there", out.display()))?;
        let book = book_title.replace('\n', " ");
        if manifest.book != book {
            return Err(format!("{} narrates {}, not {book}", out.display(), manifest.book).into());
        }
        Ok(Self {
            out: out.to_path_buf(),
            chunks: chunks_of(chapters, &manifest.voice, manifest.speed),
            speed: manifest.speed,
            noise_limit_db: manifest.noise_limit_db.unwrap_or(MAX_FLOOR_DB),
            manifest,
        })
    }

    pub fn open(
        out: &Path,
        book_title: &str,
        chapters: &[Vec<String>],
        voice: u64,
        settings: &RenderSettings,
        given_noise_limit: Option<f32>,
    ) -> Result<Self, Box<dyn Error>> {
        let saved = read_manifest(out)?;
        let speed = settings.speed.or(saved.as_ref().map(|saved| saved.speed)).unwrap_or(NORMAL_SPEED);
        let noise_limit = settle_noise_limit(saved.as_ref(), given_noise_limit, out);
        let chunks = chunks_of(chapters, &format!("{voice:x}"), speed);
        fs::create_dir_all(out.join("chunks"))?;
        let manifest = Manifest {
            book: book_title.replace('\n', " "),
            voice: format!("{voice:x}"),
            precision: settings.precision.clone(),
            speed,
            noise_limit_db: noise_limit.saved,
        };
        let store = Self { out: out.to_path_buf(), chunks, speed, noise_limit_db: noise_limit.limit_db, manifest };
        store.require_same(saved)?;
        if noise_limit.save_now {
            store.write_manifest()?;
        }
        if let Some(note) = noise_limit.note {
            eprintln!("{note}");
        }
        Ok(store)
    }

    /// Refuses anything other than what this folder has already rendered.
    fn require_same(&self, saved: Option<Manifest>) -> Result<(), Box<dyn Error>> {
        if let Some(saved) = saved {
            return saved.require(&self.manifest, &self.out);
        }
        if let Some(older) = self.older_manifest(&self.manifest)? {
            older.require(&self.manifest, &self.out)?;
            self.save_manifest()?;
        }
        Ok(())
    }

    /// Saves what this folder narrates, once it has audio to go with it.
    fn save_manifest(&self) -> Result<(), Box<dyn Error>> {
        if !self.out.join(MANIFEST).exists() {
            self.write_manifest()?;
        }
        Ok(())
    }

    fn write_manifest(&self) -> Result<(), Box<dyn Error>> {
        let path = self.out.join(MANIFEST);
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, self.manifest.render())?;
        fs::rename(&temporary, &path)?;
        Ok(())
    }

    /// Older versions saved nothing, so a folder with chunks can only be trusted
    /// if some of them belong to this narration. They always used the default
    /// precision unless overridden for testing, and normal speed.
    fn older_manifest(&self, wanted: &Manifest) -> Result<Option<Manifest>, Box<dyn Error>> {
        let has_audio = fs::read_dir(self.out.join("chunks"))?
            .filter_map(Result::ok)
            .any(|entry| entry.path().extension().is_some_and(|extension| extension == "wav"));
        if !has_audio {
            return Ok(None);
        }
        if wanted.speed != NORMAL_SPEED {
            return Err(format!(
                "{} was rendered at normal speed, not {}. Use a new output directory.",
                self.out.display(),
                wanted.speed
            )
            .into());
        }
        if !self.chunks.iter().any(|chunk| self.has_audio(chunk)) {
            return Err(format!(
                "{} has audio from a different voice or text. Use a new output directory.",
                self.out.display()
            )
            .into());
        }
        Ok(Some(Manifest { precision: DEFAULT_PRECISION.into(), speed: NORMAL_SPEED, ..wanted.clone() }))
    }

    /// Finds what `targets` name: chunk numbers, or words from the book. Each audio
    /// file appears once, in book order, with every chunk number that shares it.
    /// Words only match chunks with audio, as they're for fixing what's been heard.
    pub fn select(&self, targets: &[&str]) -> Result<Vec<Selected<'_>>, Box<dyn Error>> {
        let mut wanted: HashMap<&str, Vec<String>> = HashMap::new();
        for target in targets {
            if let Some(number) = parse_number(target) {
                let chunk = self.chunks.iter().find(|chunk| (chunk.chapter + 1, chunk.position + 1) == number);
                let chunk = chunk.ok_or_else(|| format!("no chunk {target} in this book"))?;
                wanted.entry(&chunk.hash).or_default();
            } else {
                for chunk in self.chunks_with_words(target)? {
                    let phrases = wanted.entry(&chunk.hash).or_default();
                    if !phrases.iter().any(|phrase| phrase == target) {
                        phrases.push((*target).to_string());
                    }
                }
            }
        }

        let mut selected: Vec<Selected> = Vec::new();
        for chunk in &self.chunks {
            if let Some(phrases) = wanted.remove(chunk.hash.as_str()) {
                let numbers = self.chunks.iter().filter(|other| other.hash == chunk.hash).map(Chunk::number).collect();
                selected.push(Selected { chunk, phrases, numbers });
            }
        }
        Ok(selected)
    }

    fn chunks_with_words(&self, words: &str) -> Result<Vec<&Chunk>, Box<dyn Error>> {
        let (with_audio, without): (Vec<&Chunk>, Vec<&Chunk>) = self
            .chunks
            .iter()
            .filter(|chunk| contains_phrase(&chunk.text, words))
            .partition(|chunk| self.has_audio(chunk));
        if !with_audio.is_empty() {
            return Ok(with_audio);
        }
        if without.is_empty() {
            let hint = if words.parse::<usize>().is_ok() { "; chunk numbers look like 3.12, see report.tsv" } else { "" };
            return Err(format!("nothing in the book matches \"{words}\"{hint}").into());
        }
        let numbers: Vec<String> = without.iter().map(|chunk| chunk.number()).collect();
        Err(format!(
            "\"{words}\" is only in chunks with no audio: {}. Name them by number to record them.",
            numbers.join(", ")
        )
        .into())
    }

    /// The speaking speed this folder narrates at.
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// The background level a take's pauses must stay under in this output directory.
    pub fn noise_limit_db(&self) -> f32 {
        self.noise_limit_db
    }

    /// Every chunk in the book, whichever chapters have been rendered.
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    pub fn audio_path(&self, chunk: &Chunk) -> PathBuf {
        self.out.join("chunks").join(format!("{}.wav", chunk.hash))
    }

    pub fn has_audio(&self, chunk: &Chunk) -> bool {
        self.audio_path(chunk).exists()
    }

    /// Chunks in the chapters that still need audio, once for each audio file.
    pub fn pending(&self, chapters: &[usize]) -> Vec<&Chunk> {
        let mut seen = HashSet::new();
        self.chunks
            .iter()
            .filter(|chunk| chapters.contains(&chunk.chapter))
            .filter(|chunk| !self.has_audio(chunk) && seen.insert(&chunk.hash))
            .collect()
    }

    /// How many chunks in the whole book have no audio.
    pub fn missing_count(&self) -> usize {
        self.chunks.iter().filter(|chunk| !self.has_audio(chunk)).count()
    }

    /// Chapters with audio for every chunk, in book order.
    pub fn complete_chapters(&self) -> Vec<usize> {
        let mut complete: Vec<usize> = self.chunks.iter().map(|chunk| chunk.chapter).collect();
        complete.dedup();
        complete.retain(|&chapter| {
            self.chunks.iter().filter(|chunk| chunk.chapter == chapter).all(|chunk| self.has_audio(chunk))
        });
        complete
    }

    /// Chapters with audio for some of their chunks but not all, in book order.
    pub fn partly_rendered_chapters(&self) -> Vec<usize> {
        let complete = self.complete_chapters();
        let mut partly: Vec<usize> = self
            .chunks
            .iter()
            .filter(|chunk| self.has_audio(chunk) && !complete.contains(&chunk.chapter))
            .map(|chunk| chunk.chapter)
            .collect();
        partly.dedup();
        partly
    }

    /// How many chapters can't be in the M4B yet, because a chunk has no audio.
    pub fn incomplete_chapter_count(&self) -> usize {
        let mut incomplete: Vec<usize> =
            self.chunks.iter().filter(|chunk| !self.has_audio(chunk)).map(|chunk| chunk.chapter).collect();
        incomplete.dedup();
        incomplete.len()
    }

    /// Saves a chunk's audio and check file, replacing the audio in one step.
    pub fn store(&self, chunk: &Chunk, audio: &AudioSamples, transcript: &str, takes: u64) -> Result<(), Box<dyn Error>> {
        self.save_manifest()?;
        let path = self.audio_path(chunk);
        let transcript = transcript.replace(['\n', '\t'], " ");
        fs::write(path.with_extension("check"), format!("{takes}\n{transcript}"))?;
        let temporary = path.with_extension("tmp");
        audio.save_wav(&temporary)?;
        fs::rename(&temporary, path)?;
        Ok(())
    }

    /// Reads the stored check for a chunk that has audio.
    pub fn check(&self, chunk: &Chunk) -> Result<Check, Box<dyn Error>> {
        let sidecar = fs::read_to_string(self.audio_path(chunk).with_extension("check"))?;
        let (takes, transcript) = sidecar.split_once('\n').ok_or("malformed chunk check file")?;
        Ok(Check { takes: takes.parse()?, transcript: transcript.to_string() })
    }

    /// Records more tried seeds without changing the audio.
    pub fn record_tried(&self, chunk: &Chunk, takes: u64) -> Result<(), Box<dyn Error>> {
        let transcript = self.check(chunk)?.transcript;
        let path = self.audio_path(chunk).with_extension("check");
        let temporary = path.with_extension("check.tmp");
        fs::write(&temporary, format!("{takes}\n{transcript}"))?;
        fs::rename(&temporary, path)?;
        Ok(())
    }

    /// Moves a chunk's audio and check file to `replaced/`, ready for a new take.
    pub fn retire(&self, chunk: &Chunk) -> Result<(), Box<dyn Error>> {
        let takes = self.check(chunk)?.takes;
        let replaced = self.out.join("replaced");
        fs::create_dir_all(&replaced)?;
        let path = self.audio_path(chunk);
        let old = replaced.join(format!("{}-{takes}.wav", chunk.hash));
        fs::rename(&path, &old)?;
        fs::rename(path.with_extension("check"), old.with_extension("check"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use any_tts::config::DType;
    use std::fs;

    /// A directory that's removed when the test ends.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!("audio-berry-store-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn runtime(device: DeviceSelection, dtype: DType) -> OmniVoiceRuntimeChoice {
        OmniVoiceRuntimeChoice { device, dtype }
    }

    fn settings() -> RenderSettings {
        RenderSettings { precision: "f16".into(), speed: None }
    }

    fn chapters(texts: &[&[&str]]) -> Vec<Vec<String>> {
        texts.iter().map(|chapter| chapter.iter().map(|text| text.to_string()).collect()).collect()
    }

    fn open(dir: &TempDir, texts: &[&[&str]]) -> NarrationStore {
        NarrationStore::open(&dir.0, "A Book", &chapters(texts), 0xabc, &settings(), None).unwrap()
    }

    fn tone() -> AudioSamples {
        AudioSamples::new(vec![0.1; 240], 24_000)
    }

    /// Opens a folder and stores a take, which is when it saves its settings.
    fn render_as(dir: &TempDir, title: &str, voice: u64, precision: &str) {
        let store = open_as(dir, title, voice, precision).unwrap();
        store.store(&store.chunks()[0], &tone(), "a", 1).unwrap();
    }

    fn render_at(dir: &TempDir, speed: Option<f64>) {
        let store = open_at(dir, speed).unwrap();
        store.store(&store.chunks()[0], &tone(), "a", 1).unwrap();
    }

    #[test]
    fn opening_a_folder_saves_nothing_until_a_take_is_stored() {
        let dir = TempDir::new("nothing-yet");
        open_at(&dir, None).unwrap();
        assert!(!dir.0.join(MANIFEST).exists());
        assert!(open_at(&dir, Some(0.9)).is_ok(), "nothing has been rendered yet");
        render_at(&dir, Some(0.9));
        assert!(open_at(&dir, Some(1.1)).is_err());
    }

    #[test]
    fn speed_on_an_older_folder_is_refused_by_name() {
        let dir = TempDir::new("older-speed-name");
        fs::create_dir_all(dir.0.join("chunks")).unwrap();
        fs::write(dir.0.join("chunks").join("026926e6f4d06f3e.wav"), "audio").unwrap();
        let error = open_at(&dir, Some(0.9)).err().unwrap().to_string();
        assert!(error.contains("at normal speed"), "{error}");
    }

    #[test]
    fn partly_rendered_chapters_are_named_and_missing_ones_counted() {
        let dir = TempDir::new("partial");
        let store = open(&dir, &[&["a", "b"], &["c"], &[], &["d"], &["e"]]);
        store.store(&store.chunks()[0], &tone(), "a", 1).unwrap();
        store.store(&store.chunks()[3], &tone(), "d", 1).unwrap();
        assert_eq!(store.partly_rendered_chapters(), [0]);
        assert_eq!(store.incomplete_chapter_count(), 3);
    }

    fn open_as(dir: &TempDir, title: &str, voice: u64, precision: &str) -> Result<NarrationStore, Box<dyn Error>> {
        let settings = RenderSettings { precision: precision.into(), speed: None };
        NarrationStore::open(&dir.0, title, &chapters(&[&["a"]]), voice, &settings, None)
    }

    fn numbers(selected: &[Selected]) -> Vec<String> {
        selected.iter().map(|item| item.chunk.number()).collect()
    }

    #[test]
    fn a_number_selects_that_chunk_even_with_no_audio() {
        let dir = TempDir::new("select-number");
        let store = open(&dir, &[&["one"], &["two", "three"]]);
        let selected = store.select(&["2.2"]).unwrap();
        assert_eq!(numbers(&selected), ["2.2"]);
        assert!(selected[0].phrases.is_empty());
    }

    #[test]
    fn selecting_never_adds_chunks_that_were_not_named() {
        let dir = TempDir::new("select-only");
        let store = open(&dir, &[&["one"], &["two"], &["three"]]);
        store.store(&store.chunks()[0], &tone(), "one", 1).unwrap();
        assert_eq!(numbers(&store.select(&["1.1"]).unwrap()), ["1.1"]);
    }

    #[test]
    fn a_number_outside_the_book_is_an_error() {
        let dir = TempDir::new("select-range");
        let store = open(&dir, &[&["one"]]);
        for target in ["1.2", "2.1", "0.1", "1.0"] {
            let error = store.select(&[target]).err().unwrap().to_string();
            assert!(error.contains(target), "{error}");
        }
    }

    #[test]
    fn words_select_only_chunks_with_audio() {
        let dir = TempDir::new("select-words");
        let store = open(&dir, &[&["the white rabbit"], &["a white hat"]]);
        store.store(&store.chunks()[0], &tone(), "x", 1).unwrap();
        let selected = store.select(&["white"]).unwrap();
        assert_eq!(numbers(&selected), ["1.1"]);
        assert_eq!(selected[0].phrases, ["white"]);
    }

    #[test]
    fn words_matching_only_missing_chunks_ask_for_a_number() {
        let dir = TempDir::new("select-missing");
        let store = open(&dir, &[&["the white rabbit"]]);
        let error = store.select(&["white"]).err().unwrap().to_string();
        assert!(error.contains("1.1"), "{error}");
    }

    #[test]
    fn a_bare_number_that_matches_nothing_shows_the_new_form() {
        let dir = TempDir::new("select-bare");
        let store = open(&dir, &[&["one"]]);
        let error = store.select(&["32"]).err().unwrap().to_string();
        assert!(error.contains("3.12"), "{error}");
    }

    #[test]
    fn shared_audio_is_selected_once_with_every_number() {
        let dir = TempDir::new("select-shared");
        let store = open(&dir, &[&["same line", "other"], &["same line"]]);
        store.store(&store.chunks()[0], &tone(), "same line", 1).unwrap();
        let selected = store.select(&["same line", "2.1"]).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].numbers, ["1.1", "2.1"]);
        assert_eq!(selected[0].phrases, ["same line"]);
    }

    #[test]
    fn selections_come_back_in_book_order() {
        let dir = TempDir::new("select-order");
        let store = open(&dir, &[&["a"], &["b"]]);
        assert_eq!(numbers(&store.select(&["2.1", "1.1"]).unwrap()), ["1.1", "2.1"]);
    }

    fn open_at(dir: &TempDir, speed: Option<f64>) -> Result<NarrationStore, Box<dyn Error>> {
        let settings = RenderSettings { speed, ..settings() };
        NarrationStore::open(&dir.0, "A Book", &chapters(&[&["Hello there."]]), 0xabc, &settings, None)
    }

    #[test]
    fn a_new_speed_gets_its_own_cache_key() {
        // Worked out independently with FNV-1a of "3\nabc\nspeed=0.9\nHello there.".
        let dir = TempDir::new("speed-key");
        let store = open_at(&dir, Some(0.9)).unwrap();
        assert_eq!(store.audio_path(&store.chunks()[0]).file_name().unwrap(), "6c7ca5b9cb1c27dd.wav");
    }

    #[test]
    fn normal_speed_keeps_the_existing_cache_key() {
        let dir = TempDir::new("speed-normal");
        let store = open_at(&dir, Some(1.0)).unwrap();
        assert_eq!(store.audio_path(&store.chunks()[0]).file_name().unwrap(), "026926e6f4d06f3e.wav");
    }

    #[test]
    fn leaving_out_the_speed_uses_the_saved_one() {
        let dir = TempDir::new("speed-saved");
        render_at(&dir, Some(0.9));
        let store = open_at(&dir, None).unwrap();
        assert_eq!(store.speed(), 0.9);
        assert_eq!(store.audio_path(&store.chunks()[0]).file_name().unwrap(), "6c7ca5b9cb1c27dd.wav");
    }

    #[test]
    fn a_different_speed_is_refused_by_name() {
        let dir = TempDir::new("speed-refused");
        render_at(&dir, Some(0.9));
        let error = open_at(&dir, Some(1.1)).err().unwrap().to_string();
        assert!(error.contains("with speed"), "{error}");
    }

    #[test]
    fn an_older_folder_is_normal_speed() {
        let dir = TempDir::new("speed-older");
        fs::create_dir_all(dir.0.join("chunks")).unwrap();
        fs::write(dir.0.join("chunks").join("026926e6f4d06f3e.wav"), "audio").unwrap();
        assert_eq!(open_at(&dir, None).unwrap().speed(), 1.0);
        assert!(open_at(&dir, Some(0.9)).is_err());
    }

    #[test]
    fn a_folder_from_before_speed_was_saved_is_normal_speed() {
        let dir = TempDir::new("speed-manifest");
        fs::write(dir.0.join(MANIFEST), "book=A Book\nvoice=abc\nprecision=f16\n").unwrap();
        assert_eq!(open_at(&dir, None).unwrap().speed(), 1.0);
    }

    #[test]
    fn a_folder_reopens_with_the_same_voice_and_settings() {
        let dir = TempDir::new("same");
        render_as(&dir, "A Book", 1, "f16");
        assert!(open_as(&dir, "A Book", 1, "f16").is_ok());
    }

    #[test]
    fn a_different_voice_is_refused_by_name() {
        let dir = TempDir::new("voice");
        render_as(&dir, "A Book", 1, "f16");
        let error = open_as(&dir, "A Book", 2, "f16").err().unwrap().to_string();
        assert!(error.contains("with voice"), "{error}");
        assert!(error.contains("new output directory"), "{error}");
    }

    #[test]
    fn a_different_precision_is_refused_by_name() {
        let dir = TempDir::new("precision");
        render_as(&dir, "A Book", 1, "f16");
        let error = open_as(&dir, "A Book", 1, "f32").err().unwrap().to_string();
        assert!(error.contains("with precision"), "{error}");
    }

    #[test]
    fn a_different_book_is_refused_by_name() {
        let dir = TempDir::new("book");
        render_as(&dir, "A Book", 1, "f16");
        let error = open_as(&dir, "Another Book", 1, "f16").err().unwrap().to_string();
        assert!(error.contains("Another Book"), "{error}");
    }

    /// Makes a folder as older versions left it: chunks, but no saved settings.
    fn older_folder(dir: &TempDir, voice: u64) {
        let scratch = TempDir::new(&format!("older-of-{}", dir.0.display()).replace('/', "-"));
        let store = open_as(&scratch, "A Book", voice, "f16").unwrap();
        let name = store.audio_path(&store.chunks()[0]);
        fs::create_dir_all(dir.0.join("chunks")).unwrap();
        fs::write(dir.0.join("chunks").join(name.file_name().unwrap()), "audio").unwrap();
    }

    #[test]
    fn an_older_folder_with_matching_chunks_adopts_the_voice() {
        let dir = TempDir::new("adopt");
        older_folder(&dir, 1);
        open_as(&dir, "A Book", 1, "f16").unwrap();
        assert!(open_as(&dir, "A Book", 2, "f16").is_err(), "the adopted voice should now be saved");
    }

    #[test]
    fn an_older_folder_with_no_matching_chunks_is_refused() {
        let dir = TempDir::new("mismatch");
        older_folder(&dir, 1);
        let error = open_as(&dir, "A Book", 2, "f16").err().unwrap().to_string();
        assert!(error.contains("new output directory"), "{error}");
    }

    #[test]
    fn an_older_folder_is_assumed_to_be_the_default_precision() {
        let dir = TempDir::new("older-precision");
        older_folder(&dir, 1);
        assert!(open_as(&dir, "A Book", 1, "f32").is_err());
    }

    #[test]
    fn precision_comes_from_the_environment_value_or_defaults() {
        let cuda = runtime(DeviceSelection::Cuda(0), DType::F16);
        assert_eq!(RenderSettings::from_dtype(None, cuda).precision, "f16");
        assert_eq!(RenderSettings::from_dtype(Some(" BF16 "), cuda).precision, "bf16");
        assert_eq!(RenderSettings::from_dtype(Some("nonsense"), cuda).precision, "f16");
    }

    /// The expected values are the rules in OmniVoiceModel::load.
    #[test]
    fn precision_is_what_the_model_runs_on_metal_and_cpu() {
        for device in [DeviceSelection::Metal(0), DeviceSelection::Cpu] {
            let choice = runtime(device, DType::F32);
            assert_eq!(RenderSettings::from_dtype(None, choice).precision, "f32");
            assert_eq!(RenderSettings::from_dtype(Some("f16"), choice).precision, "f16");
            assert_eq!(RenderSettings::from_dtype(Some("bf16"), choice).precision, "f32");
            assert_eq!(RenderSettings::from_dtype(Some("nonsense"), choice).precision, "f32");
        }
    }

    fn open_store(dir: &TempDir, noise_limit: Option<f32>) -> Result<NarrationStore, Box<dyn Error>> {
        NarrationStore::open(&dir.0, "A Book", &chapters(&[&["Hello there."]]), 0xabc, &settings(), noise_limit)
    }

    fn open_with_limit(dir: &TempDir, noise_limit: Option<f32>) -> NarrationStore {
        open_store(dir, noise_limit).unwrap()
    }

    fn render_with_limit(dir: &TempDir, noise_limit: Option<f32>) {
        let store = open_with_limit(dir, noise_limit);
        store.store(&store.chunks()[0], &tone(), "a", 1).unwrap();
    }

    fn saved_limit(dir: &TempDir) -> Option<String> {
        let text = fs::read_to_string(dir.0.join(MANIFEST)).unwrap();
        text.lines().find_map(|line| line.strip_prefix("noise_limit_db=")).map(str::to_string)
    }

    #[test]
    fn the_noise_limit_is_saved_with_the_first_take() {
        let dir = TempDir::new("limit-first");
        let store = open_with_limit(&dir, Some(-50.0));
        assert_eq!(store.noise_limit_db(), -50.0);
        assert!(!dir.0.join(MANIFEST).exists());
        store.store(&store.chunks()[0], &tone(), "a", 1).unwrap();
        assert_eq!(saved_limit(&dir).as_deref(), Some("-50"));
    }

    #[test]
    fn the_default_noise_limit_is_saved_when_none_is_given() {
        let dir = TempDir::new("limit-default");
        render_with_limit(&dir, None);
        assert_eq!(saved_limit(&dir).as_deref(), Some("-55"));
    }

    #[test]
    fn the_saved_noise_limit_is_used_when_none_is_given() {
        let dir = TempDir::new("limit-reused");
        render_with_limit(&dir, Some(-50.0));
        assert_eq!(open_with_limit(&dir, None).noise_limit_db(), -50.0);
        assert_eq!(saved_limit(&dir).as_deref(), Some("-50"));
    }

    #[test]
    fn a_new_noise_limit_replaces_the_saved_one_and_keeps_the_audio() {
        let dir = TempDir::new("limit-changed");
        render_with_limit(&dir, Some(-50.0));
        let before = fs::read_to_string(dir.0.join(MANIFEST)).unwrap();
        let store = open_with_limit(&dir, Some(-48.5));
        assert_eq!(store.noise_limit_db(), -48.5);
        assert!(store.has_audio(&store.chunks()[0]), "the cached take should still be found");
        assert_eq!(store.audio_path(&store.chunks()[0]).file_name().unwrap(), "026926e6f4d06f3e.wav");
        assert_eq!(saved_limit(&dir).as_deref(), Some("-48.5"));
        let after = fs::read_to_string(dir.0.join(MANIFEST)).unwrap();
        assert_eq!(before.replace("noise_limit_db=-50", "noise_limit_db=-48.5"), after);
        assert_eq!(open_with_limit(&dir, None).noise_limit_db(), -48.5);
    }

    /// A `narration.txt` as saved before the noise limit was.
    const OLDER_MANIFEST: &str = "book=A Book\nvoice=abc\nprecision=f16\nspeed=1\n";

    fn folder_without_a_saved_limit(name: &str) -> TempDir {
        let dir = TempDir::new(name);
        fs::create_dir_all(dir.0.join("chunks")).unwrap();
        fs::write(dir.0.join(MANIFEST), OLDER_MANIFEST).unwrap();
        dir
    }

    #[test]
    fn an_older_folder_uses_the_default_without_saving_it() {
        // It may have been rendered with a limit that wasn't saved, so writing the
        // default would quietly judge its takes against the wrong one.
        let dir = folder_without_a_saved_limit("limit-older");
        assert_eq!(open_with_limit(&dir, None).noise_limit_db(), MAX_FLOOR_DB);
        assert_eq!(fs::read_to_string(dir.0.join(MANIFEST)).unwrap(), OLDER_MANIFEST);
        let settled = settle_noise_limit(Some(&Manifest::parse(OLDER_MANIFEST).unwrap()), None, &dir.0);
        assert!(settled.note.unwrap().contains("give it once"));
    }

    #[test]
    fn an_older_folder_saves_a_limit_when_one_is_given() {
        let dir = folder_without_a_saved_limit("limit-older-new");
        assert_eq!(open_with_limit(&dir, Some(-50.0)).noise_limit_db(), -50.0);
        assert_eq!(saved_limit(&dir).as_deref(), Some("-50"));
        assert_eq!(open_with_limit(&dir, None).noise_limit_db(), -50.0);
        let settled = settle_noise_limit(Some(&Manifest::parse(OLDER_MANIFEST).unwrap()), Some(-50.0), &dir.0);
        let note = settled.note.unwrap();
        assert!(note.contains("saved in") && !note.contains("replacing"), "{note}");
    }

    #[test]
    fn the_note_names_the_limit_being_replaced() {
        let saved = Manifest::parse(&format!("{OLDER_MANIFEST}noise_limit_db=-50\n")).unwrap();
        let settled = settle_noise_limit(Some(&saved), Some(-48.0), Path::new("out"));
        assert!(settled.note.unwrap().contains("replacing -50 dB"));
        let unchanged = settle_noise_limit(Some(&saved), Some(-50.0), Path::new("out"));
        assert!(!unchanged.save_now);
    }

    #[test]
    fn a_bad_saved_noise_limit_is_an_error_that_says_how_to_fix_it() {
        let dir = folder_without_a_saved_limit("limit-bad");
        fs::write(dir.0.join(MANIFEST), format!("{OLDER_MANIFEST}noise_limit_db=loud\n")).unwrap();
        let error = open_store(&dir, Some(-50.0)).err().unwrap().to_string();
        assert!(error.contains("noise_limit_db=loud") && error.contains("delete that line"), "{error}");
    }

    #[test]
    fn a_stored_take_can_be_read_back() {
        let dir = TempDir::new("roundtrip");
        let store = open(&dir, &[&["a"]]);
        let chunk = &store.chunks()[0];
        assert!(!store.has_audio(chunk));
        store.store(chunk, &tone(), "heard\tthis\nline", 2).unwrap();
        assert!(store.has_audio(chunk));
        assert_eq!(store.check(chunk).unwrap(), Check { takes: 2, transcript: "heard this line".into() });
    }

    #[test]
    fn reading_a_folder_finds_its_takes_at_its_saved_speed() {
        let dir = TempDir::new("read");
        render_at(&dir, Some(0.9));
        let store = NarrationStore::read(&dir.0, "A Book", &chapters(&[&["Hello there."]])).unwrap();
        assert_eq!(store.speed(), 0.9);
        assert_eq!(store.check(&store.chunks()[0]).unwrap(), Check { takes: 1, transcript: "a".into() });
        assert!(NarrationStore::read(&dir.0, "Another Book", &chapters(&[&["Hello there."]])).is_err());
    }

    #[test]
    fn reading_a_folder_with_nothing_rendered_is_an_error_that_changes_nothing() {
        let dir = TempDir::new("read-empty");
        let error = NarrationStore::read(&dir.0, "A Book", &chapters(&[&["a"]])).err().unwrap().to_string();
        assert!(error.ends_with("has no narration.txt, so nothing has been rendered there"), "{error}");
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 0);
    }

    #[test]
    fn a_malformed_check_file_is_an_error() {
        let dir = TempDir::new("malformed");
        let store = open(&dir, &[&["a"]]);
        let chunk = &store.chunks()[0];
        store.store(chunk, &tone(), "heard", 1).unwrap();
        fs::write(store.audio_path(chunk).with_extension("check"), "no newline").unwrap();
        assert!(store.check(chunk).is_err());
    }

    #[test]
    fn recording_tried_seeds_keeps_the_transcript() {
        let dir = TempDir::new("tried");
        let store = open(&dir, &[&["a"]]);
        let chunk = &store.chunks()[0];
        store.store(chunk, &tone(), "heard", 1).unwrap();
        store.record_tried(chunk, 7).unwrap();
        assert_eq!(store.check(chunk).unwrap(), Check { takes: 7, transcript: "heard".into() });
    }

    #[test]
    fn retiring_a_take_keeps_it_in_replaced() {
        let dir = TempDir::new("retire");
        let store = open(&dir, &[&["a"]]);
        let chunk = &store.chunks()[0];
        store.store(chunk, &tone(), "heard", 3).unwrap();
        store.retire(chunk).unwrap();
        assert!(!store.has_audio(chunk));
        let name = store.audio_path(chunk).file_stem().unwrap().to_str().unwrap().to_string();
        assert!(dir.0.join("replaced").join(format!("{name}-3.wav")).exists());
        assert_eq!(fs::read_to_string(dir.0.join("replaced").join(format!("{name}-3.check"))).unwrap(), "3\nheard");
    }

    #[test]
    fn pending_lists_only_the_chapters_asked_for_and_shared_audio_once() {
        let dir = TempDir::new("pending");
        let store = open(&dir, &[&["a", "b", "a"], &["c"]]);
        store.store(&store.chunks()[1], &tone(), "b", 1).unwrap();
        let numbers: Vec<String> = store.pending(&[0]).iter().map(|chunk| chunk.number()).collect();
        assert_eq!(numbers, ["1.1"]);
    }

    #[test]
    fn only_chapters_with_all_their_audio_are_complete() {
        let dir = TempDir::new("complete");
        let store = open(&dir, &[&["a", "b"], &["c"], &[], &["d"]]);
        store.store(&store.chunks()[0], &tone(), "a", 1).unwrap();
        store.store(&store.chunks()[3], &tone(), "d", 1).unwrap();
        assert_eq!(store.complete_chapters(), [3]);
        assert_eq!(store.missing_count(), 2);
    }

    #[test]
    fn chunk_numbers_count_from_one_within_each_chapter() {
        let dir = TempDir::new("numbers");
        let store = open(&dir, &[&["a", "b"], &["c"]]);
        let numbers: Vec<String> = store.chunks().iter().map(Chunk::number).collect();
        assert_eq!(numbers, ["1.1", "1.2", "2.1"]);
    }

    #[test]
    fn chunk_files_keep_the_existing_cache_key() {
        // Worked out independently with FNV-1a of "3\nabc\nHello there.".
        let dir = TempDir::new("key");
        let store = open(&dir, &[&["Hello there."]]);
        let path = store.audio_path(&store.chunks()[0]);
        assert_eq!(path.file_name().unwrap(), "026926e6f4d06f3e.wav");
    }
}
