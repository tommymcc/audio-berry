//! An output directory: one narration of one book, in one voice, with one set
//! of render settings. Owns chunk numbering, the cache key and the files.

use crate::checker;
use crate::narrator::PRECISION_VAR;
use crate::quality::{DEFAULT_NOISE_LIMIT_DB, valid_noise_limit};
use crate::accuracy::contains_phrase;
use crate::files::read_if_present;
use crate::pitch::{PITCH_RANGE_LIMITS, valid_pitch_range};
use crate::takes::{TAKES_RANGE, TakePolicy};
use crate::text::ChunkLimit;
use any_tts::{AudioSamples, DeviceSelection};
use any_tts::models::omnivoice::{OmniVoiceRuntimeChoice, preferred_runtime_choice};
use std::collections::{HashMap, HashSet};
use color_eyre::eyre::{Result, WrapErr, bail, eyre};
use fs_err as fs;
use std::path::{Path, PathBuf};

const CACHE_VERSION: &str = "3";
const DEFAULT_PRECISION: &str = "f16";
const NORMAL_SPEED: f64 = 1.0;
const MANIFEST: &str = "narration.txt";
/// The key `narration.txt` saves the chunk limit's weight under.
const CHUNK_WEIGHT: &str = "chunk_weight";

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
    /// The settings for `speed` and the precision the model will load at, which
    /// it reads from `PRECISION_VAR`.
    pub fn from_env(speed: Option<f64>) -> Self {
        let dtype = std::env::var(PRECISION_VAR).ok();
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
    /// The model that heard the transcript, or `None` for a check file written
    /// before it was recorded, when Whisper heard every take.
    pub checker: Option<String>,
}

impl Check {
    /// The check file's contents: the takes tried, the transcript, and the
    /// checker if known, a line each.
    fn to_saved(&self) -> String {
        let Check { takes, transcript, checker } = self;
        match checker {
            Some(checker) => format!("{takes}\n{transcript}\n{checker}"),
            None => format!("{takes}\n{transcript}"),
        }
    }

    /// Reads what `to_saved` wrote, or a check file from before the checker was
    /// recorded, which has only the first two lines.
    fn from_saved(text: &str) -> Result<Check> {
        let mut lines = text.splitn(3, '\n');
        let (Some(takes), Some(transcript)) = (lines.next(), lines.next()) else {
            bail!("malformed chunk check file");
        };
        Ok(Check { takes: takes.parse()?, transcript: transcript.to_string(), checker: lines.next().map(String::from) })
    }
}

fn read_manifest(out: &Path) -> Result<Option<Manifest>> {
    read_if_present(&out.join(MANIFEST))?.map(|text| Manifest::parse(&text)).transpose()
}

/// A keep setting, saved in `narration.txt` under `KEY`. A run uses the one
/// given, else the one saved, else `DEFAULT`, and saves a given one.
pub trait KeepSetting {
    /// A value it can be given.
    type Value: Copy + PartialEq;
    /// What `KeepRules` holds: `Value` for a setting with a default, or
    /// `Option<Value>` for one without, where `None` is no value.
    type Rule: Copy + PartialEq + From<Self::Value> + Into<Option<Self::Value>>;
    /// Its key in `narration.txt`.
    const KEY: &'static str;
    /// What a run uses with none given or saved.
    const DEFAULT: Self::Rule;
    /// The values `narration.txt` may hold, as the error for a bad line names them.
    fn valid_range_text() -> String;
    /// Reads what `to_saved` wrote, if it's valid.
    fn from_saved(text: &str) -> Option<Self::Value>;
    /// How `narration.txt` saves `value`.
    fn to_saved(value: Self::Value) -> String;
    /// How a note names `value`.
    fn display(value: Self::Value) -> String;
    /// The note saying a run uses the value named `displayed`.
    fn using_note(displayed: &str) -> String;
}

/// The noise limit, in dB.
pub struct NoiseLimit;

impl KeepSetting for NoiseLimit {
    type Value = f32;
    type Rule = f32;
    const KEY: &'static str = "noise_limit_db";
    const DEFAULT: f32 = DEFAULT_NOISE_LIMIT_DB;
    fn valid_range_text() -> String {
        "a level from -100 to 0 dB".into()
    }
    fn from_saved(text: &str) -> Option<f32> {
        valid_noise_limit(text)
    }
    fn to_saved(limit: f32) -> String {
        limit.to_string()
    }
    fn display(limit: f32) -> String {
        format!("{limit} dB")
    }
    fn using_note(displayed: &str) -> String {
        format!("Noise limit: {displayed}")
    }
}

/// The take policy.
pub struct TakePolicySetting;

impl KeepSetting for TakePolicySetting {
    type Value = TakePolicy;
    type Rule = TakePolicy;
    const KEY: &'static str = "takes";
    const DEFAULT: TakePolicy = TakePolicy::FirstPassing;
    fn valid_range_text() -> String {
        format!("first-passing or best-of-{} to best-of-{}", TAKES_RANGE.start(), TAKES_RANGE.end())
    }
    fn from_saved(text: &str) -> Option<TakePolicy> {
        TakePolicy::from_saved(text)
    }
    fn to_saved(policy: TakePolicy) -> String {
        policy.saved()
    }
    fn display(policy: TakePolicy) -> String {
        policy.to_string()
    }
    fn using_note(displayed: &str) -> String {
        format!("Takes: keeping {displayed}")
    }
}

/// The pitch range target, in semitones, which has no default.
pub struct PitchRangeTarget;

impl KeepSetting for PitchRangeTarget {
    type Value = f32;
    type Rule = Option<f32>;
    const KEY: &'static str = "pitch_range_st";
    const DEFAULT: Option<f32> = None;
    fn valid_range_text() -> String {
        format!("a range from {} to {} semitones", PITCH_RANGE_LIMITS.start(), PITCH_RANGE_LIMITS.end())
    }
    fn from_saved(text: &str) -> Option<f32> {
        valid_pitch_range(text)
    }
    fn to_saved(range: f32) -> String {
        range.to_string()
    }
    fn display(range: f32) -> String {
        range.to_string()
    }
    fn using_note(displayed: &str) -> String {
        format!("Pitch range: aiming for {displayed} semitones")
    }
}

/// The keep settings given on the command line, each `None` when not given, so
/// the output directory's saved one is used. `settle` turns them into the
/// `KeepRules` a run follows.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KeepSettings {
    /// See `NoiseLimit`.
    pub noise_limit_db: Option<f32>,
    /// See `TakePolicySetting`.
    pub takes: Option<TakePolicy>,
    /// See `PitchRangeTarget`.
    pub pitch_range_target: Option<f32>,
}

/// The keep settings a run follows, settled from those given and those saved,
/// with defaults filled in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeepRules {
    /// The noise limit takes must pass, in dB.
    pub noise_limit_db: f32,
    /// The take policy.
    pub takes: TakePolicy,
    /// The pitch range, in semitones, that takes which pass are ranked by
    /// nearness to, if any.
    pub pitch_range_target: Option<f32>,
}

impl KeepRules {
    /// Reads each setting from its line in `narration.txt`, which `line` finds,
    /// with the default for one that's missing.
    fn parse(line: impl Fn(&str) -> Option<String>) -> Result<Self> {
        Ok(Self {
            noise_limit_db: parse_setting::<NoiseLimit>(line(NoiseLimit::KEY))?,
            takes: parse_setting::<TakePolicySetting>(line(TakePolicySetting::KEY))?,
            pitch_range_target: parse_setting::<PitchRangeTarget>(line(PitchRangeTarget::KEY))?,
        })
    }

    /// Appends the lines `narration.txt` saves the settings as to `text`.
    fn render(&self, text: &mut String) {
        render_setting::<NoiseLimit>(self.noise_limit_db, text);
        render_setting::<TakePolicySetting>(self.takes, text);
        render_setting::<PitchRangeTarget>(self.pitch_range_target, text);
    }

    /// Settles each setting from the one given and the one `saved` in the
    /// `narration.txt` at `path`, if there is one.
    fn settle(saved: Option<&Self>, given: KeepSettings, path: &Path, settled: &mut Settled) -> Self {
        Self {
            noise_limit_db: settled.setting::<NoiseLimit>(
                saved.map(|saved| saved.noise_limit_db),
                given.noise_limit_db,
                path,
            ),
            takes: settled.setting::<TakePolicySetting>(saved.map(|saved| saved.takes), given.takes, path),
            pitch_range_target: settled.setting::<PitchRangeTarget>(
                saved.map(|saved| saved.pitch_range_target),
                given.pitch_range_target,
                path,
            ),
        }
    }
}

/// The rule a setting's line in `narration.txt` holds, if any, or the default.
fn parse_setting<S: KeepSetting>(line: Option<String>) -> Result<S::Rule> {
    let Some(text) = line else { return Ok(S::DEFAULT) };
    let value = S::from_saved(&text).ok_or_else(|| {
        eyre!("{MANIFEST} has {}={text}, which isn't {}. Correct or delete that line", S::KEY, S::valid_range_text())
    })?;
    Ok(value.into())
}

/// Appends a setting's line to `text`, unless it has no value.
fn render_setting<S: KeepSetting>(rule: S::Rule, text: &mut String) {
    if let Some(value) = rule.into() {
        text.push_str(&format!("{}={}\n", S::KEY, S::to_saved(value)));
    }
}

/// What settling the keep settings found to save and print.
#[derive(Default)]
struct Settled {
    /// Whether to rewrite an existing `narration.txt` now. A new output
    /// directory saves its file with the first take instead.
    save_now: bool,
    /// What the run will use.
    notes: Vec<String>,
}

impl Settled {
    /// A given setting wins, and is saved if it differs from the one `saved` in
    /// the `narration.txt` at `path`; otherwise the saved one is used, or the
    /// default with none saved. Notes a value given, or saved and not the default.
    fn setting<S: KeepSetting>(&mut self, saved: Option<S::Rule>, given: Option<S::Value>, path: &Path) -> S::Rule {
        let note = |rule: S::Rule| rule.into().map(|value| S::using_note(&S::display(value)));
        let given = given.map(S::Rule::from);
        let (rule, note) = match (saved, given) {
            (None, given) => (given.unwrap_or(S::DEFAULT), given.and_then(note)),
            (Some(old), Some(new)) if old != new => {
                self.save_now = true;
                let old: Option<S::Value> = old.into();
                let change = old.map_or("now saved".into(), |old| format!("replacing {}", S::display(old)));
                (new, note(new).map(|note| format!("{note}, {change} in {}", path.display())))
            }
            (Some(old), given) => (old, (given.is_some() || old != S::DEFAULT).then(|| note(old)).flatten()),
        };
        self.notes.extend(note);
        rule
    }
}

/// An output directory whose `KeepRules` are settled but which isn't open yet,
/// since opening it needs the voice, and the voice sample is judged against the
/// settled noise limit. Nothing has been written or printed yet.
pub struct SettledDir {
    out: PathBuf,
    saved: Option<Manifest>,
    rules: KeepRules,
    /// The noise limit given, else the one saved or defaulted in an existing
    /// `narration.txt`, else `None` in a new output directory.
    sample_noise_limit: Option<f32>,
    settled: Settled,
}

/// Settles the keep settings for a run in `out` from those `given` and those
/// `narration.txt` saved there, reading it once, so a bad line in it fails here.
pub fn settle(out: &Path, given: KeepSettings) -> Result<SettledDir> {
    let saved = read_manifest(out)?;
    let saved_rules = saved.as_ref().map(|saved| &saved.keep);
    let mut settled = Settled::default();
    let rules = KeepRules::settle(saved_rules, given, &out.join(MANIFEST), &mut settled);
    Ok(SettledDir {
        out: out.to_path_buf(),
        sample_noise_limit: given.noise_limit_db.or(saved_rules.map(|saved| saved.noise_limit_db)),
        saved,
        rules,
        settled,
    })
}

impl SettledDir {
    /// The keep settings the run follows.
    pub fn rules(&self) -> &KeepRules {
        &self.rules
    }

    /// The noise limit to judge the voice sample against: the one given, else the
    /// one an existing `narration.txt` saved, or the default if it has no
    /// `noise_limit_db` line, else `None` in a new output directory, so its advice
    /// names the default.
    pub fn sample_noise_limit(&self) -> Option<f32> {
        self.sample_noise_limit
    }

    /// The speed a run with `render` settings uses: the one given, else the one
    /// saved, else normal speed.
    pub fn speed(&self, render: &RenderSettings) -> f64 {
        render.speed.or(self.saved.as_ref().map(|saved| saved.speed)).unwrap_or(NORMAL_SPEED)
    }

    /// Opens the output directory to render the book's `chapters`, split within
    /// `limit`, in `voice` with `render` settings, saving the keep settings and
    /// the limit if they changed and saying what the run will use. Refuses a
    /// directory rendered any other way.
    pub fn open(
        self,
        book_title: &str,
        chapters: &[Vec<String>],
        limit: &ChunkLimit,
        voice: u64,
        render: &RenderSettings,
    ) -> Result<NarrationStore> {
        let speed = self.speed(render);
        let chunks = chunks_of(chapters, &format!("{voice:x}"), speed);
        fs::create_dir_all(self.out.join("chunks"))?;
        let manifest = Manifest {
            book: book_title.replace('\n', " "),
            voice: format!("{voice:x}"),
            precision: render.precision.clone(),
            speed,
            chunk_weight: Some(limit.weight()),
            keep: self.rules,
        };
        let limit_changed = self.saved.as_ref().is_some_and(|saved| saved.chunk_weight != manifest.chunk_weight);
        let store = NarrationStore { out: self.out, chunks, speed, manifest };
        store.require_same(self.saved)?;
        if self.settled.save_now || limit_changed {
            store.write_manifest()?;
        }
        for note in &self.settled.notes {
            eprintln!("{note}");
        }
        if limit_changed {
            eprintln!(
                "Chunks resized to stay under the length any-tts splits, so some will be generated again. \
                 {} saves the new limit.",
                store.out.join(MANIFEST).display()
            );
        }
        Ok(store)
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
    /// The `ChunkLimit` weight the chapters were split within, so the chunks can
    /// be found again without the voice sample. `None` when `narration.txt` has
    /// no `chunk_weight` line, and chunks are then limited only by characters.
    chunk_weight: Option<f64>,
    keep: KeepRules,
}

impl Manifest {
    fn render(&self) -> String {
        let mut text = format!(
            "book={}\nvoice={}\nprecision={}\nspeed={}\n",
            self.book, self.voice, self.precision, self.speed
        );
        if let Some(weight) = self.chunk_weight {
            text.push_str(&format!("{CHUNK_WEIGHT}={weight}\n"));
        }
        self.keep.render(&mut text);
        text
    }

    fn parse(text: &str) -> Result<Self> {
        let line = |name: &str| {
            text.lines().find_map(|line| line.strip_prefix(name)?.strip_prefix('=')).map(str::to_string)
        };
        let field = |name: &str| line(name).ok_or_else(|| eyre!("{MANIFEST} has no {name}"));
        Ok(Self {
            book: field("book")?,
            voice: field("voice")?,
            precision: field("precision")?,
            speed: line("speed").map_or(Ok(NORMAL_SPEED), |speed| speed.parse())?,
            chunk_weight: line(CHUNK_WEIGHT).map(|weight| weight.parse()).transpose()?,
            keep: KeepRules::parse(line)?,
        })
    }

    fn require(&self, wanted: &Self, out: &Path) -> Result<()> {
        let fields = [
            ("book", &self.book, &wanted.book),
            ("voice", &self.voice, &wanted.voice),
            ("precision", &self.precision, &wanted.precision),
            ("speed", &self.speed.to_string(), &wanted.speed.to_string()),
        ];
        for (name, saved, asked) in fields {
            if saved != asked {
                bail!(
                    "{} was rendered with {name} {saved}, not {asked}. Use a new output directory.",
                    out.display()
                );
            }
        }
        Ok(())
    }
}

pub struct NarrationStore {
    out: PathBuf,
    chunks: Vec<Chunk>,
    speed: f64,
    manifest: Manifest,
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
    /// the voice and settings it saved, changing nothing. `chapters` splits the
    /// book's chapters within the limit saved there, or only by characters in a
    /// directory with none saved.
    pub fn read(
        out: &Path,
        book_title: &str,
        chapters: impl FnOnce(&ChunkLimit) -> Vec<Vec<String>>,
    ) -> Result<Self> {
        let manifest = read_manifest(out)?
            .ok_or_else(|| eyre!("{} has no {MANIFEST}, so nothing has been rendered there", out.display()))?;
        let book = book_title.replace('\n', " ");
        if manifest.book != book {
            bail!("{} narrates {}, not {book}", out.display(), manifest.book);
        }
        let limit = manifest.chunk_weight.map_or_else(ChunkLimit::unlimited_weight, ChunkLimit::with_weight);
        Ok(Self {
            out: out.to_path_buf(),
            chunks: chunks_of(&chapters(&limit), &manifest.voice, manifest.speed),
            speed: manifest.speed,
            manifest,
        })
    }

    /// Refuses anything other than what this folder has already rendered.
    fn require_same(&self, saved: Option<Manifest>) -> Result<()> {
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
    fn save_manifest(&self) -> Result<()> {
        if !self.out.join(MANIFEST).exists() {
            self.write_manifest()?;
        }
        Ok(())
    }

    fn write_manifest(&self) -> Result<()> {
        let path = self.out.join(MANIFEST);
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, self.manifest.render())?;
        fs::rename(&temporary, &path)?;
        Ok(())
    }

    /// Older versions saved nothing, so a folder with chunks can only be trusted
    /// if some of them belong to this narration. They always used the default
    /// precision unless overridden for testing, and normal speed.
    fn older_manifest(&self, wanted: &Manifest) -> Result<Option<Manifest>> {
        let has_audio = fs::read_dir(self.out.join("chunks"))?
            .filter_map(Result::ok)
            .any(|entry| entry.path().extension().is_some_and(|extension| extension == "wav"));
        if !has_audio {
            return Ok(None);
        }
        if wanted.speed != NORMAL_SPEED {
            bail!(
                "{} was rendered at normal speed, not {}. Use a new output directory.",
                self.out.display(),
                wanted.speed
            );
        }
        if !self.chunks.iter().any(|chunk| self.has_audio(chunk)) {
            bail!(
                "{} has audio from a different voice or text. Use a new output directory.",
                self.out.display()
            );
        }
        Ok(Some(Manifest { precision: DEFAULT_PRECISION.into(), speed: NORMAL_SPEED, ..wanted.clone() }))
    }

    /// Finds what `targets` name: chunk numbers, or words from the book. Each audio
    /// file appears once, in book order, with every chunk number that shares it.
    /// Words only match chunks with audio, as they're for fixing what's been heard.
    pub fn select(&self, targets: &[&str]) -> Result<Vec<Selected<'_>>> {
        let mut wanted: HashMap<&str, Vec<String>> = HashMap::new();
        for target in targets {
            if let Some(number) = parse_number(target) {
                let chunk = self.chunks.iter().find(|chunk| (chunk.chapter + 1, chunk.position + 1) == number);
                let chunk = chunk.ok_or_else(|| eyre!("no chunk {target} in this book"))?;
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

    fn chunks_with_words(&self, words: &str) -> Result<Vec<&Chunk>> {
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
            bail!("nothing in the book matches \"{words}\"{hint}");
        }
        let numbers: Vec<String> = without.iter().map(|chunk| chunk.number()).collect();
        Err(eyre!(
            "\"{words}\" is only in chunks with no audio: {}. Name them by number to record them.",
            numbers.join(", ")
        ))
    }

    /// The speaking speed this folder narrates at.
    pub fn speed(&self) -> f64 {
        self.speed
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

    /// Saves a chunk's audio and check file, replacing the audio in one step. The
    /// check file records the checker in use, which heard `transcript`.
    pub fn store(&self, chunk: &Chunk, audio: &AudioSamples, transcript: &str, takes: u64) -> Result<()> {
        self.save_manifest()?;
        let path = self.audio_path(chunk);
        let transcript = transcript.replace(['\n', '\t'], " ");
        let check = Check { takes, transcript, checker: Some(checker::MODEL.to_string()) };
        fs::write(path.with_extension("check"), check.to_saved())?;
        let temporary = path.with_extension("tmp");
        audio.save_wav(&temporary).wrap_err_with(|| format!("couldn't save {}", temporary.display()))?;
        fs::rename(&temporary, path)?;
        Ok(())
    }

    /// Reads the stored check for a chunk that has audio.
    pub fn check(&self, chunk: &Chunk) -> Result<Check> {
        Check::from_saved(&fs::read_to_string(self.audio_path(chunk).with_extension("check"))?)
    }

    /// Records more tried seeds without changing the audio, its transcript or
    /// the checker that heard it.
    pub fn record_tried(&self, chunk: &Chunk, takes: u64) -> Result<()> {
        let check = Check { takes, ..self.check(chunk)? };
        let path = self.audio_path(chunk).with_extension("check");
        let temporary = path.with_extension("check.tmp");
        fs::write(&temporary, check.to_saved())?;
        fs::rename(&temporary, path)?;
        Ok(())
    }

    /// Saves a new take of a chunk as `store` does, first moving the take it has,
    /// if any, and its check file to `replaced/`.
    pub fn replace(&self, chunk: &Chunk, audio: &AudioSamples, transcript: &str, takes: u64) -> Result<()> {
        if self.has_audio(chunk) {
            self.retire(chunk)?;
        }
        self.store(chunk, audio, transcript, takes)
    }

    /// Moves a chunk's audio and check file to `replaced/`, ready for a new take.
    fn retire(&self, chunk: &Chunk) -> Result<()> {
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
    use crate::sample_check::SampleCheck;
    use crate::test_dir::TempDir;
    use any_tts::config::DType;
    use std::fs;

    fn runtime(device: DeviceSelection, dtype: DType) -> OmniVoiceRuntimeChoice {
        OmniVoiceRuntimeChoice { device, dtype }
    }

    fn settings() -> RenderSettings {
        RenderSettings { precision: "f16".into(), speed: None }
    }

    fn chapters(texts: &[&[&str]]) -> Vec<Vec<String>> {
        texts.iter().map(|chapter| chapter.iter().map(|text| text.to_string()).collect()).collect()
    }

    /// The chunk limit's weight tests open folders with.
    const LIMIT_WEIGHT: f64 = 412.5;

    /// Opens `dir` as a run does: settling `given`, then opening it.
    fn open_with(
        dir: &TempDir,
        title: &str,
        texts: &[&[&str]],
        voice: u64,
        render: &RenderSettings,
        given: KeepSettings,
    ) -> Result<NarrationStore> {
        open_within(dir, title, texts, LIMIT_WEIGHT, voice, render, given)
    }

    /// Opens `dir` as `open_with` does, splitting within `weight`.
    fn open_within(
        dir: &TempDir,
        title: &str,
        texts: &[&[&str]],
        weight: f64,
        voice: u64,
        render: &RenderSettings,
        given: KeepSettings,
    ) -> Result<NarrationStore> {
        settle(dir.path(), given)?.open(title, &chapters(texts), &ChunkLimit::with_weight(weight), voice, render)
    }

    #[test]
    fn reading_a_folder_splits_the_book_within_its_saved_limit() {
        let dir = TempDir::new("store-read-limit");
        render_at(&dir, None);
        assert!(fs::read_to_string(dir.join(MANIFEST)).unwrap().contains("\nchunk_weight=412.5\n"));
        let mut limit = None;
        NarrationStore::read(dir.path(), "A Book", |given| {
            limit = Some(*given);
            chapters(&[&["Hello there."]])
        })
        .unwrap();
        assert_eq!(limit, Some(ChunkLimit::with_weight(LIMIT_WEIGHT)));
    }

    #[test]
    fn a_new_limit_is_saved_on_opening() {
        let dir = TempDir::new("store-new-limit");
        render_at(&dir, None);
        open_within(&dir, "A Book", &[&["Hello there."]], 300.0, 0xabc, &settings(), KeepSettings::default()).unwrap();
        assert!(fs::read_to_string(dir.join(MANIFEST)).unwrap().contains("\nchunk_weight=300\n"));
    }

    #[test]
    fn reading_a_folder_without_a_saved_limit_limits_chunks_only_by_characters() {
        let dir = TempDir::new("store-read-no-limit");
        fs::write(dir.join(MANIFEST), "book=A Book\nvoice=abc\nprecision=f16\nspeed=1\n").unwrap();
        let mut limit = None;
        NarrationStore::read(dir.path(), "A Book", |given| {
            limit = Some(*given);
            chapters(&[&["a"]])
        })
        .unwrap();
        assert_eq!(limit, Some(ChunkLimit::unlimited_weight()));
    }

    fn open(dir: &TempDir, texts: &[&[&str]]) -> NarrationStore {
        open_with(dir, "A Book", texts, 0xabc, &settings(), KeepSettings::default()).unwrap()
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
        let dir = TempDir::new("store-nothing-yet");
        open_at(&dir, None).unwrap();
        assert!(!dir.join(MANIFEST).exists());
        assert!(open_at(&dir, Some(0.9)).is_ok(), "nothing has been rendered yet");
        render_at(&dir, Some(0.9));
        assert!(open_at(&dir, Some(1.1)).is_err());
    }

    #[test]
    fn speed_on_an_older_folder_is_refused_by_name() {
        let dir = TempDir::new("store-older-speed-name");
        fs::create_dir_all(dir.join("chunks")).unwrap();
        fs::write(dir.join("chunks").join("026926e6f4d06f3e.wav"), "audio").unwrap();
        let error = open_at(&dir, Some(0.9)).err().unwrap().to_string();
        assert!(error.contains("at normal speed"), "{error}");
    }

    #[test]
    fn partly_rendered_chapters_are_named_and_missing_ones_counted() {
        let dir = TempDir::new("store-partial");
        let store = open(&dir, &[&["a", "b"], &["c"], &[], &["d"], &["e"]]);
        store.store(&store.chunks()[0], &tone(), "a", 1).unwrap();
        store.store(&store.chunks()[3], &tone(), "d", 1).unwrap();
        assert_eq!(store.partly_rendered_chapters(), [0]);
        assert_eq!(store.incomplete_chapter_count(), 3);
    }

    fn open_as(dir: &TempDir, title: &str, voice: u64, precision: &str) -> Result<NarrationStore> {
        let settings = RenderSettings { precision: precision.into(), speed: None };
        open_with(dir, title, &[&["a"]], voice, &settings, KeepSettings::default())
    }

    fn numbers(selected: &[Selected]) -> Vec<String> {
        selected.iter().map(|item| item.chunk.number()).collect()
    }

    #[test]
    fn a_number_selects_that_chunk_even_with_no_audio() {
        let dir = TempDir::new("store-select-number");
        let store = open(&dir, &[&["one"], &["two", "three"]]);
        let selected = store.select(&["2.2"]).unwrap();
        assert_eq!(numbers(&selected), ["2.2"]);
        assert!(selected[0].phrases.is_empty());
    }

    #[test]
    fn selecting_never_adds_chunks_that_were_not_named() {
        let dir = TempDir::new("store-select-only");
        let store = open(&dir, &[&["one"], &["two"], &["three"]]);
        store.store(&store.chunks()[0], &tone(), "one", 1).unwrap();
        assert_eq!(numbers(&store.select(&["1.1"]).unwrap()), ["1.1"]);
    }

    #[test]
    fn a_number_outside_the_book_is_an_error() {
        let dir = TempDir::new("store-select-range");
        let store = open(&dir, &[&["one"]]);
        for target in ["1.2", "2.1", "0.1", "1.0"] {
            let error = store.select(&[target]).err().unwrap().to_string();
            assert!(error.contains(target), "{error}");
        }
    }

    #[test]
    fn words_select_only_chunks_with_audio() {
        let dir = TempDir::new("store-select-words");
        let store = open(&dir, &[&["the white rabbit"], &["a white hat"]]);
        store.store(&store.chunks()[0], &tone(), "x", 1).unwrap();
        let selected = store.select(&["white"]).unwrap();
        assert_eq!(numbers(&selected), ["1.1"]);
        assert_eq!(selected[0].phrases, ["white"]);
    }

    #[test]
    fn words_matching_only_missing_chunks_ask_for_a_number() {
        let dir = TempDir::new("store-select-missing");
        let store = open(&dir, &[&["the white rabbit"]]);
        let error = store.select(&["white"]).err().unwrap().to_string();
        assert!(error.contains("1.1"), "{error}");
    }

    #[test]
    fn a_bare_number_that_matches_nothing_shows_the_new_form() {
        let dir = TempDir::new("store-select-bare");
        let store = open(&dir, &[&["one"]]);
        let error = store.select(&["32"]).err().unwrap().to_string();
        assert!(error.contains("3.12"), "{error}");
    }

    #[test]
    fn shared_audio_is_selected_once_with_every_number() {
        let dir = TempDir::new("store-select-shared");
        let store = open(&dir, &[&["same line", "other"], &["same line"]]);
        store.store(&store.chunks()[0], &tone(), "same line", 1).unwrap();
        let selected = store.select(&["same line", "2.1"]).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].numbers, ["1.1", "2.1"]);
        assert_eq!(selected[0].phrases, ["same line"]);
    }

    #[test]
    fn selections_come_back_in_book_order() {
        let dir = TempDir::new("store-select-order");
        let store = open(&dir, &[&["a"], &["b"]]);
        assert_eq!(numbers(&store.select(&["2.1", "1.1"]).unwrap()), ["1.1", "2.1"]);
    }

    fn open_at(dir: &TempDir, speed: Option<f64>) -> Result<NarrationStore> {
        let settings = RenderSettings { speed, ..settings() };
        open_with(dir, "A Book", &[&["Hello there."]], 0xabc, &settings, KeepSettings::default())
    }

    #[test]
    fn a_new_speed_gets_its_own_cache_key() {
        // Worked out independently with FNV-1a of "3\nabc\nspeed=0.9\nHello there.".
        let dir = TempDir::new("store-speed-key");
        let store = open_at(&dir, Some(0.9)).unwrap();
        assert_eq!(store.audio_path(&store.chunks()[0]).file_name().unwrap(), "6c7ca5b9cb1c27dd.wav");
    }

    #[test]
    fn normal_speed_keeps_the_existing_cache_key() {
        let dir = TempDir::new("store-speed-normal");
        let store = open_at(&dir, Some(1.0)).unwrap();
        assert_eq!(store.audio_path(&store.chunks()[0]).file_name().unwrap(), "026926e6f4d06f3e.wav");
    }

    #[test]
    fn leaving_out_the_speed_uses_the_saved_one() {
        let dir = TempDir::new("store-speed-saved");
        render_at(&dir, Some(0.9));
        let store = open_at(&dir, None).unwrap();
        assert_eq!(store.speed(), 0.9);
        assert_eq!(store.audio_path(&store.chunks()[0]).file_name().unwrap(), "6c7ca5b9cb1c27dd.wav");
    }

    #[test]
    fn a_different_speed_is_refused_by_name() {
        let dir = TempDir::new("store-speed-refused");
        render_at(&dir, Some(0.9));
        let error = open_at(&dir, Some(1.1)).err().unwrap().to_string();
        assert!(error.contains("with speed"), "{error}");
    }

    #[test]
    fn an_older_folder_is_normal_speed() {
        let dir = TempDir::new("store-speed-older");
        fs::create_dir_all(dir.join("chunks")).unwrap();
        fs::write(dir.join("chunks").join("026926e6f4d06f3e.wav"), "audio").unwrap();
        assert_eq!(open_at(&dir, None).unwrap().speed(), 1.0);
        assert!(open_at(&dir, Some(0.9)).is_err());
    }

    #[test]
    fn a_folder_from_before_speed_was_saved_is_normal_speed() {
        let dir = TempDir::new("store-speed-manifest");
        fs::write(dir.join(MANIFEST), "book=A Book\nvoice=abc\nprecision=f16\n").unwrap();
        assert_eq!(open_at(&dir, None).unwrap().speed(), 1.0);
    }

    #[test]
    fn a_folder_reopens_with_the_same_voice_and_settings() {
        let dir = TempDir::new("store-same");
        render_as(&dir, "A Book", 1, "f16");
        assert!(open_as(&dir, "A Book", 1, "f16").is_ok());
    }

    #[test]
    fn a_different_voice_is_refused_by_name() {
        let dir = TempDir::new("store-voice");
        render_as(&dir, "A Book", 1, "f16");
        let error = open_as(&dir, "A Book", 2, "f16").err().unwrap().to_string();
        assert!(error.contains("with voice"), "{error}");
        assert!(error.contains("new output directory"), "{error}");
    }

    #[test]
    fn a_different_precision_is_refused_by_name() {
        let dir = TempDir::new("store-precision");
        render_as(&dir, "A Book", 1, "f16");
        let error = open_as(&dir, "A Book", 1, "f32").err().unwrap().to_string();
        assert!(error.contains("with precision"), "{error}");
    }

    #[test]
    fn a_different_book_is_refused_by_name() {
        let dir = TempDir::new("store-book");
        render_as(&dir, "A Book", 1, "f16");
        let error = open_as(&dir, "Another Book", 1, "f16").err().unwrap().to_string();
        assert!(error.contains("Another Book"), "{error}");
    }

    /// Makes a folder as older versions left it: chunks, but no saved settings.
    fn older_folder(dir: &TempDir, voice: u64) {
        let scratch = TempDir::new(&format!("store-older-of-{}", dir.path().display()).replace('/', "-"));
        let store = open_as(&scratch, "A Book", voice, "f16").unwrap();
        let name = store.audio_path(&store.chunks()[0]);
        fs::create_dir_all(dir.join("chunks")).unwrap();
        fs::write(dir.join("chunks").join(name.file_name().unwrap()), "audio").unwrap();
    }

    #[test]
    fn an_older_folder_with_matching_chunks_adopts_the_voice() {
        let dir = TempDir::new("store-adopt");
        older_folder(&dir, 1);
        open_as(&dir, "A Book", 1, "f16").unwrap();
        assert!(open_as(&dir, "A Book", 2, "f16").is_err(), "the adopted voice should now be saved");
    }

    #[test]
    fn an_older_folder_with_no_matching_chunks_is_refused() {
        let dir = TempDir::new("store-mismatch");
        older_folder(&dir, 1);
        let error = open_as(&dir, "A Book", 2, "f16").err().unwrap().to_string();
        assert!(error.contains("new output directory"), "{error}");
    }

    #[test]
    fn an_older_folder_is_assumed_to_be_the_default_precision() {
        let dir = TempDir::new("store-older-precision");
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

    /// Settles `given` in `dir` and opens it, as a run does, saving the keep
    /// settings if they changed.
    fn open_keeping(dir: &TempDir, given: KeepSettings) -> NarrationStore {
        open_with(dir, "A Book", &[&["Hello there."]], 0xabc, &settings(), given).unwrap()
    }

    /// Opens `dir` with `given` and stores a take, which saves the keep settings.
    fn render_keeping(dir: &TempDir, given: KeepSettings) {
        let store = open_keeping(dir, given);
        store.store(&store.chunks()[0], &tone(), "a", 1).unwrap();
    }

    /// The rules a run in `dir` given `given` follows, which settling alone decides.
    fn rules(dir: &TempDir, given: KeepSettings) -> KeepRules {
        *settle(dir.path(), given).unwrap().rules()
    }

    /// Every rule in `rules`, given.
    fn given(rules: KeepRules) -> KeepSettings {
        KeepSettings {
            noise_limit_db: Some(rules.noise_limit_db),
            takes: Some(rules.takes),
            pitch_range_target: rules.pitch_range_target,
        }
    }

    /// The error from settling the keep settings in `dir`.
    fn settle_error(dir: &TempDir) -> String {
        settle(dir.path(), KeepSettings::default()).err().unwrap().to_string()
    }

    /// The lines of `dir`'s `narration.txt` after book, voice, precision, speed and
    /// chunk weight.
    fn saved_lines(dir: &TempDir) -> Vec<String> {
        fs::read_to_string(dir.join(MANIFEST)).unwrap().lines().skip(5).map(str::to_string).collect()
    }

    const DEFAULT_RULES: KeepRules =
        KeepRules { noise_limit_db: DEFAULT_NOISE_LIMIT_DB, takes: TakePolicy::FirstPassing, pitch_range_target: None };
    const FIRST: KeepRules =
        KeepRules { noise_limit_db: -50.0, takes: TakePolicy::BestOf(4), pitch_range_target: Some(12.4) };
    const FIRST_LINES: [&str; 3] = ["noise_limit_db=-50", "takes=best-of-4", "pitch_range_st=12.4"];
    const SECOND: KeepRules =
        KeepRules { noise_limit_db: -48.5, takes: TakePolicy::BestOf(2), pitch_range_target: Some(10.0) };
    const SECOND_LINES: [&str; 3] = ["noise_limit_db=-48.5", "takes=best-of-2", "pitch_range_st=10"];

    #[test]
    fn a_given_setting_wins_then_the_saved_one_then_the_default() {
        let dir = TempDir::new("store-keep-for");
        assert_eq!(rules(&dir, KeepSettings::default()), DEFAULT_RULES);
        assert_eq!(rules(&dir, given(FIRST)), FIRST);
        render_keeping(&dir, given(FIRST));
        assert_eq!(rules(&dir, KeepSettings::default()), FIRST);
        assert_eq!(rules(&dir, given(SECOND)), SECOND);
        assert_eq!(saved_lines(&dir), FIRST_LINES, "only opening saves new settings");
        open_keeping(&dir, KeepSettings::default());
        assert_eq!(saved_lines(&dir), FIRST_LINES);
    }

    #[test]
    fn the_settings_are_saved_with_the_first_take() {
        let dir = TempDir::new("store-keep-first");
        let store = open_keeping(&dir, given(FIRST));
        assert!(!dir.join(MANIFEST).exists());
        store.store(&store.chunks()[0], &tone(), "a", 1).unwrap();
        assert_eq!(saved_lines(&dir), FIRST_LINES);
    }

    #[test]
    fn the_defaults_are_saved_when_none_are_given_and_no_pitch_range_target() {
        let dir = TempDir::new("store-keep-default");
        render_keeping(&dir, KeepSettings::default());
        assert_eq!(saved_lines(&dir), ["noise_limit_db=-55", "takes=first-passing"]);
        assert_eq!(rules(&dir, KeepSettings::default()), DEFAULT_RULES);
    }

    #[test]
    fn a_new_setting_replaces_only_its_saved_line_and_keeps_the_audio() {
        let changes = [
            KeepRules { noise_limit_db: SECOND.noise_limit_db, ..FIRST },
            KeepRules { takes: SECOND.takes, ..FIRST },
            KeepRules { pitch_range_target: SECOND.pitch_range_target, ..FIRST },
        ];
        for (index, changed) in changes.into_iter().enumerate() {
            let dir = TempDir::new(&format!("store-keep-changed-{index}"));
            render_keeping(&dir, given(FIRST));
            let before = fs::read_to_string(dir.join(MANIFEST)).unwrap();
            let store = open_keeping(&dir, given(changed));
            assert!(store.has_audio(&store.chunks()[0]), "the cached take should still be found");
            assert_eq!(store.audio_path(&store.chunks()[0]).file_name().unwrap(), "026926e6f4d06f3e.wav");
            let after = fs::read_to_string(dir.join(MANIFEST)).unwrap();
            assert_eq!(before.replace(FIRST_LINES[index], SECOND_LINES[index]), after);
            assert_eq!(rules(&dir, KeepSettings::default()), changed);
        }
    }

    /// A `narration.txt` with none of the keep settings.
    const OLDER_MANIFEST: &str = "book=A Book\nvoice=abc\nprecision=f16\nspeed=1\nchunk_weight=412.5\n";

    fn folder_without_saved_settings(name: &str) -> TempDir {
        let dir = TempDir::new(name);
        fs::create_dir_all(dir.join("chunks")).unwrap();
        fs::write(dir.join(MANIFEST), OLDER_MANIFEST).unwrap();
        dir
    }

    #[test]
    fn a_folder_without_saved_settings_uses_the_defaults_until_some_are_given() {
        let dir = folder_without_saved_settings("keep-older");
        assert_eq!(rules(&dir, KeepSettings::default()), DEFAULT_RULES);
        open_keeping(&dir, KeepSettings::default());
        assert_eq!(fs::read_to_string(dir.join(MANIFEST)).unwrap(), OLDER_MANIFEST);
        open_keeping(&dir, given(FIRST));
        assert_eq!(saved_lines(&dir), FIRST_LINES);
        assert_eq!(rules(&dir, KeepSettings::default()), FIRST);
    }

    #[test]
    fn a_manifest_is_saved_as_it_was_read() {
        let text = format!("{OLDER_MANIFEST}noise_limit_db=-50\ntakes=best-of-3\n");
        let manifest = Manifest::parse(&text).unwrap();
        assert_eq!(manifest.keep, KeepRules { takes: TakePolicy::BestOf(3), pitch_range_target: None, ..FIRST });
        assert_eq!(manifest.render(), text);
    }

    /// The note settling `S` from `saved` and `given` prints, and whether it saves.
    fn note<S: KeepSetting>(saved: Option<S::Rule>, given: Option<S::Value>) -> (Option<String>, bool) {
        let mut settled = Settled::default();
        settled.setting::<S>(saved, given, Path::new("out/narration.txt"));
        (settled.notes.pop(), settled.save_now)
    }

    #[test]
    fn notes_say_what_a_run_uses_and_what_it_replaces() {
        let said = |text: &str| Some(text.to_string());
        assert_eq!(note::<NoiseLimit>(None, None), (None, false));
        assert_eq!(note::<NoiseLimit>(None, Some(-50.0)), (said("Noise limit: -50 dB"), false));
        assert_eq!(
            note::<NoiseLimit>(Some(-50.0), Some(-48.0)),
            (said("Noise limit: -48 dB, replacing -50 dB in out/narration.txt"), true)
        );
        assert_eq!(note::<NoiseLimit>(Some(-50.0), Some(-50.0)), (said("Noise limit: -50 dB"), false));
        assert_eq!(note::<NoiseLimit>(Some(-50.0), None), (said("Noise limit: -50 dB"), false));
        assert_eq!(note::<NoiseLimit>(Some(DEFAULT_NOISE_LIMIT_DB), None), (None, false));

        let (best_of_2, best_of_4) = (Some(TakePolicy::BestOf(2)), Some(TakePolicy::BestOf(4)));
        assert_eq!(note::<TakePolicySetting>(None, best_of_4), (said("Takes: keeping the best of 4"), false));
        assert_eq!(
            note::<TakePolicySetting>(best_of_4, best_of_2),
            (said("Takes: keeping the best of 2, replacing the best of 4 in out/narration.txt"), true)
        );
        assert_eq!(note::<TakePolicySetting>(best_of_4, None), (said("Takes: keeping the best of 4"), false));
        assert_eq!(note::<TakePolicySetting>(Some(TakePolicy::FirstPassing), None), (None, false));

        let (unset, wide, narrow) = (Some(PitchRangeTarget::DEFAULT), Some(12.4), Some(10.0));
        assert_eq!(note::<PitchRangeTarget>(None, wide), (said("Pitch range: aiming for 12.4 semitones"), false));
        assert_eq!(
            note::<PitchRangeTarget>(unset, wide),
            (said("Pitch range: aiming for 12.4 semitones, now saved in out/narration.txt"), true)
        );
        assert_eq!(
            note::<PitchRangeTarget>(wide.map(Some), narrow),
            (said("Pitch range: aiming for 10 semitones, replacing 12.4 in out/narration.txt"), true)
        );
        assert_eq!(
            note::<PitchRangeTarget>(narrow.map(Some), narrow),
            (said("Pitch range: aiming for 10 semitones"), false)
        );
        assert_eq!(note::<PitchRangeTarget>(unset, None), (None, false));
    }

    #[test]
    fn a_bad_saved_setting_is_an_error_that_says_how_to_fix_it() {
        let bad = [
            ("noise_limit_db=loud", "a level from -100 to 0 dB"),
            ("takes=best-of-50", "first-passing or best-of-1 to best-of-10"),
            ("pitch_range_st=40", "a range from 2 to 30 semitones"),
        ];
        for (line, valid) in bad {
            let dir = folder_without_saved_settings("keep-bad");
            fs::write(dir.join(MANIFEST), format!("{OLDER_MANIFEST}{line}\n")).unwrap();
            let error = settle_error(&dir);
            assert_eq!(error, format!("narration.txt has {line}, which isn't {valid}. Correct or delete that line"));
        }
    }

    #[test]
    fn the_voice_sample_is_judged_against_the_given_or_saved_limit_or_the_default() {
        let dir = TempDir::new("store-limit-sample");
        let given_limit = |noise_limit_db| KeepSettings { noise_limit_db, ..KeepSettings::default() };
        let limit = |given| settle(dir.path(), given_limit(given)).unwrap().sample_noise_limit();
        assert_eq!(limit(None), None, "so its advice can name the default");
        assert_eq!(limit(Some(-50.0)), Some(-50.0));
        render_keeping(&dir, KeepSettings::default());
        assert_eq!(limit(None), Some(DEFAULT_NOISE_LIMIT_DB));
        assert_eq!(limit(Some(-60.0)), Some(-60.0));
    }

    #[test]
    fn a_folder_without_a_saved_limit_gives_the_voice_sample_the_default_limit() {
        let dir = folder_without_saved_settings("keep-older-sample");
        let limit = settle(dir.path(), KeepSettings::default()).unwrap().sample_noise_limit();
        assert_eq!(limit, Some(DEFAULT_NOISE_LIMIT_DB));
        let advice = SampleCheck::from_floor(-45.3, limit).advice(Path::new("voices/dad.wav"), 10.7).unwrap();
        assert!(advice.contains("above the noise limit of -55 dB"), "{advice}");
    }

    #[test]
    fn giving_one_setting_to_a_folder_without_saved_settings_saves_the_default_noise_limit() {
        let dir = folder_without_saved_settings("keep-older-one");
        open_keeping(&dir, KeepSettings { takes: Some(TakePolicy::BestOf(3)), ..KeepSettings::default() });
        assert_eq!(saved_lines(&dir), ["noise_limit_db=-55", "takes=best-of-3"]);
    }

    /// A check saying `takes` and `transcript`, heard by the checker in use now.
    fn checked(takes: u64, transcript: &str) -> Check {
        Check { takes, transcript: transcript.into(), checker: Some(checker::MODEL.into()) }
    }

    #[test]
    fn a_stored_take_can_be_read_back_with_the_checker_that_heard_it() {
        let dir = TempDir::new("store-roundtrip");
        let store = open(&dir, &[&["a"]]);
        let chunk = &store.chunks()[0];
        assert!(!store.has_audio(chunk));
        store.store(chunk, &tone(), "heard\tthis\nline", 2).unwrap();
        assert!(store.has_audio(chunk));
        assert_eq!(store.check(chunk).unwrap(), checked(2, "heard this line"));
    }

    #[test]
    fn a_check_file_from_before_checkers_were_recorded_reads_with_none() {
        let dir = TempDir::new("store-old-check");
        let store = open(&dir, &[&["a"]]);
        let chunk = &store.chunks()[0];
        store.store(chunk, &tone(), "heard", 1).unwrap();
        fs::write(store.audio_path(chunk).with_extension("check"), "4\nheard by whisper").unwrap();
        let check = store.check(chunk).unwrap();
        assert_eq!(check, Check { takes: 4, transcript: "heard by whisper".into(), checker: None });
    }

    #[test]
    fn reading_a_folder_finds_its_takes_at_its_saved_speed() {
        let dir = TempDir::new("store-read");
        render_at(&dir, Some(0.9));
        let store = NarrationStore::read(dir.path(), "A Book", |_| chapters(&[&["Hello there."]])).unwrap();
        assert_eq!(store.speed(), 0.9);
        assert_eq!(store.check(&store.chunks()[0]).unwrap(), checked(1, "a"));
        assert!(NarrationStore::read(dir.path(), "Another Book", |_| chapters(&[&["Hello there."]])).is_err());
    }

    #[test]
    fn reading_a_folder_with_nothing_rendered_is_an_error_that_changes_nothing() {
        let dir = TempDir::new("store-read-empty");
        let error = NarrationStore::read(dir.path(), "A Book", |_| chapters(&[&["a"]])).err().unwrap().to_string();
        assert!(error.ends_with("has no narration.txt, so nothing has been rendered there"), "{error}");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_malformed_check_file_is_an_error() {
        let dir = TempDir::new("store-malformed");
        let store = open(&dir, &[&["a"]]);
        let chunk = &store.chunks()[0];
        store.store(chunk, &tone(), "heard", 1).unwrap();
        fs::write(store.audio_path(chunk).with_extension("check"), "no newline").unwrap();
        assert!(store.check(chunk).is_err());
    }

    #[test]
    fn recording_tried_seeds_keeps_the_transcript() {
        let dir = TempDir::new("store-tried");
        let store = open(&dir, &[&["a"]]);
        let chunk = &store.chunks()[0];
        store.store(chunk, &tone(), "heard", 1).unwrap();
        store.record_tried(chunk, 7).unwrap();
        assert_eq!(store.check(chunk).unwrap(), checked(7, "heard"));
    }

    #[test]
    fn recording_tried_seeds_keeps_an_old_check_files_unrecorded_checker() {
        let dir = TempDir::new("store-tried-old");
        let store = open(&dir, &[&["a"]]);
        let chunk = &store.chunks()[0];
        store.store(chunk, &tone(), "heard", 1).unwrap();
        let path = store.audio_path(chunk).with_extension("check");
        fs::write(&path, "4\nheard by whisper").unwrap();
        store.record_tried(chunk, 9).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "9\nheard by whisper");
    }

    #[test]
    fn replacing_a_take_keeps_the_old_one_in_replaced() {
        let dir = TempDir::new("store-retire");
        let store = open(&dir, &[&["a"]]);
        let chunk = &store.chunks()[0];
        store.store(chunk, &tone(), "heard", 3).unwrap();
        store.replace(chunk, &AudioSamples::new(vec![0.2; 480], 24_000), "new", 5).unwrap();
        assert_eq!(store.check(chunk).unwrap(), checked(5, "new"));
        assert_eq!(AudioSamples::from_wav_file(store.audio_path(chunk)).unwrap().samples.len(), 480);
        let name = store.audio_path(chunk).file_stem().unwrap().to_str().unwrap().to_string();
        let old = dir.join("replaced").join(format!("{name}-3.wav"));
        assert_eq!(AudioSamples::from_wav_file(&old).unwrap().samples.len(), 240);
        let old_check = fs::read_to_string(old.with_extension("check")).unwrap();
        assert_eq!(old_check, format!("3\nheard\n{}", checker::MODEL));
    }

    #[test]
    fn replacing_a_chunk_with_no_take_only_stores_the_new_one() {
        let dir = TempDir::new("store-replace-new");
        let store = open(&dir, &[&["a"]]);
        let chunk = &store.chunks()[0];
        store.replace(chunk, &tone(), "heard", 2).unwrap();
        assert_eq!(store.check(chunk).unwrap(), checked(2, "heard"));
        assert!(!dir.join("replaced").exists());
    }

    #[test]
    fn pending_lists_only_the_chapters_asked_for_and_shared_audio_once() {
        let dir = TempDir::new("store-pending");
        let store = open(&dir, &[&["a", "b", "a"], &["c"]]);
        store.store(&store.chunks()[1], &tone(), "b", 1).unwrap();
        let numbers: Vec<String> = store.pending(&[0]).iter().map(|chunk| chunk.number()).collect();
        assert_eq!(numbers, ["1.1"]);
    }

    #[test]
    fn only_chapters_with_all_their_audio_are_complete() {
        let dir = TempDir::new("store-complete");
        let store = open(&dir, &[&["a", "b"], &["c"], &[], &["d"]]);
        store.store(&store.chunks()[0], &tone(), "a", 1).unwrap();
        store.store(&store.chunks()[3], &tone(), "d", 1).unwrap();
        assert_eq!(store.complete_chapters(), [3]);
        assert_eq!(store.missing_count(), 2);
    }

    #[test]
    fn chunk_numbers_count_from_one_within_each_chapter() {
        let dir = TempDir::new("store-numbers");
        let store = open(&dir, &[&["a", "b"], &["c"]]);
        let numbers: Vec<String> = store.chunks().iter().map(Chunk::number).collect();
        assert_eq!(numbers, ["1.1", "1.2", "2.1"]);
    }

    #[test]
    fn chunk_files_keep_the_existing_cache_key() {
        // Worked out independently with FNV-1a of "3\nabc\nHello there.".
        let dir = TempDir::new("store-key");
        let store = open(&dir, &[&["Hello there."]]);
        let path = store.audio_path(&store.chunks()[0]);
        assert_eq!(path.file_name().unwrap(), "026926e6f4d06f3e.wav");
    }
}
