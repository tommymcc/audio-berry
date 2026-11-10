//! Hears what's said in audio with NVIDIA's Parakeet TDT 0.6b v3, run on ONNX
//! Runtime. It transcribes each take so the take can be checked against its
//! text, because OmniVoice sometimes produces fluent nonsense that sounds clean
//! and runs to a plausible length. It also transcribes voice samples, and the
//! recordings `sections` looks through. The transcript gives each word's timing
//! as well as the words.

use crate::audio::resample_as_saved;
use any_tts::AudioSamples;
use any_tts::download::download_file;
use color_eyre::eyre::{Result, WrapErr, eyre};
use parakeet_rs::ort::environment::Environment;
use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, TimestampMode, TranscriptionResult, Transcriber};
use std::ops::Range;
use std::path::PathBuf;

/// The checker's name, as check files record it.
pub const MODEL: &str = "parakeet-tdt-0.6b-v3";
/// The Hugging Face repository holding the model, in a folder named `MODEL`.
const REPO: &str = "altunenes/parakeet-rs";
/// The model's files. The encoder's weights, 2.5 GB, are in its `.data` file.
const FILES: [&str; 4] = ["vocab.txt", "encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"];
/// The sample rate Parakeet hears at.
const SAMPLE_RATE: u32 = 16_000;

/// The longest stretch of audio Parakeet hears at once. Longer audio is heard
/// in windows, since on a 65 second recording Parakeet left out a whole phrase
/// that it heard in a clip cut from it. Heard whole, a 102 second interview
/// lost "I was a young lad of seventeen, my eldest brother", and in windows of
/// up to 45 seconds it lost "I was a young lad of seventeen". Windows of 30 or
/// 20 seconds kept it, raw or cleaned. Each cut costs a word or two near it
/// now and then, such as "close enough" heard as "closing after" just after
/// one, and windows of 20 seconds cut twice as often as 30 for no fewer
/// mishearings. On the GPU, the 102 seconds took 0.8 seconds in any windows.
const MAX_WINDOW_SECONDS: usize = 30;
/// A window ends in the quietest stretch of its last this many seconds, so it
/// isn't cut in the middle of a word.
const CUT_SEARCH_SECONDS: usize = 10;
/// Loudness is measured in frames of a tenth of a second at Parakeet's sample
/// rate.
const CUT_FRAME_SAMPLES: usize = SAMPLE_RATE as usize / 10;
/// The quietest stretch is this many frames long, half a second, so a window
/// ends in the middle of a long pause, such as between sentences, rather than
/// a short one between words.
const QUIET_FRAMES: usize = 5;

/// What the checker heard in some audio, such as a take.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transcript {
    /// The words heard, with Parakeet's capitals and punctuation.
    pub text: String,
    /// The words heard, without punctuation.
    pub words: Vec<Word>,
    /// The words heard and, between them, the punctuation Parakeet gives as
    /// tokens of its own, such as "." and ",", in the order heard.
    pub tokens: Vec<Word>,
}

/// A word heard, or a punctuation mark, with when it was said, in seconds from
/// the start of the audio.
///
/// Parakeet's end times run on into the silence after a word, so the time from
/// one word's end to the next one's start is shorter than the pause between them,
/// and isn't a measure of its length.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

/// What was heard in a take, as a take is judged by it: the transcript, and the
/// words heard with their times when known. A check file from before word times
/// were saved has none.
#[derive(Clone, Copy, Debug)]
pub struct Heard<'a> {
    pub transcript: &'a str,
    pub words: Option<&'a [Word]>,
}

impl Transcript {
    /// The transcript and its words with their times.
    pub fn heard(&self) -> Heard<'_> {
        Heard { transcript: &self.text, words: Some(&self.words) }
    }

    /// Parakeet's words, with the punctuation it gives as words of their own
    /// kept apart from them, heard in audio starting `offset` seconds in.
    fn from_result(result: TranscriptionResult, offset: f32) -> Transcript {
        let tokens: Vec<Word> = result
            .tokens
            .into_iter()
            .map(|token| Word { text: token.text, start: offset + token.start, end: offset + token.end })
            .collect();
        let words = tokens.iter().filter(|token| token.text.chars().any(char::is_alphanumeric)).cloned().collect();
        Transcript { text: result.text.split_whitespace().collect::<Vec<_>>().join(" "), words, tokens }
    }

    /// Adds what was heard in the next window of the audio.
    fn extend(&mut self, next: Transcript) {
        if !next.text.is_empty() {
            if !self.text.is_empty() {
                self.text.push(' ');
            }
            self.text.push_str(&next.text);
        }
        self.words.extend(next.words);
        self.tokens.extend(next.tokens);
    }
}

/// A loaded Parakeet model.
pub struct Checker {
    model: ParakeetTDT,
}

impl Checker {
    /// Downloads the model on first use, and loads it on the GPU the build is for:
    /// CUDA with the `cuda` feature, CoreML with `metal`, else the CPU. It stops
    /// with an error if that GPU's provider can't start, rather than run on the CPU.
    pub fn start() -> Result<Checker> {
        let directory = download_model()?;
        let environment = parakeet_rs::ort::init().build().wrap_err("couldn't start ONNX Runtime")?;
        let provider = provider(&environment)?;
        let config = ExecutionConfig::new().with_environment(environment).with_execution_provider(provider);
        let model = ParakeetTDT::from_pretrained(&directory, Some(config))
            .map_err(|error| eyre!("{error}"))
            .wrap_err_with(|| format!("couldn't load {MODEL} from {}", directory.display()))?;
        Ok(Checker { model })
    }

    /// What's said in `audio`, heard as it would be once saved as a WAV file,
    /// window by window if it's long, with each word's times from its start.
    pub fn transcribe(&mut self, audio: &AudioSamples) -> Result<Transcript> {
        let samples = resample_as_saved(audio, SAMPLE_RATE)?;
        let mut heard = Transcript::default();
        for window in windows(&samples) {
            let offset = window.start as f32 / SAMPLE_RATE as f32;
            let result = self
                .model
                .transcribe_samples(samples[window].to_vec(), SAMPLE_RATE, 1, Some(TimestampMode::Words))
                .map_err(|error| eyre!("{error}"))
                .wrap_err_with(|| format!("{MODEL} couldn't transcribe the audio"))?;
            heard.extend(Transcript::from_result(result, offset));
        }
        Ok(heard)
    }
}

/// The windows `samples`, at Parakeet's sample rate, are heard in, from start
/// to end: each at most `MAX_WINDOW_SECONDS` long, and cut at the quietest
/// half second in its last `CUT_SEARCH_SECONDS`.
fn windows(samples: &[f32]) -> Vec<Range<usize>> {
    let max = MAX_WINDOW_SECONDS * SAMPLE_RATE as usize;
    let search = CUT_SEARCH_SECONDS * SAMPLE_RATE as usize;
    let loudness = |frame: &[f32]| frame.iter().map(|sample| sample * sample).sum::<f32>();
    let mut windows = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        let mut end = samples.len();
        if end - start > max {
            let from = start + max - search;
            let frames: Vec<f32> = samples[from..start + max].chunks_exact(CUT_FRAME_SAMPLES).map(loudness).collect();
            let quietest = frames
                .windows(QUIET_FRAMES)
                .map(|stretch| stretch.iter().sum::<f32>())
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map_or(0, |(index, _)| index);
            let middle = quietest + QUIET_FRAMES / 2;
            end = from + middle * CUT_FRAME_SAMPLES + CUT_FRAME_SAMPLES / 2;
        }
        windows.push(start..end);
        start = end;
    }
    windows
}

/// The folder of the model's files, downloading any not yet cached.
fn download_model() -> Result<PathBuf> {
    let mut directory = None;
    for file in FILES {
        let path = download_file(REPO, &format!("{MODEL}/{file}"))
            .wrap_err_with(|| format!("couldn't download {file} of {MODEL} from {REPO}"))?;
        directory = path.parent().map(PathBuf::from);
    }
    directory.ok_or_else(|| eyre!("{MODEL} has no files"))
}

/// The CUDA provider, once it's certain to run. On Linux, the libraries it needs
/// are loaded first, so a missing one is named.
#[cfg(feature = "cuda")]
fn provider(environment: &Environment) -> Result<ExecutionProvider> {
    #[cfg(target_os = "linux")]
    {
        let missing = missing_libraries(&CUDA_LIBRARIES, can_load);
        if !missing.is_empty() {
            return Err(eyre!(missing_libraries_message(&missing, false)));
        }
    }
    register(environment, parakeet_rs::ort::ep::CUDA::default().build()).wrap_err_with(|| {
        format!(
            "couldn't start ONNX Runtime's CUDA provider to run {MODEL} on the GPU. Or {}",
            other_builds(cfg!(target_os = "macos"))
        )
    })?;
    Ok(ExecutionProvider::Cuda)
}

/// The CoreML provider, once it's certain to run.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn provider(environment: &Environment) -> Result<ExecutionProvider> {
    register(environment, parakeet_rs::ort::ep::CoreML::default().build())
        .wrap_err_with(|| format!("couldn't start ONNX Runtime's CoreML provider to run {MODEL} on the GPU"))?;
    Ok(ExecutionProvider::CoreML)
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn provider(_: &Environment) -> Result<ExecutionProvider> {
    Ok(ExecutionProvider::Cpu)
}

/// Registers `provider` on a session that's then dropped, failing if it can't
/// be registered. When a provider can't be, ONNX Runtime runs on the CPU
/// instead without saying so.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn register(environment: &Environment, provider: parakeet_rs::ort::ep::ExecutionProviderDispatch) -> Result<()> {
    parakeet_rs::ort::session::Session::builder(environment)
        .and_then(|builder| builder.with_execution_providers([provider.error_on_failure()]).map_err(Into::into))
        .map(drop)
        .map_err(|error| eyre!("{error}"))
}

/// The builds that run the checker without CUDA, for a CUDA build's errors: on
/// a Mac, `metal` as well as the CPU.
#[cfg(any(feature = "cuda", test))]
fn other_builds(macos: bool) -> &'static str {
    if macos {
        "build with --no-default-features --features metal to run it on a Mac's GPU, or with \
         --no-default-features to run it on the CPU"
    } else {
        "build with --no-default-features to run it on the CPU"
    }
}

/// A shared library ONNX Runtime's CUDA provider loads, by the name it's loaded by.
#[cfg(any(all(feature = "cuda", target_os = "linux"), test))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Library {
    file: &'static str,
    /// What the library is part of.
    provided_by: &'static str,
    /// The Ubuntu packages that install it.
    packages: &'static str,
}

/// The libraries `libonnxruntime_providers_cuda.so` needs, other than the C and
/// C++ runtimes.
#[cfg(any(all(feature = "cuda", target_os = "linux"), test))]
const CUDA_LIBRARIES: [Library; 6] = [
    Library { file: "libcuda.so.1", provided_by: "the NVIDIA driver", packages: "an NVIDIA driver" },
    Library { file: "libcudart.so.13", provided_by: "the CUDA 13 runtime", packages: "cuda-cudart-13-1" },
    Library { file: "libcublas.so.13", provided_by: "cuBLAS 13", packages: "libcublas-13-1" },
    Library { file: "libcublasLt.so.13", provided_by: "cuBLAS 13", packages: "libcublas-13-1" },
    Library {
        file: "libcurand.so.10",
        provided_by: "cuRAND 10",
        packages: "libcurand10 (Ubuntu's) or libcurand-13-1 (NVIDIA's)",
    },
    Library { file: "libcudnn.so.9", provided_by: "cuDNN 9 for CUDA 13", packages: "libcudnn9-cuda-13" },
];

/// The libraries of `libraries` that `can_load` can't load.
#[cfg(any(all(feature = "cuda", target_os = "linux"), test))]
fn missing_libraries(libraries: &[Library], can_load: impl Fn(&str) -> bool) -> Vec<Library> {
    libraries.iter().copied().filter(|library| !can_load(library.file)).collect()
}

/// Says the CUDA build needs `missing`, naming each library, what it's part of
/// and the packages that install it, and the other builds, as on a Mac if `macos`.
#[cfg(any(all(feature = "cuda", target_os = "linux"), test))]
fn missing_libraries_message(missing: &[Library], macos: bool) -> String {
    let distinct = |field: fn(&Library) -> &'static str| {
        let mut values: Vec<&str> = Vec::new();
        for library in missing {
            if !values.contains(&field(library)) {
                values.push(field(library));
            }
        }
        values
    };
    let files: Vec<&str> = missing.iter().map(|library| library.file).collect();
    format!(
        "{MODEL} runs on the GPU with ONNX Runtime's CUDA provider, which needs {}: {} couldn't be \
         loaded. On Ubuntu, install {}, or {}",
        joined_with_and(&distinct(|library| library.provided_by)),
        joined_with_and(&files),
        joined_with_and(&distinct(|library| library.packages)),
        other_builds(macos),
    )
}

/// Names joined with commas and a final "and".
#[cfg(any(all(feature = "cuda", target_os = "linux"), test))]
fn joined_with_and(names: &[&str]) -> String {
    match names {
        [] => String::new(),
        [one] => one.to_string(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// Whether the dynamic loader can load `library`, by the name ONNX Runtime uses.
#[cfg(all(target_os = "linux", any(feature = "cuda", test)))]
fn can_load(library: &str) -> bool {
    let Ok(name) = std::ffi::CString::new(library) else {
        return false;
    };
    // Safety: the name is a valid C string, and the library is only loaded, as
    // ONNX Runtime would load it, and released again.
    unsafe {
        let handle = libc::dlopen(name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
        if handle.is_null() {
            return false;
        }
        libc::dlclose(handle);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use parakeet_rs::TimedToken;

    fn token(text: &str, start: f32, end: f32) -> TimedToken {
        TimedToken { text: text.into(), start, end }
    }

    fn files(libraries: &[Library]) -> Vec<&str> {
        libraries.iter().map(|library| library.file).collect()
    }

    #[test]
    fn transcript_words_leave_out_punctuation_and_keep_their_times() {
        let result = TranscriptionResult {
            text: "Chapter one.  Roving,  wasn't it?".into(),
            tokens: vec![
                token("Chapter", 0.08, 0.56),
                token("one", 0.64, 0.96),
                token(".", 0.96, 1.6),
                token("Roving", 1.6, 2.08),
                token(",", 2.08, 2.16),
                token("wasn't", 2.24, 2.56),
                token("it", 2.56, 2.72),
                token("?", 2.72, 3.0),
            ],
        };
        let transcript = Transcript::from_result(result, 0.0);
        assert_eq!(transcript.text, "Chapter one. Roving, wasn't it?");
        assert_eq!(
            texts_and_times(&transcript.words),
            [("Chapter", 0.08, 0.56), ("one", 0.64, 0.96), ("Roving", 1.6, 2.08), ("wasn't", 2.24, 2.56), ("it", 2.56, 2.72)]
        );
        let tokens: Vec<&str> = transcript.tokens.iter().map(|token| token.text.as_str()).collect();
        assert_eq!(tokens, ["Chapter", "one", ".", "Roving", ",", "wasn't", "it", "?"]);
    }

    fn texts_and_times(words: &[Word]) -> Vec<(&str, f32, f32)> {
        words.iter().map(|word| (word.text.as_str(), word.start, word.end)).collect()
    }

    #[test]
    fn nothing_heard_is_no_words() {
        let transcript = Transcript::from_result(TranscriptionResult { text: String::new(), tokens: Vec::new() }, 0.0);
        assert_eq!(transcript, Transcript::default());
    }

    #[test]
    fn windows_join_with_times_from_the_start_of_the_audio() {
        let window = |text: &str, tokens: Vec<TimedToken>, offset| {
            Transcript::from_result(TranscriptionResult { text: text.into(), tokens }, offset)
        };
        let mut heard = Transcript::default();
        heard.extend(window("One.", vec![token("One", 0.5, 0.9), token(".", 0.9, 1.0)], 0.0));
        heard.extend(window("", Vec::new(), 30.0));
        heard.extend(window("Two", vec![token("Two", 0.25, 0.75)], 52.5));
        assert_eq!(heard.text, "One. Two");
        assert_eq!(texts_and_times(&heard.words), [("One", 0.5, 0.9), ("Two", 52.75, 53.25)]);
        assert_eq!(heard.tokens.len(), 3);
    }

    /// Loud audio, `seconds` long, at Parakeet's sample rate.
    fn loud(seconds: f32) -> Vec<f32> {
        vec![0.5; (seconds * SAMPLE_RATE as f32) as usize]
    }

    fn seconds(windows: &[Range<usize>]) -> Vec<(f32, f32)> {
        let at = |sample: usize| sample as f32 / SAMPLE_RATE as f32;
        windows.iter().map(|window| (at(window.start), at(window.end))).collect()
    }

    #[test]
    fn audio_up_to_the_longest_window_is_heard_at_once() {
        assert_eq!(seconds(&windows(&loud(MAX_WINDOW_SECONDS as f32))), [(0.0, MAX_WINDOW_SECONDS as f32)]);
        assert!(windows(&[]).is_empty());
    }

    #[test]
    fn long_audio_is_cut_at_its_quietest_moments() {
        // Pauses at 25 and at 50 seconds, in 70 seconds of speech.
        let mut audio = loud(25.0);
        audio.extend(vec![0.0; SAMPLE_RATE as usize / 2]);
        audio.extend(loud(24.5));
        audio.extend(vec![0.01; SAMPLE_RATE as usize / 2]);
        audio.extend(loud(20.0));
        let windows = windows(&audio);
        let cuts: Vec<f32> = seconds(&windows).iter().map(|(_, end)| *end).collect();
        assert_eq!(windows.len(), 3, "{cuts:?}");
        assert!((25.0..25.5).contains(&cuts[0]) && (50.0..50.5).contains(&cuts[1]), "{cuts:?}");
        assert_eq!(windows[1].start, windows[0].end);
        assert_eq!(windows[2].end, audio.len());
    }

    #[test]
    fn a_window_ends_in_a_long_pause_rather_than_a_short_silent_gap() {
        // A silent gap of a fifth of a second at 22 seconds, and a pause of
        // over half a second at 27 with a little noise in it.
        let mut audio = loud(22.0);
        audio.extend(vec![0.0; SAMPLE_RATE as usize / 5]);
        audio.extend(loud(4.8));
        audio.extend(vec![0.01; SAMPLE_RATE as usize * 6 / 10]);
        audio.extend(loud(10.0));
        let end = seconds(&windows(&audio))[0].1;
        assert!((27.0..27.6).contains(&end), "{end}");
    }

    #[test]
    fn every_library_the_cuda_provider_needs_is_checked() {
        // The NEEDED entries of libonnxruntime_providers_cuda.so, other than the
        // C and C++ runtimes.
        let mut needed = files(&CUDA_LIBRARIES);
        needed.sort_unstable();
        assert_eq!(
            needed,
            ["libcublas.so.13", "libcublasLt.so.13", "libcuda.so.1", "libcudart.so.13", "libcudnn.so.9", "libcurand.so.10"]
        );
    }

    #[test]
    fn missing_cuda_libraries_are_named_with_what_provides_them_and_their_packages() {
        let missing = missing_libraries(&CUDA_LIBRARIES, |library| !library.starts_with("libcublas"));
        assert_eq!(files(&missing), ["libcublas.so.13", "libcublasLt.so.13"]);
        assert_eq!(
            missing_libraries_message(&missing, false),
            "parakeet-tdt-0.6b-v3 runs on the GPU with ONNX Runtime's CUDA provider, which needs cuBLAS 13: \
             libcublas.so.13 and libcublasLt.so.13 couldn't be loaded. On Ubuntu, install libcublas-13-1, or build \
             with --no-default-features to run it on the CPU"
        );
        assert!(missing_libraries(&CUDA_LIBRARIES, |_| true).is_empty());
    }

    #[test]
    fn a_missing_curand_names_both_packages() {
        let missing = missing_libraries(&CUDA_LIBRARIES, |library| library != "libcurand.so.10");
        let message = missing_libraries_message(&missing, false);
        assert!(
            message.contains(
                "which needs cuRAND 10: libcurand.so.10 couldn't be loaded. On Ubuntu, install libcurand10 \
                 (Ubuntu's) or libcurand-13-1 (NVIDIA's)"
            ),
            "{message}"
        );
    }

    #[test]
    fn every_cuda_library_is_named_when_all_are_missing() {
        let message = missing_libraries_message(&missing_libraries(&CUDA_LIBRARIES, |_| false), false);
        assert!(
            message.contains(
                "needs the NVIDIA driver, the CUDA 13 runtime, cuBLAS 13, cuRAND 10 and cuDNN 9 for CUDA 13: \
                 libcuda.so.1, libcudart.so.13, libcublas.so.13, libcublasLt.so.13, libcurand.so.10 and \
                 libcudnn.so.9 couldn't be loaded"
            ),
            "{message}"
        );
    }

    #[test]
    fn on_a_mac_a_cuda_build_points_to_the_metal_build() {
        assert_eq!(
            other_builds(true),
            "build with --no-default-features --features metal to run it on a Mac's GPU, or with \
             --no-default-features to run it on the CPU"
        );
        assert_eq!(other_builds(false), "build with --no-default-features to run it on the CPU");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_library_that_isnt_installed_cant_be_loaded_and_libc_can() {
        assert!(!can_load("libaudio-berry-not-a-library.so.0"));
        assert!(can_load("libc.so.6"));
    }
}

