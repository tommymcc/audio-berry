//! Transcribes each take with NVIDIA's Parakeet TDT 0.6b v3, run on ONNX
//! Runtime, so the take can be checked against its text, because OmniVoice
//! sometimes produces fluent nonsense that sounds clean and runs to a plausible
//! length. The transcript gives each word's timing as well as the words.

use crate::audio::resample_as_saved;
use any_tts::AudioSamples;
use any_tts::download::download_file;
use color_eyre::eyre::{Result, WrapErr, eyre};
use parakeet_rs::ort::environment::Environment;
use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, TimestampMode, TranscriptionResult, Transcriber};
use std::path::PathBuf;

/// The checker's name, as check files record it.
pub const MODEL: &str = "parakeet-tdt-0.6b-v3";
/// The Hugging Face repository holding the model, in a folder named `MODEL`.
const REPO: &str = "altunenes/parakeet-rs";
/// The model's files. The encoder's weights, 2.5 GB, are in its `.data` file.
const FILES: [&str; 4] = ["vocab.txt", "encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"];
/// The sample rate Parakeet hears at.
const SAMPLE_RATE: u32 = 16_000;

/// What the checker heard in a take.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    /// The words heard, with Parakeet's capitals and punctuation.
    pub text: String,
    pub words: Vec<Word>,
}

/// A word heard, with when it was said, in seconds from the start of the take.
///
/// Parakeet's end times run on into the silence after a word, so the time from
/// one word's end to the next one's start is shorter than the pause between them,
/// and isn't a measure of its length.
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code, reason = "nothing reads the timings yet")]
pub struct Word {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

impl Transcript {
    /// Parakeet's words, leaving out the punctuation it gives as words of their own.
    fn from_result(result: TranscriptionResult) -> Transcript {
        let words = result
            .tokens
            .into_iter()
            .filter(|token| token.text.chars().any(char::is_alphanumeric))
            .map(|token| Word { text: token.text, start: token.start, end: token.end })
            .collect();
        Transcript { text: result.text.split_whitespace().collect::<Vec<_>>().join(" "), words }
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

    /// What's said in `audio`, heard as it would be once saved as a WAV file.
    pub fn transcribe(&mut self, audio: &AudioSamples) -> Result<Transcript> {
        let samples = resample_as_saved(audio, SAMPLE_RATE)?;
        let result = self
            .model
            .transcribe_samples(samples, SAMPLE_RATE, 1, Some(TimestampMode::Words))
            .map_err(|error| eyre!("{error}"))
            .wrap_err_with(|| format!("{MODEL} couldn't transcribe a take"))?;
        Ok(Transcript::from_result(result))
    }
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
            "couldn't start ONNX Runtime's CUDA provider to check takes on the GPU. Or {}",
            other_builds(cfg!(target_os = "macos"))
        )
    })?;
    Ok(ExecutionProvider::Cuda)
}

/// The CoreML provider, once it's certain to run.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn provider(environment: &Environment) -> Result<ExecutionProvider> {
    register(environment, parakeet_rs::ort::ep::CoreML::default().build())
        .wrap_err("couldn't start ONNX Runtime's CoreML provider to check takes on the GPU")?;
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

/// The builds that check takes without CUDA, for a CUDA build's errors: on a
/// Mac, `metal` as well as the CPU.
#[cfg(any(feature = "cuda", test))]
fn other_builds(macos: bool) -> &'static str {
    if macos {
        "build with --no-default-features --features metal to check takes on a Mac's GPU, or with \
         --no-default-features to check them on the CPU"
    } else {
        "build with --no-default-features to check takes on the CPU"
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
        "{MODEL} checks takes on the GPU with ONNX Runtime's CUDA provider, which needs {}: {} couldn't be \
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
        let transcript = Transcript::from_result(result);
        assert_eq!(transcript.text, "Chapter one. Roving, wasn't it?");
        let words: Vec<(&str, f32, f32)> =
            transcript.words.iter().map(|word| (word.text.as_str(), word.start, word.end)).collect();
        assert_eq!(
            words,
            [("Chapter", 0.08, 0.56), ("one", 0.64, 0.96), ("Roving", 1.6, 2.08), ("wasn't", 2.24, 2.56), ("it", 2.56, 2.72)]
        );
    }

    #[test]
    fn nothing_heard_is_no_words() {
        let transcript = Transcript::from_result(TranscriptionResult { text: String::new(), tokens: Vec::new() });
        assert_eq!(transcript, Transcript { text: String::new(), words: Vec::new() });
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
            "parakeet-tdt-0.6b-v3 checks takes on the GPU with ONNX Runtime's CUDA provider, which needs cuBLAS 13: \
             libcublas.so.13 and libcublasLt.so.13 couldn't be loaded. On Ubuntu, install libcublas-13-1, or build \
             with --no-default-features to check takes on the CPU"
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
            "build with --no-default-features --features metal to check takes on a Mac's GPU, or with \
             --no-default-features to check them on the CPU"
        );
        assert_eq!(other_builds(false), "build with --no-default-features to check takes on the CPU");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_library_that_isnt_installed_cant_be_loaded_and_libc_can() {
        assert!(!can_load("libaudio-berry-not-a-library.so.0"));
        assert!(can_load("libc.so.6"));
    }
}
