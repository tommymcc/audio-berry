//! Hears what's said in a take with NVIDIA's Parakeet TDT 0.6b v3, run on ONNX
//! Runtime, so the take can be checked against its text, because OmniVoice
//! sometimes produces fluent nonsense that sounds clean and runs to a plausible
//! length. It gives each word's timing as well as the words.

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

/// What was heard in a take.
#[derive(Clone, Debug, PartialEq)]
pub struct Heard {
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

impl Heard {
    /// Parakeet's words, leaving out the punctuation it gives as words of their own.
    fn from_result(result: TranscriptionResult) -> Heard {
        let words = result
            .tokens
            .into_iter()
            .filter(|token| token.text.chars().any(char::is_alphanumeric))
            .map(|token| Word { text: token.text, start: token.start, end: token.end })
            .collect();
        Heard { text: result.text.split_whitespace().collect::<Vec<_>>().join(" "), words }
    }
}

/// A loaded Parakeet model.
pub struct Checker {
    model: ParakeetTDT,
}

impl Checker {
    /// Downloads the model on first use, and loads it on the GPU the build is for:
    /// CUDA with the `cuda` feature, CoreML with `metal`, else the CPU. It never
    /// falls back to the CPU from the GPU, which would be many times slower.
    pub fn start() -> Result<Checker> {
        let directory = download()?;
        let environment = parakeet_rs::ort::init().build().wrap_err("couldn't start ONNX Runtime")?;
        let provider = provider(&environment)?;
        let config = ExecutionConfig::new().with_environment(environment).with_execution_provider(provider);
        let model = ParakeetTDT::from_pretrained(&directory, Some(config))
            .map_err(|error| eyre!("{error}"))
            .wrap_err_with(|| format!("couldn't load {MODEL} from {}", directory.display()))?;
        Ok(Checker { model })
    }

    /// What's said in `audio`, heard as it would be once saved as a WAV file.
    pub fn hear(&mut self, audio: &AudioSamples) -> Result<Heard> {
        let samples = resample_as_saved(audio, SAMPLE_RATE)?;
        let result = self
            .model
            .transcribe_samples(samples, SAMPLE_RATE, 1, Some(TimestampMode::Words))
            .map_err(|error| eyre!("{error}"))
            .wrap_err_with(|| format!("{MODEL} couldn't transcribe a take"))?;
        Ok(Heard::from_result(result))
    }
}

/// The folder of the model's files, downloading any not yet cached.
fn download() -> Result<PathBuf> {
    let mut directory = None;
    for file in FILES {
        let path = download_file(REPO, &format!("{MODEL}/{file}"))
            .wrap_err_with(|| format!("couldn't download {file} of {MODEL} from {REPO}"))?;
        directory = path.parent().map(PathBuf::from);
    }
    directory.ok_or_else(|| eyre!("{MODEL} has no files"))
}

/// The libraries ONNX Runtime's CUDA provider loads, and what provides them.
#[cfg(any(feature = "cuda", test))]
const CUDA_LIBRARIES: [(&str, &str); 4] = [
    ("libcudart.so.13", "the CUDA 13 runtime"),
    ("libcublas.so.13", "cuBLAS 13"),
    ("libcublasLt.so.13", "cuBLAS 13"),
    ("libcudnn.so.9", "cuDNN 9 for CUDA 13"),
];

/// The CUDA provider, once it's certain to run.
#[cfg(feature = "cuda")]
fn provider(environment: &Environment) -> Result<ExecutionProvider> {
    let missing = missing_libraries(&CUDA_LIBRARIES, loads);
    if !missing.is_empty() {
        return Err(eyre!(missing_libraries_message(&missing)));
    }
    register(environment, parakeet_rs::ort::ep::CUDA::default().build())
        .wrap_err("couldn't start ONNX Runtime's CUDA provider to check takes on the GPU")?;
    Ok(ExecutionProvider::Cuda)
}

/// The CoreML provider, once it's certain to run.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn provider(environment: &Environment) -> Result<ExecutionProvider> {
    register(environment, parakeet_rs::ort::ep::CoreML::default().build())
        .wrap_err("couldn't start ONNX Runtime's CoreML provider to check takes on the GPU")?;
    Ok(ExecutionProvider::CoreML)
}

/// Registers `provider` on a session that's then dropped, failing if it can't
/// be registered. When a provider can't be, ONNX Runtime runs on the CPU
/// instead without saying so, which for CUDA was 13 times slower.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn register(environment: &Environment, provider: parakeet_rs::ort::ep::ExecutionProviderDispatch) -> Result<()> {
    parakeet_rs::ort::session::Session::builder(environment)
        .and_then(|builder| builder.with_execution_providers([provider.error_on_failure()]).map_err(Into::into))
        .map(drop)
        .map_err(|error| eyre!("{error}"))
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn provider(_: &Environment) -> Result<ExecutionProvider> {
    Ok(ExecutionProvider::Cpu)
}

/// The libraries of `libraries` that `loads` can't load.
#[cfg(any(feature = "cuda", test))]
fn missing_libraries<'a>(
    libraries: &[(&'a str, &'a str)],
    loads: impl Fn(&str) -> bool,
) -> Vec<(&'a str, &'a str)> {
    libraries.iter().copied().filter(|(library, _)| !loads(library)).collect()
}

/// Says the CUDA build needs `missing`, naming each library and what provides it.
#[cfg(any(feature = "cuda", test))]
fn missing_libraries_message(missing: &[(&str, &str)]) -> String {
    let mut providers: Vec<&str> = Vec::new();
    for (_, provider) in missing {
        if !providers.contains(provider) {
            providers.push(provider);
        }
    }
    let libraries: Vec<&str> = missing.iter().map(|(library, _)| *library).collect();
    format!(
        "{MODEL} checks takes on the GPU with ONNX Runtime's CUDA provider, which needs {}: {} couldn't be loaded. \
         Install them, or build with --no-default-features to check takes on the CPU",
        list(&providers),
        list(&libraries),
    )
}

/// Names joined with commas and a final "and".
#[cfg(any(feature = "cuda", test))]
fn list(names: &[&str]) -> String {
    match names {
        [] => String::new(),
        [one] => one.to_string(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// Whether the dynamic loader can load `library`, by the name ONNX Runtime uses.
#[cfg(all(unix, any(feature = "cuda", test)))]
fn loads(library: &str) -> bool {
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

    #[test]
    fn heard_words_leave_out_punctuation_and_keep_their_times() {
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
        let heard = Heard::from_result(result);
        assert_eq!(heard.text, "Chapter one. Roving, wasn't it?");
        let words: Vec<(&str, f32, f32)> =
            heard.words.iter().map(|word| (word.text.as_str(), word.start, word.end)).collect();
        assert_eq!(
            words,
            [("Chapter", 0.08, 0.56), ("one", 0.64, 0.96), ("Roving", 1.6, 2.08), ("wasn't", 2.24, 2.56), ("it", 2.56, 2.72)]
        );
    }

    #[test]
    fn nothing_heard_is_no_words() {
        let heard = Heard::from_result(TranscriptionResult { text: String::new(), tokens: Vec::new() });
        assert_eq!(heard, Heard { text: String::new(), words: Vec::new() });
    }

    #[test]
    fn missing_cuda_libraries_are_named_with_what_provides_them() {
        let missing = missing_libraries(&CUDA_LIBRARIES, |library| !library.starts_with("libcublas"));
        assert_eq!(missing, [("libcublas.so.13", "cuBLAS 13"), ("libcublasLt.so.13", "cuBLAS 13")]);
        assert_eq!(
            missing_libraries_message(&missing),
            "parakeet-tdt-0.6b-v3 checks takes on the GPU with ONNX Runtime's CUDA provider, which needs cuBLAS 13: \
             libcublas.so.13 and libcublasLt.so.13 couldn't be loaded. Install them, or build with \
             --no-default-features to check takes on the CPU"
        );
        assert!(missing_libraries(&CUDA_LIBRARIES, |_| true).is_empty());
    }

    #[test]
    fn every_cuda_library_is_named_when_all_are_missing() {
        let message = missing_libraries_message(&missing_libraries(&CUDA_LIBRARIES, |_| false));
        assert!(
            message.contains(
                "needs the CUDA 13 runtime, cuBLAS 13 and cuDNN 9 for CUDA 13: libcudart.so.13, libcublas.so.13, \
                 libcublasLt.so.13 and libcudnn.so.9 couldn't be loaded"
            ),
            "{message}"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_library_that_isnt_installed_doesnt_load_and_libc_does() {
        assert!(!loads("libaudio-berry-not-a-library.so.0"));
        assert!(loads("libc.so.6"));
    }
}
