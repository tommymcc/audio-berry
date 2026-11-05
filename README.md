# audio-berry

Narrates text in a cloned voice, entirely on your own machine, and plays the
result on a Cast speaker. Speech runs on OmniVoice through a fork of
[`any-tts`](https://github.com/tommymcc/any-tts). The only network access is
downloading the models once (about 3.3 GB for speech, plus the Whisper models
for checking) and talking to speakers on your LAN.

## Requirements

- Rust 1.85+ and `ffmpeg`
- An NVIDIA GPU, with `nvidia-smi` working. A 25-second clip peaked at about
  2.9 GB of VRAM on an 8 GB card. The CUDA build fails without a GPU. Build with
  `--no-default-features` for CPU only, which hasn't been tested with OmniVoice.

```sh
cargo build --release
```

## Adding a voice

Only clone your own voice, or the voice of someone who has agreed to it.

1. **Get a recording.** Use one person speaking with no music or other voices,
   in the delivery you want for the narration. Aim for a normal level with no
   clipping.

2. **Cut about 10 seconds of it.** OmniVoice recommends 3-10 seconds, and 11.4
   is what's been tested for narration. Over 20 seconds, the app uses up to the
   first 15, ending at a pause. `voices/` is ignored by git, so recordings stay
   out of the repo. To take 10 seconds starting 3 minutes in:

   ```sh
   mkdir -p voices
   ffmpeg -ss 180 -t 10 -i recording.mp3 -ac 1 -ar 24000 voices/me.wav
   ```

3. **Check the levels.** `max_volume` should be below 0 dB. If it reads 0.0 dB,
   the recording is probably clipped, so pick another section or record it
   again.

   ```sh
   ffmpeg -hide_banner -i voices/me.wav -af volumedetect -f null - 2>&1 | grep -E 'mean_volume|max_volume'
   ```

4. **Listen to a test.** Judge it by ear, ideally on the speaker you'll use.

   ```sh
   mkdir -p output
   ./target/release/audio-berry speak voices/me.wav "Twinkle, twinkle, little star." output/test.wav
   ./target/release/audio-berry play kitchen output/test.wav
   ```

OmniVoice also needs the words spoken in the sample. The first run transcribes
it locally and saves the text beside it, as `voices/me.txt`. Read that file and
correct any mistakes, since a wrong word no longer matches the audio. You can
also write the file yourself before the first run.

Takes vary, so if one sounds wrong, render another with a different seed before
blaming the sample:

```sh
OMNIVOICE_SEED=2 ./target/release/audio-berry speak voices/me.wav "Twinkle, twinkle, little star." output/test.wav
```

## Commands

```text
audio-berry speak <voice-sample> <text> <output.wav> [--speed <factor>]
audio-berry chapters <book.epub>
audio-berry book <voice-sample> <book.epub> <output-dir> [chapter numbers] [--speed <factor>]
audio-berry retake <voice-sample> <book.epub> <output-dir> <words or chunk number>...
audio-berry speakers
audio-berry play <speaker> <file>...
```

`play` matches part of a speaker's name, ignoring case, and needs exactly one
match. It interrupts whatever the speaker is playing. While playing, it serves
only the requested files to your LAN on a temporary port.

Pausing on the speaker doesn't count towards `play`'s time limit, which is the
length the speaker reports plus a minute. Leave `play` running until playback ends.

## Narrating a book

```sh
./target/release/audio-berry chapters book.epub
./target/release/audio-berry book voices/me.wav book.epub output/book
```

Chapter numbers come from `chapters`. Add them to `book` to narrate only those.
The book must be DRM-free, and each spine document becomes one chapter.

To try it on a free book, download
[Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11.epub3.images)
from Project Gutenberg. Its last chapter is Project Gutenberg's license, so name
chapters 1 to 13 if you don't want that narrated. This isn't the edition used in the
benchmarks below, which splits chapter 3 into 30 chunks rather than 29.

The model sometimes adds background noise, skips words, runs on or says fluent
nonsense, so every take is checked and retried up to 5 times with a new seed:

- **Background noise:** the quietest 5% of the take must be below -55 dB.
- **Length:** the duration must be plausible for the amount of text.
- **Accuracy:** Whisper, running on candle like OmniVoice, transcribes the take
  locally and compares it with the text. The models download on first use:
  `base.en` (290 MB) for checks and `medium.en` (3 GB) for voice samples.
  Whisper mishears some correct speech, so treat a flag as a reason to listen,
  not as proof of a mistake.

If no take passes, the best one is kept and marked `CHECK` in `report.tsv`, with
its text, transcript and chunk file so you can listen to it. Finished chunks are
cached in `chunks/`, so rerunning the same command resumes an interrupted book.

The M4B and `report.tsv` cover every chapter rendered in that output directory so
far, so rendering chapter 3 and then chapter 4 builds up one audiobook. The M4B
holds only complete chapters, and the command prints how many chunks and chapters
have no audio. For one chapter on its own, use its WAV in `chapters/`.

`book` with no chapter numbers renders everything still missing, which in a
partly rendered directory can take a long time. To rebuild without rendering,
name chapters that are already complete.

## Speed

`--speed` sets the speaking speed from 0.5 to 2.0, where 0.8 is slower and 1.25 is
faster. Actual lengths won't change by exactly that factor, so judge it by ear:

```sh
./target/release/audio-berry speak voices/me.wav "Twinkle, twinkle, little star." output/test.wav --speed 0.9
./target/release/audio-berry book voices/me.wav book.epub output/book-slow --speed 0.9
```

## One output directory, one narration

An output directory holds one book in one voice, with one speed and model
precision (`OMNIVOICE_DTYPE`). It saves them in `narration.txt` with its first take, and
later runs reuse the saved speed, so you don't have to repeat `--speed`. A run
with a different book, voice, speed or precision stops with an error rather than
rendering everything again, so use a new directory for those. Correcting a voice's
`.txt` file counts as a different voice.

Directories made before this was added have no `narration.txt`. They're treated as
normal speed and default precision, and they adopt the voice you give them if at
least one of their chunks matches. If none match, the run stops with an error.

Chunks are numbered by chapter and position, so `3.12` is the twelfth chunk of
chapter 3. To fix a slip you hear, give `retake` words from the book, or a chunk
number from `report.tsv`. It leaves the M4B and chapter WAVs as they were, and ends
by printing the `book` command that rebuilds them:

```sh
./target/release/audio-berry retake voices/me.wav book.epub output/book "took a watch out"
```

`retake` tries up to 6 new seeds per chunk. It keeps a new take only if it beats
the current one and, given words, if those words are heard. Replaced audio moves
to `replaced/`. A short phrase can match several chunks,
and each one gets new takes. `retake` only records what you name: words only match
chunks that already have audio, and a chunk with no audio has to be named by number.
Only `book` fills gaps. Chunks with identical text share one audio file, so a retake
changes all of them, and it prints each chunk number affected.

The output directory holds a chaptered `.m4b`, one WAV per chapter in `chapters/`,
the text as it was sent to the model in `text/`, and its saved settings in
`narration.txt`.

`GLOSSARY.md` defines the terms used here.

## Benchmarks

Narrating chapter 3 of `aliceDynamic.epub` (30 chunks, 536.8 s of audio) in a new
output directory, on an RTX 3070 Laptop GPU and a Ryzen 9 5900HS. Each build ran 3
times, taking turns. The times cover the whole command, including loading the model,
checking every take with Whisper and building the M4B.

| Build | Median | Range | Realtime |
|---|---|---|---|
| Before | 302.9 s | 302.7-303.1 s | 1.77x |
| Now | 282.8 s | 282.7-283.9 s | 1.90x |

That's 6.6% faster. All 30 chunks are byte-identical to the takes from before, in
every run, so the audio is unchanged. Every chunk passed on its first take, so a
chapter that needs retakes will take longer. Peak GPU memory stayed at 3203 MiB,
sampled once a second.

To repeat it:

```sh
time ./target/release/audio-berry book output/tommy2_ref.wav aliceDynamic.epub <new-dir> 3
```

On the slower build, a run with timers spent 255 s of its 312 s in the model's forward
passes, 21 s in the audio decoder and 16 s in Whisper.

Tried and not kept:

- **Fat LTO:** no faster, and the build took over 3 minutes.
- **One matmul per KV head, with no K and V copies:** 286 s when used alone, but
  rounding changed enough to give different takes on 4 of the 30 chunks.
- **RoPE by index selection, and output logits for target frames only:** no faster.
- **Rank 3 for the attention output's reshape:** 0.05 s faster on a 15 s clip, not
  worth a separate path.

## The any-tts fork

Speech uses the `integration` branch of
[`tommymcc/any-tts`](https://github.com/tommymcc/any-tts), pinned to a commit in
`Cargo.toml`. It's `any-tts` 0.2.0 with these changes, each on its own branch:

- **OmniVoice voice cloning:** adds reference encoding (HuBERT and DAC encoders
  with residual quantization). Run at F32, it picks the same tokens as Python given
  the same inputs and no random position choice. `scripts/omnivoice_parity.py` and
  the ignored `matches_python_oracle` test in the fork repeat that comparison.
- **Matching the Python package:** the masking schedule, length estimate, silence
  removal, fade-out and F32 audio codec.
- **Speaking speed:** `request.speed` divides the estimated audio length, as the
  Python package does. The original crate rejected it.
- **Running on CUDA:** OmniVoice crashed at the original default of BF16. It now
  runs, and defaults to F16, as BF16's first-step logits were 8 times further from
  F32. `OMNIVOICE_DTYPE` overrides it.
- **Seed mixing:** nearby values of `OMNIVOICE_SEED` now give unrelated takes.
- **Fewer strided operations:** attention expands K and V with `broadcast_as` instead
  of `repeat`, and `RmsNorm` works at rank 2. Strided operations are slow on CUDA, as
  they upload their layout and index each element by rank. The audio is byte-identical.

The branch also has VibeVoice fixes and faster depthwise convolutions, which
audio-berry doesn't use. The `vibevoice-support` branch has those on their own, in
case VibeVoice comes back as an option.

## License

The source code in this repository is MIT licensed. See [LICENSE](LICENSE).

It depends on the [`epub`](https://crates.io/crates/epub) crate, which is licensed
under GPL-3.0-or-later. Compiled binaries include that crate, so anyone distributing
a binary has to follow the GPL for the whole program.
