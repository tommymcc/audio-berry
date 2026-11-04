# audio-berry

Narrates text in a cloned voice, entirely on your own machine, and plays the
result on a Cast speaker. Speech runs on VibeVoice-1.5B through a patched copy
of [`any-tts`](https://github.com/TM9657/any-tts). The only network access is
downloading the model once (about 5.4 GB) and talking to speakers on your LAN.

## Requirements

- Rust 1.85+ and `ffmpeg`
- An NVIDIA GPU with about 8 GB of VRAM, with `nvidia-smi` working. The CUDA
  build fails without it. Build with `--no-default-features` for CPU only, which
  needs about 10 GB of RAM and ran at 0.06x realtime when last measured, before
  the convolution fix.

```sh
cargo build --release
```

## Adding a voice

Only clone your own voice, or the voice of someone who has agreed to it.

1. **Get a recording.** Use one person speaking with no music or other voices,
   in the delivery you want for the narration. Aim for a normal level with no
   clipping.

2. **Cut about 30 seconds of it.** This was the length tested for narration. 60
   seconds also works but peaks at about 7.1 GB of VRAM, against about 6.2 GB
   for 30 seconds. `voices/` is ignored by git, so recordings stay out of the
   repo. To take 30 seconds starting 3 minutes in:

   ```sh
   mkdir -p voices
   ffmpeg -ss 180 -t 30 -i recording.mp3 -ac 1 -ar 24000 voices/me.wav
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

Takes vary, and some pick up faint music-like background noise. If you hear it,
render another take with a different seed before blaming the sample:

```sh
VIBEVOICE_SEED=2 ./target/release/audio-berry speak voices/me.wav "Twinkle, twinkle, little star." output/test.wav
```

## Commands

```text
audio-berry speak <voice-sample> <text> <output.wav>
audio-berry chapters <book.epub>
audio-berry book <voice-sample> <book.epub> <output-dir> [chapter numbers]
audio-berry retake <voice-sample> <book.epub> <output-dir> <words or chunk number>...
audio-berry speakers
audio-berry play <speaker> <file>...
```

`play` matches part of a speaker's name, ignoring case, and needs exactly one
match. It interrupts whatever the speaker is playing. While playing, it serves
only the requested files to your LAN on a temporary port.

## Narrating a book

```sh
./target/release/audio-berry chapters book.epub
./target/release/audio-berry book voices/me.wav book.epub output/book
```

Chapter numbers come from `chapters`. Add them to `book` to narrate only those.
The book must be DRM-free, and each spine document becomes one chapter.

The model sometimes adds background noise, skips words, runs on or says fluent
nonsense, so every take is checked and retried up to 5 times with a new seed:

- **Background noise:** the quietest 5% of the take must be below -55 dB.
- **Length:** the duration must be plausible for the amount of text.
- **Accuracy:** `scripts/transcribe.py` transcribes the take locally and compares
  it with the text. This needs [`uv`](https://docs.astral.sh/uv/), which downloads
  faster-whisper and a small model on first use.

If no take passes, the best one is kept and marked `CHECK` in `report.tsv`, with
its text, transcript and chunk file so you can listen to it. Finished chunks are
cached in `chunks/`, so rerunning the same command resumes an interrupted book.

To fix a slip you hear, give `retake` words from the book, or a chunk number from
`report.tsv`, then rerun `book` to rebuild the M4B:

```sh
./target/release/audio-berry retake voices/me.wav book.epub output/book "took a watch out"
./target/release/audio-berry book voices/me.wav book.epub output/book
```

`retake` tries up to 6 new seeds per chunk. It keeps a new take only if it beats
the current one and, given words, if those words are heard. Replaced audio moves
to `replaced/`. It also records any chunk with no audio, such as one whose text
has changed, and it leaves the M4B alone. A short phrase can match several chunks,
and each one gets new takes.

The output directory holds a chaptered `.m4b`, one WAV per chapter in `chapters/`,
and the text as it was sent to the model in `text/`.

## The patched any-tts

`vendor/any-tts` is `any-tts` 0.2.0 with these changes:

- **BF16 on CUDA:** VibeVoice crashed on dtype mismatches, because several
  tensors and scalars stayed F32 while the model was BF16.
- **Depthwise convolutions:** `layers/conv.rs` runs them in one batch. candle
  makes one convolution call per channel, which made audio feedback take about
  1 s per frame. This took generation from 0.12x to about 1.48x realtime on an
  RTX 3070 Mobile.
- **`VIBEVOICE_DIFFUSION_STEPS`:** overrides the diffusion step count. The
  default of 20 is recommended, as lowering it saved almost no time.
