# audio-berry

Narrates text in a cloned voice, entirely on your own machine, and plays the
result on a Cast speaker. Speech runs on OmniVoice through a fork of
[`any-tts`](https://github.com/tommymcc/any-tts). The only network access is
downloading the models once (about 3.3 GB for speech, plus the Whisper models
for checking) and talking to speakers on your LAN.

## Make an audiobook

You need a DRM-free EPUB, a recording of the voice you want, and either an NVIDIA GPU
or an Apple Silicon Mac. Only clone your own voice, or the voice of someone who has
agreed to it. To try it on a free book, download
[Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11.epub3.images)
from Project Gutenberg.

1. **Build it.** Install Rust 1.85+ and `ffmpeg`, then run one of these in this
   folder:

   ```sh
   cargo build --release                                         # NVIDIA GPU
   cargo build --release --no-default-features --features metal  # Apple Silicon Mac
   ```

2. **Cut 10 seconds of the voice.** Choose a stretch with one person speaking and
   no music. This takes 10 seconds starting 3 minutes into `recording.mp3`:

   ```sh
   mkdir -p voices
   ffmpeg -ss 180 -t 10 -i recording.mp3 -ac 1 -ar 24000 voices/me.wav
   ```

   For a recording with background noise, see
   [Adding a voice](#adding-a-voice) for `clean`, and for `sections`, which
   suggests the best stretches to cut.

3. **Test the voice.** The first run downloads the models (about 6.3 GB) and
   writes down what's said in the sample as `voices/me.txt`. Correct any wrong
   words in that file, then listen to `output/test.wav`:

   ```sh
   mkdir -p output
   ./target/release/audio-berry speak voices/me.wav "Twinkle, twinkle, little star." output/test.wav
   ```

4. **List the chapters:**

   ```sh
   ./target/release/audio-berry chapters book.epub
   ```

5. **Narrate the book.** This narrates every chapter. To narrate only some, add
   their numbers from the list, such as `2 3 4`. If it stops, run the same command
   again and it carries on where it left off.

   ```sh
   ./target/release/audio-berry book voices/me.wav book.epub output/my-book
   ```

6. **Listen.** The audiobook is the `.m4b` file in `output/my-book`, with chapter
   markers, ready for any audiobook player. The text is narrated in chunks of a few
   sentences, and each chunk is checked as it's made. Any chunk that still fails
   after 5 takes is marked `CHECK` in `output/my-book/report.tsv`.
   [Narrating a book](#narrating-a-book) explains the checks and how to fix a chunk
   with `retake`.

On an RTX 3070 Laptop GPU, narration takes about half as long as the audio it makes.
On the one Apple Silicon Mac tested, a short chapter took about 4 times as long. For Alice, name
chapters 1 to 13, since chapter 14 is Project Gutenberg's license.

## Requirements

- Rust 1.85+ and `ffmpeg`
- An NVIDIA GPU, with `nvidia-smi` working. A 25-second clip peaked at about
  2.9 GB of VRAM on an 8 GB card. The CUDA build fails without a GPU. Build with
  `--no-default-features` for CPU only, which hasn't been tested with OmniVoice.
- Or an Apple Silicon Mac, built with `--no-default-features --features metal`.

### Apple Silicon

Metal has been tested on one Mac with 32 GB of memory, narrating 62 seconds of
audio from Alice. All 7 chunks passed their checks on the first take, and it took
230 s including loading the model. Its peak memory footprint was 7.4 GB. A full
chapter, retrying a failed take, `retake` and `play` haven't been tried on a Mac.

The `Model loaded` line says which device it's using, which should be `metal:0`.
On Metal the model runs at F32, with the audio decoder and voice encoder on the
CPU. Audio from a Mac won't be byte-identical to audio from CUDA.

To build, narrate a short test and collect the results in one report:

```sh
scripts/metal_check.sh voices/me.wav
```

## Adding a voice

Only clone your own voice, or the voice of someone who has agreed to it.

1. **Get a recording.** Use one person speaking with no music or other voices,
   in the delivery you want for the narration. Aim for a normal level with no
   clipping.

2. **Find the best stretches.** `sections` suggests stretches of the recording
   to use as a voice sample, best first, each with the `clean` command that
   makes it. Add a number to get more or fewer than 3:

   ```sh
   ./target/release/audio-berry sections recording.mp3
   ```

   ```text
   1. 20.1-30.7 (10.6s), floor -45.2 dB, 70% speech
      Roughly: "Used to work on the reverse sail most times. Very late times. ..."
      audio-berry clean recording.mp3 20.1 30.7 voices/recording-20.1s.wav
   ```

   It cleans the whole recording as `clean` does, then scores each 8 to 15
   second stretch that starts and ends in a pause. Mostly speech, whole
   sentences and clear words score higher. Long pauses, clipping, a noisy floor
   and an uneven level score lower, but don't rule a stretch out, so check the
   floor and any `clipped` mark. The words are only a rough guide: the sample's
   first use transcribes it properly. It can't tell voices apart, so with more
   than one speaker, check each suggestion is the right one. It looks through
   at most 30 minutes, so cut a longer recording first.

3. **Make the voice sample.** Run `clean` for two or three of the suggestions,
   to compare them in step 6. About 10 seconds is right: OmniVoice recommends
   3-10 seconds, and 11.4 is what's been tested for narration. Over 20 seconds,
   the app uses up to the first 15, ending at a pause. `voices/` is ignored by
   git, so recordings stay out of the repo.

   `clean` removes background noise, such as birds or wind, as it cuts. It
   cleans the whole recording with DeepFilterNet 3, which is built in and runs
   on the CPU, then takes 20.1 to 30.8 seconds as a voice sample. It reproduces
   the earlier Python DeepFilterNet 0.5.6 result:

   ```sh
   ./target/release/audio-berry clean recording.mp3 20.1 30.8 voices/me.wav
   ```

   Times can also be minutes and seconds, such as `1:05.5`. Pick the start and end
   in pauses between words. It refuses an end before the start or past the end of
   the recording.

   `--reduction <dB>` caps how much noise is removed, 15 dB by default. More
   removes more background but makes the clone sound less like the speaker: on one
   outdoor recording, voice ID was 0.85 at 15 dB, 0.80 at 20 and 0.74 at 100 (no
   limit). The default keeps a little room tone, which every take copies, so you
   may need a higher [noise limit](#noise-limit).

   A recording with no background noise doesn't need cleaning. This takes 10
   seconds starting 3 minutes in:

   ```sh
   mkdir -p voices
   ffmpeg -ss 180 -t 10 -i recording.mp3 -ac 1 -ar 24000 voices/me.wav
   ```

4. **Check the levels.** `max_volume` should be below 0 dB. If it reads 0.0 dB,
   the recording is probably clipped, so pick another section or record it
   again.

   ```sh
   ffmpeg -hide_banner -i voices/me.wav -af volumedetect -f null - 2>&1 | grep -E 'mean_volume|max_volume'
   ```

5. **Check its words.** OmniVoice also needs the words spoken in the sample. The
   first command to use it, such as `audition` in the next step, transcribes it
   locally and saves the text beside it, as `voices/me.txt`. Read that file and
   correct any mistakes, since a wrong word no longer matches the audio. You can
   also write the file yourself before the first run.

   Every run also prints the sample's [floor](#noise-limit), since takes copy it:

   ```text
   Voice sample: using 10.7s, floor -66.6 dB
   ```

   A floor more than 7 dB above the noise limit (-48 dB for the default) is marked
   noisy, since many takes will then fail the limit. Above -40 dB, it asks whether
   the sample is a raw recording. On the first run, it also says what to do: raise
   the noise limit, or clean again with a higher reduction, or clean a raw
   recording. It only advises: the run carries on either way.

6. **Listen to a test.** Judge it by ear, ideally on the speaker you'll use.
   To compare the clone with the original, `audition` says the sample's own
   words in the cloned voice, then plays the sample and the clone in turn.
   Keep the sample most like the speaker:

   ```sh
   ./target/release/audio-berry audition voices/me.wav kitchen
   ```

   To hear it say something else:

   ```sh
   mkdir -p output
   ./target/release/audio-berry speak voices/me.wav "Twinkle, twinkle, little star." output/test.wav
   ./target/release/audio-berry play kitchen output/test.wav
   ```

   Takes vary, so if one sounds wrong, render another with a different seed before
   blaming the sample:

   ```sh
   ./target/release/audio-berry speak voices/me.wav "Twinkle, twinkle, little star." output/test.wav --seed 2
   ```

7. **Add allowances, if needed.** Once you've narrated some of a book, if
   Whisper hears the speaker's accent the same wrong way again and again, such as
   "in" as "and", `mishearings` lists the candidates and an `.allowances` file
   beside the sample lets those takes pass. See [Allowances](#allowances).

## Commands

```text
audio-berry speak [OPTIONS] <voice-sample> <text> <output.wav>
audio-berry chapters <book.epub>
audio-berry book [OPTIONS] <voice-sample> <book.epub> <output-dir> [chapter]...
audio-berry retake [OPTIONS] <voice-sample> <book.epub> <output-dir> <target>...
audio-berry mishearings <book.epub> <output-dir> [voice-sample]
audio-berry speakers
audio-berry play <speaker> <file>...
audio-berry audition [OPTIONS] <voice-sample> [speaker]
audio-berry sections [OPTIONS] <recording> [count]
audio-berry clean [OPTIONS] <recording> <start> <end> <output.wav>
```

`audio-berry <command> --help` describes a command's arguments and options:
`--speed <factor>` for `speak`, `book` and `audition`, `--seed <n>` for `speak`
and `audition`, `--noise-limit <dB>` and `--precision <f32|f16|bf16>` for
`speak`, `book`, `retake` and `audition`, and `--reduction <dB>` for `sections`
and `clean`. An environment variable can give `--noise-limit`
(`AUDIO_BERRY_NOISE_LIMIT_DB`), `--seed` (`OMNIVOICE_SEED`) or `--precision`
(`OMNIVOICE_DTYPE`) instead, and the option wins if both are given. An invalid
value stops a command that takes the option. `book` and `retake` pick their own
seeds, so they ignore `OMNIVOICE_SEED`. A `retake` target is words from the book, or a
chunk number such as 3.12.

`book` and `retake` are explained in [Narrating a book](#narrating-a-book) and
[One output directory, one narration](#one-output-directory-one-narration).

`mishearings` lists the words most often heard as other words in the takes kept
in an output directory, to help write a voice's allowances. Given the voice
sample, it marks the pairs already allowed. It only reads. See
[Allowances](#allowances).

`play` matches part of a speaker's name, ignoring case, and needs exactly one
match. It interrupts whatever the speaker is playing. While playing, it serves
only the requested files and their covers to your LAN on a temporary port.

Speakers with a screen show each file's title, author, book and cover. `play`
reads these from the file's tags. `book` writes them into the M4B and each chapter
WAV, and saves the EPUB's cover as `cover.jpg` or `cover.png` in the output
folder. The M4B holds its cover. For a WAV, `play` looks for a cover in the WAV's
folder and the folder above. A file with no tags shows its file name.

Pausing on the speaker doesn't count towards `play`'s time limit, which is the
length the speaker reports plus a minute. Leave `play` running until playback ends.

`audition` says a voice sample's own words, from its `.txt` file, in the cloned
voice, then plays the sample and the clone in turn on the speaker, so you hear the
same words in both. If there's no `.txt` yet, it transcribes the sample first.
It saves them in `output/audition`, so `voices/me.wav` gives `me-original.wav` and
`me-clone.wav`. The original is the part of the sample the model hears, at 24 kHz
mono, and trimmed to at most 15 seconds if the sample is over 20. If you wrote the
`.txt` for a longer recording than that, it warns that the clone will say more
than the original. Without a speaker, it only saves them. Like `speak`, it takes `--seed`.

`sections` suggests the best `count` stretches of a recording to use as a voice
sample, 3 by default, each with the `clean` command that makes it. With
`--reduction`, it cleans as `clean` would with it. See
[Adding a voice](#adding-a-voice).

`clean` makes a voice sample from a noisy recording: it removes background noise
from the whole recording, then saves `start` to `end` as 24 kHz mono 16-bit WAV.
If the output already has a `.txt` beside it, it renames that to `.txt.old`,
since it describes the old audio, so the first use transcribes the new sample.
See [Adding a voice](#adding-a-voice).

## Narrating a book

```sh
./target/release/audio-berry chapters book.epub
./target/release/audio-berry book voices/me.wav book.epub output/book
```

Chapter numbers come from `chapters`. Add them to `book` to narrate only those.
The book must be DRM-free, and each spine document becomes one chapter.

The Project Gutenberg copy of Alice linked above isn't the edition used in the
benchmarks below, which splits chapter 3 into 30 chunks rather than 29.

The model sometimes adds background noise, skips words, runs on or says fluent
nonsense, so every take is checked and retried up to 5 times with a new seed:

- **Background noise:** the quietest 5% of the take must be below -55 dB.
  Some voices need a different limit; see [Noise limit](#noise-limit).
- **Length:** the duration must be plausible for the amount of text.
- **Accuracy:** Whisper, running on candle like OmniVoice, transcribes the take
  locally and compares it with the text. The models download on first use:
  `base.en` (290 MB) for checks and `medium.en` (3 GB) for voice samples.
  Whisper mishears some correct speech, so treat a flag as a reason to listen,
  not as proof of a mistake. If it mishears a speaker's accent the same way
  again and again, see [Allowances](#allowances).

If no take passes, the best one is kept and marked `CHECK` in `report.tsv`, with
its text, transcript and chunk file so you can listen to it. Finished chunks are
cached in `chunks/`, so rerunning the same command resumes an interrupted book.

In a terminal, `book` and `retake` show a progress bar with the time left and the
latest chunk's take and recall. For `book` the bar covers the whole book, so a
resumed run starts part-way along, but the time left comes from this run's pace.
Redirected to a file, they print one line per chunk this run makes instead, such as
`[3/40] 22s of audio, take 1, recall 0.98, 2 min elapsed, about 31 min left`.

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

## Noise limit

Each take's background is measured as the level of its quietest 5% (in 50 ms
frames). A take louder than the limit is treated as noisy and retried. The
default is **-55 dB**, which suits a voice sample recorded somewhere quiet. Takes
heard as clean have measured about -91 to -96 dB, and takes with audible
background -34 to -48 dB.

Give `--noise-limit <dB>` to change the limit, from -100 to 0 dB, where higher
allows more background. It applies to `book` (whether a take passes), `retake`
(which take wins) and the `CHECK` flags in `report.tsv`. `speak` and `audition`
don't check takes, so they only judge the voice sample against it.

**When to raise it.** The model copies the sample's background, so a sample with
faint room tone, such as a noisy recording cleaned with a denoiser, gives every
take a little of it, a few dB quieter. On one speaker, a sample cleaned at 15 dB
gave takes about -54 dB and voice ID 0.85, and at 30 dB about -59 dB and 0.75,
where the speaker's own recordings scored 0.91. So keeping the 15 dB sample and
raising the limit gave the closest likeness. Each run marks a sample noisy when its
floor is more than 7 dB above the noise limit.

With such a sample, you'll see many takes rejected on the floor alone, with good
recall:

```text
  take 2 rejected: floor -51 dB, 19.2s, recall 0.94, extra 0.04
```

Each chunk is then rendered 5 times, about five times slower, and most still end
up marked `CHECK`. Listen to a few of those chunks, ideally on the speaker you'll
use. If they sound clean, raise the limit to just above the floors you see. For
a sample whose takes measure about -51 to -57 dB:

```sh
./target/release/audio-berry book voices/dad.wav book.epub output/book --noise-limit -50
```

Don't raise it to get past noise you can hear. Pick a cleaner stretch of the
recording, or clean the sample harder, which may make the voice sound less like
the speaker.

**When to lower it.** If takes that pass still have background you can hear,
such as a hum or faint music, lower the limit so they're retried:

```sh
./target/release/audio-berry book voices/me.wav book.epub output/book --noise-limit -65
```

**It's saved in the output directory.** The limit goes in `narration.txt` with
the first take, so give it once. Later `book` and `retake`
runs, and `report.tsv`, use the saved value:

```sh
./target/release/audio-berry retake voices/dad.wav book.epub output/book 3.12
```

To change it, give it again. The new value is used and saved, and the run prints
the old one:

```text
Noise limit: -48 dB, replacing -50 dB in output/book/narration.txt
```

Changing it doesn't render anything again: chunks that already have audio are
kept, and only new takes and the `CHECK` flags are judged against the new value.

An output directory started before limits were saved uses -55 until you give
one, and the run says so. Its file isn't changed, because it may have been
rendered with a limit that wasn't saved. If it was, give that limit once:

```sh
./target/release/audio-berry book voices/dad.wav book.epub output/book --noise-limit -50
```

## Allowances

Whisper mishears some speakers' accents the same way again and again. If it
hears a speaker's "in" as "and", good takes fail the word check, get rendered
again and end up marked `CHECK`. Allowances tell the check which words a voice
may be heard saying as others. They go in a file beside the voice sample, named
like its `.txt` but ending `.allowances`, so `voices/dad.wav` has
`voices/dad.allowances`:

```text
# Dad's accent: the book's word = what Whisper hears
in = and
at = had
```

Each line pairs a word from the book with the word heard instead. `#` starts a
comment. Words are compared as the check compares them, ignoring case and
punctuation, with small numbers spelled out, and each side must be one word. A
bad line stops `book` or `retake` with the file and line number. Only those
commands, which judge takes, read the file, so `speak` and `audition` ignore it.
When a voice has allowances, `book` and `retake` say so:

```text
Allowances: 2 from voices/dad.allowances
```

A pair works one way: `in = and` lets the book's "in" be heard as "and", but not
the book's "and" as "in". Add the reverse as its own line only if that happens
too.

Allowances aren't part of the voice. Adding or changing them keeps every chunk
already rendered and nothing is saved in `narration.txt`: they only change how
takes are judged from then on. That covers new takes in `book` and `retake`,
which take a `retake` keeps, and the `CHECK` flags in `report.tsv`, which are
judged again from each kept take's saved transcript. `book` doesn't render a
chunk again once it has audio, so a chunk flagged before only gets a new take
from `retake`. A `retake` phrase counts as heard through the allowances too, so
with `in = and`, `retake ... "in the boat"` accepts a take heard as "and the boat".

To find candidates, list the words most often heard as others in an output
directory. It only reads, so it can run during a render. Add the voice sample to
mark the pairs it already allows:

```sh
./target/release/audio-berry mishearings book.epub output/book voices/dad.wav
```

```text
Words heard as other words in 635 takes, most often first:
ye = you                # 39 of 109 times
peleg = pelleg          # 34 of 69 times
harpooneer = harpaneer  # 27 of 40 times
in = and                # 17 of 789 times
...
```

It shows each word heard as the same other word at least twice in the takes
kept, and how often the book's word appears. Rejected takes aren't saved, so it
can't count those. A take whose audio is there but whose check file isn't, as
happens for a moment while `retake` replaces it, is skipped and counted. In that
Moby-Dick render, most lines were names and old or British spellings, which
aren't the speaker's own. Copy only what is, such as a word other voices are
heard saying correctly.

Be careful with common words. `in = and` also passes a take that really says
"and" where the book says "in", so every allowance hides some real mistakes.
Allow only what you've heard the speaker say right.

## One output directory, one narration

An output directory holds one book in one voice, with one speed and model
precision (`--precision`). It saves them in `narration.txt` with its first take, and
later runs reuse the saved speed, so you don't have to repeat `--speed`. It also
saves the [noise limit](#noise-limit), which can change at any time. A run
with a different book, voice, speed or precision stops with an error rather than
rendering everything again, so use a new directory for those. Correcting a voice's
`.txt` file counts as a different voice, but changing its [allowances](#allowances)
doesn't.

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
