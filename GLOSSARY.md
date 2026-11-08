# Glossary

**Chapter number:** a spine document's 1-based position in the book, as printed by `chapters`. Includes front matter, so "II. The Pool of Tears" is chapter 3 of `aliceDynamic.epub`.

**Chunk:** a piece of a chapter's text short enough to narrate as one piece of audio.

**Chunk number:** `<chapter number>.<position in that chapter>`, such as `3.12`. It stays the same whichever chapters a command selects.

**Chunk limit:** how long a chunk may be: at most 500 characters, and at most 28 seconds of audio as any-tts estimates it, 2 seconds under the length above which any-tts splits a request into pieces joined by short silences: 30 seconds unless `OMNIVOICE_AUDIO_CHUNK_THRESHOLD` sets it. It comes from the voice sample: its length, what's said in it and the speed. The output directory saves it in `narration.txt` as `chunk_weight`, the most text weight a chunk may have, so `mishearings` finds the chunks without the voice sample. A directory with no `chunk_weight` has chunks limited only by characters. It isn't a render setting: when `book` or `retake` works out a different limit, it saves the new one and says so, and chunks whose text changes are generated again.

**Take:** one rendering of a chunk at one seed.

**Missing chunk:** a chunk with no audio in the output directory. Only `book` fills these. `retake` records only the chunks it's told to.

**Output directory:** holds one narration of one book, in one voice, with one set of render settings. It saves them on first use and refuses a later run that differs. Its M4B and report cover every chapter rendered there so far.

**Render settings:** settings that change a chunk's audio, such as speed and model precision (`--precision`). Changing one needs a new output directory. Gaps aren't render settings, because they only affect assembly and can change freely.

**Shared chunk:** a chunk whose text is identical to another's. They share one audio file, so a `retake` of one changes all of them, and it prints each chunk number affected.

**Floor:** a take's background level, measured as its quietest 5% of 50 ms frames. Takes below 6 seconds aren't judged on it, because they barely pause. A voice sample's floor is measured the same way on its first use, since takes copy it.

**Floor ignoring digital silence:** the floor of a take's frames that aren't exact digital silence. Takes start with about 0.1 s of zeros, which in a short take are more than its quietest 5%, so the floor would be the zeros rather than the room tone. Assembly trims each take against this floor, keeping what rises well above it and the soft edges joined to that, so a voice's room tone doesn't lengthen the gaps.

**Noise limit:** the highest floor a take can have and still pass, -55 dB unless `--noise-limit` sets it. The output directory saves it in `narration.txt`, so it's given once, and giving a new value replaces the saved one. It isn't a render setting: it decides which take is kept, not how a take sounds, so changing it keeps the audio already rendered.

**Take policy:** how many takes of a chunk `book` and `retake` render and which they keep: the first good enough unless `--takes` asks for the best of a set number. What's good enough depends on the takes' goal: for `book` a take that passes, of up to 5, and for `retake` a flawless one with its phrases heard, of up to 6 new ones. Like the noise limit, it's saved in `narration.txt` and isn't a render setting.

**Goal:** what a chunk's takes are rendered for, which the take policy follows: narrating it for `book`, or a `retake` of it, which may ask for phrases from the book to be heard. It decides what's good enough, how takes compare, and how many the default policy tries.

**Pitch range:** the span of a take's voiced pitch from the 5th to the 95th percentile, in semitones, tracked with pYIN at 16 kHz. A flat take has a narrow one. Takes with too little voiced speech, under 20 frames of 10 ms, have none. Given a target with `--pitch-range`, the take that passes nearest it is kept.

**Nearness:** how near a take's pitch range is to the target: the smaller of the two ratios between range and target, 1 when exact. It ranks in steps of 0.05, so takes in the same step are ranked by how they're heard. It only ranks takes that pass. Like the take policy, the target is saved in `narration.txt` and isn't a render setting.

**Keep settings:** the noise limit, take policy and pitch range target: what decides which take of a chunk is kept, rather than how it sounds, so none of them is a render setting. Each one given replaces the one saved in `narration.txt`, and each one not given is the saved one, or its default in a new output directory. A run settles them before loading the voice, since the voice sample is judged against the noise limit.

**Voice sample:** the recording a narration's voice is cloned from, such as `voices/me.wav`, with what's said in it in a `.txt` file beside it. `clean` makes one from a noisy recording.

**Checker:** the model that transcribes each take, so it can be compared with the text: Parakeet TDT 0.6b v3. Each chunk's check file records the checker that heard its transcript. A check file with no checker recorded holds a transcript heard by Whisper, which transcribes voice samples and the recordings `sections` looks through.

**Transcript:** what a model heard said in some audio, as text. A take's transcript is the checker's, kept in the chunk's check file and shown in `report.tsv`; the checker also gives when each word was said.

**Section:** a stretch of a recording, from one pause to another, that `sections` suggests making into a voice sample.

**Reduction:** the most noise `clean` removes from a recording when making a voice sample, in dB, 15 unless `--reduction` sets it. It's DeepFilterNet's attenuation limit. It changes the voice sample, unlike the floor and noise limit, which measure and judge takes. A higher reduction gives quieter takes but a clone less like the speaker.

**Allowance:** a word of the book that a voice may be heard saying as another, such as `in = and` for a speaker whose "in" the checker hears as "and". It's one-way, and kept beside the voice sample in a `.allowances` file, such as `voices/me.allowances`. It isn't part of the voice: changing it keeps the audio already rendered and only changes which takes pass from then on. `mishearings` suggests candidates.
