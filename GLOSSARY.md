# Glossary

**Chapter number:** a spine document's 1-based position in the book, as printed by `chapters`. Includes front matter, so "II. The Pool of Tears" is chapter 3 of `aliceDynamic.epub`.

**Chunk:** a piece of a chapter's text short enough to narrate as one piece of audio.

**Chunk number:** `<chapter number>.<position in that chapter>`, such as `3.12`. It stays the same whichever chapters a command selects.

**Take:** one rendering of a chunk at one seed.

**Missing chunk:** a chunk with no audio in the output directory. Only `book` fills these. `retake` records only the chunks it's told to.

**Output directory:** holds one narration of one book, in one voice, with one set of render settings. It saves them on first use and refuses a later run that differs. Its M4B and report cover every chapter rendered there so far.

**Render settings:** settings that change a chunk's audio, such as speed and model precision (`OMNIVOICE_DTYPE`). Changing one needs a new output directory. Gaps aren't render settings, because they only affect assembly and can change freely.

**Shared chunk:** a chunk whose text is identical to another's. They share one audio file, so a `retake` of one changes all of them, and it prints each chunk number affected.

**Floor:** a take's background level, measured as its quietest 5% of 50 ms frames. Takes below 6 seconds aren't judged on it, because they barely pause. A voice sample's floor is measured the same way on its first use, since takes copy it.

**Noise limit:** the highest floor a take can have and still pass, -55 dB unless `AUDIO_BERRY_NOISE_LIMIT_DB` sets it. The output directory saves it in `narration.txt`, so it's given once, and giving a new value replaces the saved one. It isn't a render setting: it decides which take is kept, not how a take sounds, so changing it keeps the audio already rendered.

**Voice sample:** the recording a narration's voice is cloned from, such as `voices/me.wav`, with what's said in it in a `.txt` file beside it. `clean` makes one from a noisy recording.

**Section:** a stretch of a recording, from one pause to another, that `sections` suggests making into a voice sample.

**Reduction:** the most noise `clean` removes from a recording when making a voice sample, in dB, 15 unless `--reduction` sets it. It's DeepFilterNet's attenuation limit. It changes the voice sample, unlike the floor and noise limit, which measure and judge takes. A higher reduction gives quieter takes but a clone less like the speaker.
