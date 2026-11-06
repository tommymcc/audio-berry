# Glossary

**Chapter number:** a spine document's 1-based position in the book, as printed by `chapters`. Includes front matter, so "II. The Pool of Tears" is chapter 3 of `aliceDynamic.epub`.

**Chunk:** a piece of a chapter's text short enough to narrate as one piece of audio.

**Chunk number:** `<chapter number>.<position in that chapter>`, such as `3.12`. It stays the same whichever chapters a command selects.

**Take:** one rendering of a chunk at one seed.

**Missing chunk:** a chunk with no audio in the output directory. Only `book` fills these. `retake` records only the chunks it's told to.

**Output directory:** holds one narration of one book, in one voice, with one set of render settings. It saves them on first use and refuses a later run that differs. Its M4B and report cover every chapter rendered there so far.

**Render settings:** settings that change a chunk's audio, such as speed and model precision (`OMNIVOICE_DTYPE`). Changing one needs a new output directory. Gaps aren't render settings, because they only affect assembly and can change freely.

**Shared chunk:** a chunk whose text is identical to another's. They share one audio file, so a `retake` of one changes all of them, and it prints each chunk number affected.

**Floor:** a take's background level, measured as its quietest 5% of 50 ms frames. Takes below 6 seconds aren't judged on it, because they barely pause.

**Noise limit:** the highest floor a take can have and still pass, -55 dB unless `AUDIO_BERRY_NOISE_LIMIT_DB` sets it. The output directory saves it in `narration.txt`, so it's given once, and giving a new value replaces the saved one. It isn't a render setting: it decides which take is kept, not how a take sounds, so changing it keeps the audio already rendered.
