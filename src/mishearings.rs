//! Lists the words most often heard as other words in an output directory's
//! takes, to help write a voice's allowances. It only reads.

use crate::accuracy::{self, Allowances, WordPair};
use crate::epub::read_book;
use crate::store::NarrationStore;
use crate::voice::{allowances_path, read_allowances};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::io::ErrorKind;
use std::path::Path;

/// Shown at most, most often first.
const LISTED: usize = 30;

/// How often each text word was heard as each other word.
#[derive(Default)]
struct Tally {
    takes: usize,
    heard_as: HashMap<WordPair, usize>,
    /// How often each word is in the text, as compared, so a count can be read as a share.
    in_text: HashMap<String, usize>,
}

/// One text word heard as another, `count` times of the `in_text` times it's in the text.
#[derive(Debug, PartialEq)]
struct Mishearing {
    pair: WordPair,
    count: usize,
    in_text: usize,
}

impl Tally {
    fn add(&mut self, text: &str, transcript: &str) {
        self.takes += 1;
        let substitutions = accuracy::substitutions(text, transcript);
        for word in substitutions.text_words {
            *self.in_text.entry(word).or_default() += 1;
        }
        for pair in substitutions.found {
            *self.heard_as.entry(pair).or_default() += 1;
        }
    }

    /// Those heard more than once, most often first.
    fn most_frequent(&self, limit: usize) -> Vec<Mishearing> {
        let mut found: Vec<Mishearing> = self
            .heard_as
            .iter()
            .filter(|(_, count)| **count > 1)
            .map(|(pair, count)| Mishearing {
                pair: pair.clone(),
                count: *count,
                in_text: self.in_text.get(&pair.written).copied().unwrap_or_default(),
            })
            .collect();
        found.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.pair.cmp(&b.pair)));
        found.truncate(limit);
        found
    }
}

/// The lines to print, ready for an allowances file, marking those already allowed.
fn lines(found: &[Mishearing], allowances: &Allowances) -> Vec<String> {
    let pairs: Vec<String> = found.iter().map(|item| format!("{} = {}", item.pair.written, item.pair.heard)).collect();
    let width = pairs.iter().map(|pair| pair.chars().count()).max().unwrap_or_default();
    found
        .iter()
        .zip(&pairs)
        .map(|(item, pair)| {
            let allowed = if allowances.allows(&item.pair.written, &item.pair.heard) { ", allowed" } else { "" };
            format!("{pair:width$}  # {} of {} times{allowed}", item.count, item.in_text)
        })
        .collect()
}

/// Prints the words most often heard as others in the takes kept in `out`, as
/// lines ready for a voice's allowances file. Given the voice sample, it marks
/// those its allowances already allow.
pub fn list(epub: &Path, out: &Path, voice_sample: Option<&Path>) -> Result<(), Box<dyn Error>> {
    let book = read_book(epub)?;
    let allowances = voice_sample.map(read_allowances).transpose()?.unwrap_or_default();
    let store = NarrationStore::read(out, &book.title, &book.chunks())?;
    let mut tally = Tally::default();
    let mut seen = HashSet::new();
    let mut skipped = 0;
    for item in store.chunks().iter().filter(|item| store.has_audio(item)) {
        // Shared chunks share one take, so count it once.
        if !seen.insert(store.audio_path(item)) {
            continue;
        }
        match store.check(item) {
            Ok(check) => tally.add(&item.text, &check.transcript),
            // A retake running now moves the audio and its check file one at a time.
            Err(error) if error.downcast_ref::<std::io::Error>().is_some_and(|error| error.kind() == ErrorKind::NotFound) => {
                skipped += 1;
            }
            Err(error) => return Err(error),
        }
    }
    if skipped > 0 {
        println!("Skipped {skipped} takes with no check file, perhaps being retaken now.");
    }
    let found = tally.most_frequent(LISTED);
    if found.is_empty() {
        println!("No word was heard as the same other word twice in {} takes.", tally.takes);
        return Ok(());
    }
    println!("Words heard as other words in {} takes, most often first:", tally.takes);
    for line in lines(&found, &allowances) {
        println!("{line}");
    }
    println!(
        "Copy only the speaker's accent into the voice's allowances file, such as {}. \
         Names, spellings and real mistakes don't belong there.",
        voice_sample.map_or_else(|| allowances_path(Path::new("voices/me.wav")), allowances_path).display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mishearing(written: &str, heard: &str, count: usize, in_text: usize) -> Mishearing {
        Mishearing { pair: WordPair { written: written.into(), heard: heard.into() }, count, in_text }
    }

    #[test]
    fn mishearings_are_counted_against_how_often_the_word_is_in_the_text() {
        let mut tally = Tally::default();
        tally.add("He sat in the boat.", "He sat and the boat.");
        tally.add("In the end, salt and pepper.", "And the end, salt and pepper.");
        tally.add("Ye see the whale.", "You see the whale.");
        assert_eq!(tally.takes, 3);
        // "ye" heard as "you" only once, so it isn't listed.
        assert_eq!(tally.most_frequent(LISTED), [mishearing("in", "and", 2, 2)]);
    }

    #[test]
    fn the_text_is_counted_as_it_was_compared() {
        let mut tally = Tally::default();
        for _ in 0..2 {
            tally.add("on the quarter deck", "and the quarterdeck");
        }
        assert_eq!(tally.in_text.get("quarterdeck"), Some(&2));
        assert_eq!(tally.in_text.get("quarter"), None);
    }

    #[test]
    fn the_most_frequent_come_first_up_to_the_limit() {
        let mut tally = Tally::default();
        for _ in 0..3 {
            tally.add("at the sea", "had the see");
        }
        tally.add("at", "had");
        assert_eq!(tally.most_frequent(1), [mishearing("at", "had", 4, 4)]);
    }

    #[test]
    fn lines_line_up_by_characters_and_mark_what_is_allowed() {
        let found = [mishearing("café", "cafe", 3, 4), mishearing("in", "and", 2, 9)];
        let allowances = Allowances::parse("in = and").unwrap();
        assert_eq!(
            lines(&found, &allowances),
            ["café = cafe  # 3 of 4 times", "in = and     # 2 of 9 times, allowed"]
        );
    }
}
