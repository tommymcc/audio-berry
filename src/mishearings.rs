//! Lists the words most often heard as other words in an output directory's
//! takes, to help write a voice's allowances. It only reads.

use crate::accuracy;
use crate::book::chapter_chunks;
use crate::epub::read_book;
use crate::store::NarrationStore;
use crate::voice::allowances_path;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::Path;

/// Shown at most, most often first.
const LISTED: usize = 30;

/// How often each text word was heard as each other word.
#[derive(Default)]
struct Tally {
    takes: usize,
    heard_as: HashMap<(String, String), usize>,
    /// How often each word is in the text, so a count can be read as a share.
    in_text: HashMap<String, usize>,
}

/// One text word heard as another, `count` times of the `in_text` times it's in the text.
#[derive(Debug, PartialEq)]
struct Mishearing {
    written: String,
    heard: String,
    count: usize,
    in_text: usize,
}

impl Tally {
    fn add(&mut self, text: &str, transcript: &str) {
        self.takes += 1;
        for word in accuracy::words(text) {
            *self.in_text.entry(word).or_default() += 1;
        }
        for pair in accuracy::substitutions(text, transcript) {
            *self.heard_as.entry(pair).or_default() += 1;
        }
    }

    /// Those heard more than once, most often first.
    fn most_frequent(&self, limit: usize) -> Vec<Mishearing> {
        let mut found: Vec<Mishearing> = self
            .heard_as
            .iter()
            .filter(|(_, count)| **count > 1)
            .map(|((written, heard), count)| Mishearing {
                written: written.clone(),
                heard: heard.clone(),
                count: *count,
                in_text: self.in_text.get(written).copied().unwrap_or_default(),
            })
            .collect();
        found.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| (&a.written, &a.heard).cmp(&(&b.written, &b.heard))));
        found.truncate(limit);
        found
    }
}

/// Prints the words most often heard as others in the takes kept in `out`, as
/// lines ready for a voice's allowances file.
pub fn list(epub: &Path, out: &Path) -> Result<(), Box<dyn Error>> {
    let book = read_book(epub)?;
    let store = NarrationStore::read(out, &book.title, &chapter_chunks(&book))?;
    let mut tally = Tally::default();
    let mut seen = HashSet::new();
    for item in store.chunks().iter().filter(|item| store.has_audio(item)) {
        // Shared chunks share one take, so count it once.
        if seen.insert(store.audio_path(item)) {
            tally.add(&item.text, &store.check(item)?.transcript);
        }
    }
    let found = tally.most_frequent(LISTED);
    if found.is_empty() {
        println!("No word was heard as the same other word twice in {} takes.", tally.takes);
        return Ok(());
    }
    println!("Words heard as other words in {} takes, most often first:", tally.takes);
    let width = found.iter().map(|item| item.written.len() + item.heard.len()).max().unwrap_or_default() + 3;
    for item in &found {
        let line = format!("{} = {}", item.written, item.heard);
        println!("{line:width$}  # {} of {} times", item.count, item.in_text);
    }
    println!(
        "Copy only the speaker's accent into the voice's allowances file, such as {}. \
         Names, spellings and real mistakes don't belong there.",
        allowances_path(Path::new("voices/me.wav")).display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mishearings_are_counted_against_how_often_the_word_is_in_the_text() {
        let mut tally = Tally::default();
        tally.add("He sat in the boat.", "He sat and the boat.");
        tally.add("In the end, salt and pepper.", "And the end, salt and pepper.");
        tally.add("Ye see the whale.", "You see the whale.");
        let found = tally.most_frequent(LISTED);
        assert_eq!(tally.takes, 3);
        // "ye" heard as "you" only once, so it isn't listed.
        assert_eq!(
            found,
            [Mishearing { written: "in".into(), heard: "and".into(), count: 2, in_text: 2 }]
        );
    }

    #[test]
    fn the_most_frequent_come_first_up_to_the_limit() {
        let mut tally = Tally::default();
        for _ in 0..3 {
            tally.add("at the sea", "had the see");
        }
        tally.add("at", "had");
        let found = tally.most_frequent(1);
        assert_eq!(found.len(), 1);
        assert_eq!((found[0].written.as_str(), found[0].heard.as_str(), found[0].count), ("at", "had", 4));
    }
}
