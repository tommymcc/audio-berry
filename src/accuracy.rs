//! Compares what was heard in a take with the text it should say, word by word,
//! forgiving the differences in spelling a transcriber makes, and the words a
//! voice's allowances say it hears as others.

use std::collections::{HashMap, HashSet};

/// Passages the user approved by ear scored 0.98-1.00 recall with 0.00-0.02 extra.
const MIN_RECALL: f32 = 0.92;
const MAX_EXTRA: f32 = 0.12;

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 8] = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

pub struct Accuracy {
    /// Share of the text's words heard, in order.
    pub recall: f32,
    /// Words heard that aren't in the text, relative to the text's length.
    pub extra: f32,
    pub transcript: String,
}

impl Accuracy {
    pub fn acceptable(&self) -> bool {
        self.recall >= MIN_RECALL && self.extra <= MAX_EXTRA
    }

    pub fn score(&self) -> f32 {
        self.recall - self.extra
    }

    /// Every word heard and nothing more, give or take a rounding error.
    pub fn exact(&self) -> bool {
        self.recall >= 0.99 && self.extra <= 0.01
    }
}

/// Words the transcriber hears as other words in one voice, such as a speaker
/// whose "in" is heard as "and". Each is one-way: the text's word may be heard
/// as the other, not the reverse.
#[derive(Clone, Debug, Default)]
pub struct Allowances {
    /// What each text word may be heard as, all normalised as `words` does.
    heard_as: HashMap<String, HashSet<String>>,
}

/// A word of the text and a word heard in its place.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WordPair {
    pub written: String,
    pub heard: String,
}

impl WordPair {
    fn new(written: &str, heard: &str) -> WordPair {
        WordPair { written: written.to_string(), heard: heard.to_string() }
    }
}

impl Allowances {
    /// Reads lines like `in = and`, meaning the text's "in" may be heard as "and".
    /// `#` starts a comment, and blank lines are skipped. Errors name the line.
    pub fn parse(text: &str) -> Result<Allowances, String> {
        let mut heard_as: HashMap<String, HashSet<String>> = HashMap::new();
        for (index, line) in text.lines().enumerate() {
            let number = index + 1;
            let line = line.split('#').next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }
            let (written, heard) = line
                .split_once('=')
                .ok_or_else(|| format!("line {number}: \"{line}\" isn't in the form \"in = and\""))?;
            let one_word = |side: &str| match words(side).as_slice() {
                [word] => Ok(word.clone()),
                _ => Err(format!("line {number}: \"{}\" isn't one word", side.trim())),
            };
            let (written, heard) = (one_word(written)?, one_word(heard)?);
            if written == heard {
                return Err(format!("line {number}: \"{written}\" is the same word on both sides"));
            }
            heard_as.entry(written).or_default().insert(heard);
        }
        Ok(Allowances { heard_as })
    }

    /// Whether the text's `written` word may be heard as `heard`, both normalised
    /// as `words` does.
    pub fn allows(&self, written: &str, heard: &str) -> bool {
        !self.heard_as.is_empty() && self.heard_as.get(written).is_some_and(|heard_as| heard_as.contains(heard))
    }

    /// How many pairs there are.
    pub fn len(&self) -> usize {
        self.heard_as.values().map(HashSet::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.heard_as.is_empty()
    }
}

/// Lowercases into words, spelling out small numbers so "1" matches "one".
fn words(text: &str) -> Vec<String> {
    let cleaned: String = text
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' { c } else { ' ' })
        .collect();
    let mut out = Vec::new();
    for word in cleaned.split_whitespace() {
        let word = word.trim_matches('\'');
        match word.parse::<usize>() {
            Ok(number) if number < 20 => out.push(ONES[number].to_string()),
            Ok(number) if number < 100 => {
                out.push(TENS[number / 10 - 2].to_string());
                if number % 10 != 0 {
                    out.push(ONES[number % 10].to_string());
                }
            }
            _ if !word.is_empty() => out.push(word.to_string()),
            _ => {}
        }
    }
    out
}

/// How many words, in order, `expected[i..]` and `heard[j..]` have in common, as
/// `table[i][j]`. A word matches itself, or a word the allowances let it be heard as.
fn common_subsequences(expected: &[String], heard: &[String], allowances: &Allowances) -> Vec<Vec<usize>> {
    let mut table = vec![vec![0usize; heard.len() + 1]; expected.len() + 1];
    for (i, word) in expected.iter().enumerate().rev() {
        for (j, other) in heard.iter().enumerate().rev() {
            table[i][j] = if word == other || allowances.allows(word, other) {
                table[i + 1][j + 1] + 1
            } else {
                table[i + 1][j].max(table[i][j + 1])
            };
        }
    }
    table
}

/// Joins neighbouring words where `other` has them as one word, so "quarter
/// deck" matches "quarterdeck" and "e book" matches "ebook".
fn join_split_words(words: &[String], other: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(words.len());
    let mut index = 0;
    while index < words.len() {
        if let Some(next) = words.get(index + 1) {
            let joined = format!("{}{next}", words[index]);
            if other.contains(&joined) {
                out.push(joined);
                index += 2;
                continue;
            }
        }
        out.push(words[index].clone());
        index += 1;
    }
    out
}

/// The words of a text and of its transcript, ready to line up.
fn aligned_words(text: &str, transcript: &str) -> (Vec<String>, Vec<String>) {
    let (expected, heard) = (words(text), words(transcript));
    let expected = join_split_words(&expected, &heard);
    let heard = join_split_words(&heard, &expected);
    (expected, heard)
}

/// How much of `text` was heard in `transcript`, and how much more besides. A
/// word heard as another that `allowances` allow counts as heard.
pub fn compare(text: &str, transcript: &str, allowances: &Allowances) -> Accuracy {
    let (expected, heard) = aligned_words(text, transcript);
    let matched = common_subsequences(&expected, &heard, allowances)[0][0];
    let length = expected.len().max(1) as f32;
    Accuracy {
        recall: matched as f32 / length,
        extra: heard.len().saturating_sub(matched) as f32 / length,
        transcript: transcript.to_string(),
    }
}

/// Where one word of a text was heard as one other word.
pub struct Substitutions {
    /// The text's words as they were compared, with any split words joined.
    pub text_words: Vec<String>,
    /// Each place one word was heard as one other, such as "in" heard as "and".
    /// Words missed or added, and several words heard for one, aren't included.
    pub found: Vec<WordPair>,
}

/// Lines `transcript` up with `text` and finds the words heard as others.
pub fn substitutions(text: &str, transcript: &str) -> Substitutions {
    let (expected, heard) = aligned_words(text, transcript);
    let table = common_subsequences(&expected, &heard, &Allowances::default());
    let mut found = Vec::new();
    // The words between two matches, on each side.
    let (mut missed, mut instead): (Vec<&String>, Vec<&String>) = (Vec::new(), Vec::new());
    let (mut i, mut j) = (0, 0);
    loop {
        let at_end = i == expected.len() || j == heard.len();
        if at_end {
            missed.extend(&expected[i..]);
            instead.extend(&heard[j..]);
        }
        if at_end || expected[i] == heard[j] {
            if let ([written], [other]) = (missed.as_slice(), instead.as_slice()) {
                found.push(WordPair::new(written, other));
            }
            missed.clear();
            instead.clear();
            if at_end {
                return Substitutions { text_words: expected, found };
            }
            i += 1;
            j += 1;
        } else if table[i + 1][j] >= table[i][j + 1] {
            missed.push(&expected[i]);
            i += 1;
        } else {
            instead.push(&heard[j]);
            j += 1;
        }
    }
}

/// Whether `text` has the phrase's words in a row, ignoring case and punctuation.
pub fn contains_phrase(text: &str, phrase: &str) -> bool {
    let phrase = words(phrase);
    !phrase.is_empty() && words(text).windows(phrase.len()).any(|window| window == phrase)
}

/// Whether the phrase's words were heard in a row in `transcript`, counting the
/// words `allowances` let be heard as others.
pub fn heard_phrase(transcript: &str, phrase: &str, allowances: &Allowances) -> bool {
    let phrase = words(phrase);
    let matches = |window: &[String]| {
        phrase.iter().zip(window).all(|(written, heard)| written == heard || allowances.allows(written, heard))
    };
    !phrase.is_empty() && words(transcript).windows(phrase.len()).any(matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none() -> Allowances {
        Allowances::default()
    }

    fn allow(lines: &str) -> Allowances {
        Allowances::parse(lines).unwrap()
    }

    fn pairs(found: &[(&str, &str)]) -> Vec<WordPair> {
        found.iter().map(|(written, heard)| WordPair::new(written, heard)).collect()
    }

    #[test]
    fn matching_ignores_case_punctuation_and_digit_style() {
        let accuracy = compare("Chapter 12. Down the Rabbit-Hole!", "chapter twelve, down the rabbit hole", &none());
        assert_eq!(accuracy.recall, 1.0);
        assert_eq!(accuracy.extra, 0.0);
        assert!(accuracy.acceptable());
    }

    #[test]
    fn words_joined_on_one_side_and_split_on_the_other_match() {
        // Whisper's spelling of compounds often differs from the book's.
        for (text, heard) in [
            ("Chapter 36. The Quarter-Deck.", "chapter thirty six, the quarterdeck"),
            ("This eBook is for the use of anyone", "this e-book is for the use of anyone"),
            ("a rose bud in the grub worm's way", "a rosebud in the grubworm's way"),
        ] {
            let accuracy = compare(text, heard, &none());
            assert_eq!((accuracy.recall, accuracy.extra), (1.0, 0.0), "{text} / {heard}");
        }
        // Joining is only forgiven when the joined word is really there.
        assert!(compare("the quarter deck", "the quarter", &none()).recall < 1.0);
    }

    #[test]
    fn phrases_match_whole_words_in_order() {
        assert!(contains_phrase("It took a Watch, out of its pocket", "took a watch out"));
        assert!(!contains_phrase("took a watt out", "took a watch"));
        assert!(!contains_phrase("watchman", "watch"));
    }

    #[test]
    fn nonsense_and_repeats_are_rejected() {
        let text = "the quick brown fox jumps over the lazy dog";
        assert!(!compare(text, "in what trade setups and cost free workers", &none()).acceptable());
        let repeated = format!("{text} {text}");
        let accuracy = compare(text, &repeated, &none());
        assert_eq!(accuracy.recall, 1.0);
        assert!(!accuracy.acceptable());
    }

    #[test]
    fn allowances_are_read_a_pair_a_line_and_normalised_like_the_words_compared() {
        let allowances = allow("# Dad's accent\n\nIn = AND   # heard all through Moby-Dick\n  at=had\n7 = 'seven's'\n");
        assert_eq!(allowances.len(), 3);
        assert!(allowances.allows("in", "and"));
        assert!(allowances.allows("at", "had"));
        assert!(allowances.allows("seven", "seven's"));
        assert!(allow("# nothing yet\n\n").is_empty());
        assert_eq!(allow("in = and\nin = an\nIN = and").len(), 2);
    }

    #[test]
    fn an_allowance_is_one_way() {
        let allowances = allow("in = and");
        assert!(allowances.allows("in", "and"));
        assert!(!allowances.allows("and", "in"));
    }

    #[test]
    fn a_bad_allowance_line_is_an_error_that_names_it() {
        for (text, wanted) in [
            ("in = and\nin and", "line 2: \"in and\" isn't in the form \"in = and\""),
            ("# two words\n\nin = and so", "line 3: \"and so\" isn't one word"),
            ("= and", "line 1: \"\" isn't one word"),
            ("in = ...", "line 1: \"...\" isn't one word"),
            ("In = in", "line 1: \"in\" is the same word on both sides"),
        ] {
            assert_eq!(Allowances::parse(text).unwrap_err(), wanted, "{text:?}");
        }
    }

    #[test]
    fn an_allowed_mishearing_counts_as_heard() {
        let text = "He sat in the boat in silence, and in the end he slept.";
        let heard = "He sat and the boat and silence, and and the end he slept.";
        assert!(!compare(text, heard, &none()).acceptable());
        let allowed = compare(text, heard, &allow("in = and"));
        assert_eq!((allowed.recall, allowed.extra), (1.0, 0.0));
    }

    #[test]
    fn an_allowance_forgives_nothing_the_other_way_round_or_for_other_words() {
        let allowances = allow("in = and");
        assert!(compare("salt and pepper", "salt in pepper", &allowances).recall < 1.0);
        assert!(compare("in the boat", "on the boat", &allowances).recall < 1.0);
    }

    #[test]
    fn substitutions_are_one_word_heard_as_one_other() {
        let found = substitutions(
            "Call me Ishmael, in the boat, by the old harpooneer.",
            "Call me Ishmael, and the boat, by the harpaneer. Amen",
        );
        // "old harpooneer." heard as "harpaneer. Amen" isn't one word for one.
        assert_eq!(found.found, pairs(&[("in", "and")]));
        let found = substitutions("Ye see the sea now, Bildad", "You see the see now, bill dad");
        assert_eq!(found.found, pairs(&[("ye", "you"), ("sea", "see")]));
        assert!(substitutions("all heard", "all heard").found.is_empty());
    }

    #[test]
    fn substitutions_give_the_text_words_as_compared() {
        let found = substitutions("The quarter deck, 12 feet", "the quarterdeck twelve feet");
        assert_eq!(found.text_words, ["the", "quarterdeck", "twelve", "feet"]);
    }

    #[test]
    fn a_phrase_counts_as_heard_through_an_allowance() {
        let allowances = allow("in = and");
        assert!(heard_phrase("He sat, and the boat.", "in the boat", &allowances));
        assert!(!heard_phrase("He sat, and the boat.", "in the boat", &none()));
        assert!(!heard_phrase("He sat, in the boat.", "and the boat", &allowances));
        assert!(!heard_phrase("He sat, and the boat.", "", &allowances));
    }
}
