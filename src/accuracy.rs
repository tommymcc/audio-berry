//! Compares what was heard in a take with the text it should say, word by word,
//! forgiving the differences in spelling a transcriber makes, and the words a
//! voice's allowances say it hears as others.

use crate::numbers;
use std::collections::{HashMap, HashSet};

/// Passages the user approved by ear scored 0.98-1.00 recall with 0.00-0.02 extra.
const MIN_RECALL: f32 = 0.92;
const MAX_EXTRA: f32 = 0.12;

/// The fewest words in a row of the text that count as a moved phrase when
/// heard somewhere else. See `moved_phrases`. On 5,053 stored Whisper
/// transcripts, one word flagged 76 chunks, mostly a common word such as "and"
/// missed in one place and heard in another; two flagged one chunk, a real move.
const MIN_MOVED_WORDS: usize = 2;

pub struct Accuracy {
    /// Share of the text's words heard, in order.
    pub recall: f32,
    /// Words heard that aren't in the text, relative to the text's length.
    pub extra: f32,
    /// Each phrase of the text heard in another place, as its words were
    /// compared: lowercase, with numbers under 100 spelled out.
    pub moved: Vec<String>,
    pub transcript: String,
}

impl Accuracy {
    /// Heard well enough, with no phrase in the wrong place.
    pub fn acceptable(&self) -> bool {
        self.recall >= MIN_RECALL && self.extra <= MAX_EXTRA && self.in_order()
    }

    pub fn score(&self) -> f32 {
        self.recall - self.extra
    }

    /// Every word heard in its place and nothing more, give or take a rounding error.
    pub fn exact(&self) -> bool {
        self.recall >= 0.99 && self.extra <= 0.01 && self.in_order()
    }

    /// Whether no phrase of the text was heard in another place.
    pub fn in_order(&self) -> bool {
        self.moved.is_empty()
    }

    /// The moved phrases for `report.tsv`, each in quotes, or nothing.
    pub fn moved_note(&self) -> String {
        self.moved.iter().map(|phrase| format!("\"{phrase}\"")).collect::<Vec<_>>().join("; ")
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

/// Lowercases into words, with commas between thousands dropped and numbers
/// under 100 spelled out, as `numbers` writes them.
fn words(text: &str) -> Vec<String> {
    let cleaned: String = numbers::join_thousands(&text.to_lowercase())
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' { c } else { ' ' })
        .collect();
    let mut out = Vec::new();
    for word in cleaned.split_whitespace() {
        let word = word.trim_matches('\'');
        match numbers::spell_under_100(word) {
            Some(spelled) => out.extend(spelled.into_iter().map(String::from)),
            None if !word.is_empty() => out.push(word.to_string()),
            None => {}
        }
    }
    out
}

/// The words of two texts, with each number of 100 or more that one spells out
/// and the other has in digits put in digits.
fn words_of_both(first: &str, second: &str) -> (Vec<String>, Vec<String>) {
    let (first, second) = (words(first), words(second));
    (numbers::match_digits(&first, &second), numbers::match_digits(&second, &first))
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
    let (expected, heard) = words_of_both(text, transcript);
    let expected = join_split_words(&expected, &heard);
    let heard = join_split_words(&heard, &expected);
    (expected, heard)
}

/// How much of `text` was heard in `transcript`, and how much more besides. A
/// word heard as another that `allowances` allow counts as heard.
pub fn compare(text: &str, transcript: &str, allowances: &Allowances) -> Accuracy {
    let (expected, heard) = aligned_words(text, transcript);
    let table = common_subsequences(&expected, &heard, allowances);
    let matched = table[0][0];
    let length = expected.len().max(1) as f32;
    Accuracy {
        recall: matched as f32 / length,
        extra: heard.len().saturating_sub(matched) as f32 / length,
        moved: moved_phrases(&expected, &heard, &table, allowances),
        transcript: transcript.to_string(),
    }
}

/// Which words of `expected` and of `heard` the longest common subsequence in
/// `table` pairs up.
fn paired(
    expected: &[String],
    heard: &[String],
    table: &[Vec<usize>],
    allowances: &Allowances,
) -> (Vec<bool>, Vec<bool>) {
    let (mut expected_paired, mut heard_paired) = (vec![false; expected.len()], vec![false; heard.len()]);
    let (mut i, mut j) = (0, 0);
    while i < expected.len() && j < heard.len() {
        if expected[i] == heard[j] || allowances.allows(&expected[i], &heard[j]) {
            expected_paired[i] = true;
            heard_paired[j] = true;
            i += 1;
            j += 1;
        } else if table[i + 1][j] >= table[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    (expected_paired, heard_paired)
}

/// The phrases of `expected` heard in another place: runs of at least
/// `MIN_MOVED_WORDS` words that the common subsequence in `table` leaves out on
/// both sides, because they were heard in the same order as written but not in
/// the same place relative to the words around them. Words match as `table`'s
/// do, through the allowances. The longest such run is taken first, and each
/// word belongs to one run at most. They come in the text's order.
fn moved_phrases(expected: &[String], heard: &[String], table: &[Vec<usize>], allowances: &Allowances) -> Vec<String> {
    let (mut expected_used, mut heard_used) = paired(expected, heard, table, allowances);
    // Each phrase, after where it starts in `expected`.
    let mut moved = Vec::new();
    loop {
        // run[i + 1][j + 1]: how many unpaired words in a row match, ending at
        // `expected[i]` and `heard[j]`.
        let mut run = vec![vec![0usize; heard.len() + 1]; expected.len() + 1];
        let (mut longest, mut end) = (0, (0, 0));
        for (i, word) in expected.iter().enumerate() {
            for (j, other) in heard.iter().enumerate() {
                let free = !expected_used[i] && !heard_used[j];
                if free && (word == other || allowances.allows(word, other)) {
                    run[i + 1][j + 1] = run[i][j] + 1;
                    if run[i + 1][j + 1] > longest {
                        longest = run[i + 1][j + 1];
                        end = (i + 1, j + 1);
                    }
                }
            }
        }
        if longest < MIN_MOVED_WORDS {
            moved.sort();
            return moved.into_iter().map(|(_, phrase)| phrase).collect();
        }
        let (i, j) = end;
        expected_used[i - longest..i].fill(true);
        heard_used[j - longest..j].fill(true);
        moved.push((i - longest, expected[i - longest..i].join(" ")));
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
    let (transcript, phrase) = words_of_both(transcript, phrase);
    let matches = |window: &[String]| {
        phrase.iter().zip(window).all(|(written, heard)| written == heard || allowances.allows(written, heard))
    };
    !phrase.is_empty() && transcript.windows(phrase.len()).any(matches)
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
    fn numbers_match_whether_written_in_digits_or_words() {
        // As Parakeet heard numbers written in digits, sometimes in words and
        // sometimes in digits, with its own spacing.
        for (text, heard) in [
            ("Chapter 12.", "Chapter twelve."),
            ("In 1066 he paid 45 pounds for 250 sheep.", "In ten sixty-six he paid forty-five pounds for two hundred and fifty sheep."),
            ("In 1820 there were 3,000 barrels.", "In eighteen twenty there were three thousand barrels."),
            ("There were 3,000 barrels.", "There were 3 ,000 barrels."),
            ("There were 3,000 barrels.", "There were 3000 barrels."),
            ("It was 1905.", "It was nineteen oh five."),
            ("It was 1800.", "It was eighteen hundred."),
            ("He had 2,500 men and 101 guns.", "He had two thousand five hundred men and one hundred and one guns."),
            ("the 2nd time, the 21st man, the 3rd of May", "the second time, the twenty-first man, the third of May"),
            ("A hundred and fifty men.", "150 men."),
            ("Seventeen 100-pound bags.", "Seventeen hundred pound bags."),
            ("Ten twenty men.", "Ten twenty men."),
            ("At eleven fifteen.", "At eleven fifteen."),
        ] {
            let accuracy = compare(text, heard, &none());
            assert_eq!((accuracy.recall, accuracy.extra), (1.0, 0.0), "{text} / {heard}");
        }
    }

    #[test]
    fn small_numbers_in_a_row_stay_apart() {
        // Counting isn't a number of its own, and twenty-one isn't a year.
        assert_eq!(words("one, two, three"), ["one", "two", "three"]);
        assert_eq!(words("twenty-one guns"), ["twenty", "one", "guns"]);
        assert_eq!(words("21 guns"), ["twenty", "one", "guns"]);
        assert!(compare("one hundred", "one two", &none()).recall < 1.0);
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

    /// Chunk 3.16 of The Coral Island, and what was heard: the last phrase said
    /// before the one it follows in the book, after a word or two leaked in.
    const CORAL_TEXT: &str = "Now, while engaged in the coasting trade, I fell in with many seamen who had travelled \
        to almost every quarter of the globe; and I freely confess that my heart glowed ardently within me as they \
        recounted their wild adventures in foreign lands, the dreadful storms they had weathered, the appalling \
        dangers they had escaped, the wonderful creatures they had seen both on the land and in the sea,";
    const CORAL_HEARD: &str = "Very tough. Now, while engaged in the coasting trade, I fell in with many seamen who \
        had travelled to almost every quarter of the globe. And I freely confessed that my heart glowed ardently \
        within me as they recounted their wild adventures in foreign lands. The dreadful storms they had weathered, \
        the appalling dangers they had escaped, both on the land and in the sea, the wonderful creatures they \
        had seen,";

    #[test]
    fn a_phrase_heard_in_the_wrong_place_is_found() {
        let accuracy = compare(CORAL_TEXT, CORAL_HEARD, &none());
        assert_eq!(accuracy.moved, ["the wonderful creatures they had seen"]);
        assert_eq!(accuracy.moved_note(), "\"the wonderful creatures they had seen\"");
        assert!(!accuracy.in_order());
        assert!(!accuracy.acceptable() && !accuracy.exact());
    }

    #[test]
    fn a_moved_phrase_fails_a_take_that_would_otherwise_pass() {
        let text = "The boy ran down to the shore at dawn, and he looked out across the water for a long while, \
            waiting for the boat that his father had promised would come back before the end of the summer.";
        let heard = "The boy ran down to the shore at dawn, and he looked out for a long while across the water, \
            waiting for the boat that his father had promised would come back before the end of the summer.";
        let accuracy = compare(text, heard, &none());
        assert!(accuracy.recall >= MIN_RECALL && accuracy.extra <= MAX_EXTRA, "{} {}", accuracy.recall, accuracy.extra);
        // The longer phrase keeps its place, so the shorter is the one moved.
        assert_eq!(accuracy.moved, ["across the water"]);
        assert!(!accuracy.acceptable());
    }

    #[test]
    fn words_heard_in_order_move_nothing() {
        for (text, heard) in [
            ("the quick brown fox jumps over the lazy dog", "the quick brown fox jumps over the lazy dog"),
            // Repeated and dropped words, as Whisper hears some takes.
            ("the quick brown fox jumps over the lazy dog", "the quick brown the quick brown fox jumps over the dog"),
            ("the quick brown fox jumps over the lazy dog", "the quick brown fox jumps over the lazy dog the lazy dog"),
            ("the quick brown fox jumps over the lazy dog", "quick fox over lazy"),
            // A single word heard out of place is too little to call a phrase.
            ("he said that the boat was gone", "said he that the boat was gone"),
        ] {
            let accuracy = compare(text, heard, &none());
            assert!(accuracy.in_order(), "{heard}: {:?}", accuracy.moved);
            assert_eq!(accuracy.moved_note(), "");
        }
    }

    #[test]
    fn each_moved_phrase_is_named() {
        let text = "one day the old man went down to the river and there he sat in the sun until the evening came";
        let heard = "the old man one day went down to the river and there he sat until the evening came in the sun";
        let accuracy = compare(text, heard, &none());
        assert_eq!(accuracy.moved, ["one day", "in the sun"]);
        assert_eq!(accuracy.moved_note(), "\"one day\"; \"in the sun\"");
    }

    #[test]
    fn a_moved_phrase_is_matched_through_allowances_and_numbers() {
        let text = "At the end of the day he paid 45 pounds for the old grey horse and rode it home.";
        let heard = "At the end of the day he paid for the old grey horse forty-five pounds and rode it home.";
        let accuracy = compare(text, heard, &none());
        assert_eq!(accuracy.moved, ["forty five pounds"]);
        let text = "He sat quite still all through the night in the boat with his hands upon the oars.";
        let heard = "He sat quite still and the boat all through the night with his hands upon the oars.";
        assert_eq!(compare(text, heard, &allow("in = and")).moved, ["in the boat"]);
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
