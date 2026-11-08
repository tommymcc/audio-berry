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

/// Lowercases into words, so that a number matches however it's written:
/// numbers under 100 are spelled out, so "21" and "21st" match "twenty-one" and
/// "twenty-first", and numbers spelled out from 100 up are put in digits, so
/// "two hundred and fifty" matches "250" and "ten sixty-six" matches "1066".
fn words(text: &str) -> Vec<String> {
    let cleaned: String = join_thousands(&text.to_lowercase())
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' { c } else { ' ' })
        .collect();
    let mut out = Vec::new();
    for word in cleaned.split_whitespace() {
        let word = word.trim_matches('\'');
        let (digits, suffix) = word.split_at(word.find(|c: char| !c.is_ascii_digit()).unwrap_or(word.len()));
        let ordinal = matches!(suffix, "st" | "nd" | "rd" | "th");
        match digits.parse::<usize>() {
            Ok(number) if number < 100 && (suffix.is_empty() || ordinal) => spell(number, ordinal, &mut out),
            _ if !word.is_empty() => out.push(word.to_string()),
            _ => {}
        }
    }
    numbers_in_digits(out)
}

/// Removes the commas between thousands, so "3,000" is one number, as is
/// "3 ,000", as Parakeet sometimes writes it.
fn join_thousands(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (index, &c) in chars.iter().enumerate() {
        let after_digit = out.trim_end().ends_with(|c: char| c.is_ascii_digit());
        let digits_after = chars.get(index + 1..index + 4).is_some_and(|next| next.iter().all(char::is_ascii_digit))
            && !chars.get(index + 4).is_some_and(char::is_ascii_digit);
        if c == ',' && after_digit && digits_after {
            out.truncate(out.trim_end().len());
        } else {
            out.push(c);
        }
    }
    out
}

/// Spells out a number under 100, as an ordinal if `ordinal`.
fn spell(number: usize, ordinal: bool, out: &mut Vec<String>) {
    let ones = |number: usize| if ordinal { ORDINAL_ONES[number] } else { ONES[number] };
    if number < 20 {
        out.push(ones(number).to_string());
    } else if number % 10 == 0 {
        let tens = number / 10 - 2;
        out.push(if ordinal { ORDINAL_TENS[tens] } else { TENS[tens] }.to_string());
    } else {
        out.push(TENS[number / 10 - 2].to_string());
        out.push(ones(number % 10).to_string());
    }
}

const ORDINAL_ONES: [&str; 20] = [
    "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth",
];
const ORDINAL_TENS: [&str; 8] =
    ["twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth"];

/// Replaces each number spelled out in `words` that comes to 100 or more with its
/// digits. Smaller ones stay spelled out, as `words` spells them.
fn numbers_in_digits(words: Vec<String>) -> Vec<String> {
    let mut out = Vec::with_capacity(words.len());
    let mut index = 0;
    while index < words.len() {
        match spelled_number(&words[index..]) {
            Some((value, length)) if value >= 100 => {
                out.push(value.to_string());
                index += length;
            }
            _ => {
                out.push(words[index].clone());
                index += 1;
            }
        }
    }
    out
}

/// The value of a number under 100 spelled out at the start of `words`, such as
/// "forty five", and how many words it takes.
fn spelled_under_100(words: &[String]) -> Option<(u64, usize)> {
    let position = |list: &[&str], word: &String| list.iter().position(|item| item == word);
    let first = words.first()?;
    if let Some(tens) = position(&TENS, first) {
        let tens = 10 * (tens as u64 + 2);
        return match words.get(1).and_then(|word| position(&ONES, word)) {
            Some(ones @ 1..=9) => Some((tens + ones as u64, 2)),
            _ => Some((tens, 1)),
        };
    }
    position(&ONES, first).map(|number| (number as u64, 1))
}

/// The value of a number spelled out at the start of `words`, and how many words
/// it takes: a cardinal such as "two thousand five hundred and one", or a year
/// such as "eighteen twenty" or "nineteen oh five".
fn spelled_number(words: &[String]) -> Option<(u64, usize)> {
    let (mut total, mut current) = (0, 0);
    let mut index = 0;
    // Whether `current` has a number under 100 since the last hundred or thousand.
    let mut small_since_scale = false;
    while index < words.len() {
        let word = words[index].as_str();
        match word {
            // "hundred" on its own counts as "a hundred".
            "hundred" if current < 100 => {
                current = current.max(1) * 100;
                small_since_scale = false;
            }
            "thousand" if total == 0 => {
                total = current.max(1) * 1000;
                current = 0;
                small_since_scale = false;
            }
            "and" if index > 0 && !small_since_scale && current + total >= 100 => {
                match spelled_under_100(&words[index + 1..]) {
                    Some(_) => {}
                    None => break,
                }
            }
            _ => match spelled_under_100(&words[index..]) {
                Some((value, length)) if !small_since_scale => {
                    current += value;
                    small_since_scale = true;
                    index += length;
                    continue;
                }
                _ => break,
            },
        }
        index += 1;
    }
    if index == 0 {
        return None;
    }
    let value = total + current;
    // A year said as two pairs of digits, after a first pair with nothing else.
    if total == 0 && (10..100).contains(&current) && small_since_scale {
        let rest = &words[index..];
        let second = match rest.first().map(String::as_str) {
            Some("oh") => spelled_under_100(&rest[1..]).filter(|(ones, length)| *ones < 10 && *length == 1).map(|(ones, _)| (ones, 2)),
            _ => spelled_under_100(rest).filter(|(pair, _)| *pair >= 10),
        };
        if let Some((pair, length)) = second {
            return Some((current * 100 + pair, index + length));
        }
    }
    Some((value, index))
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
            ("A hundred and fifty men.", "A 150 men."),
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

    #[test]
    fn a_phrase_counts_as_heard_through_an_allowance() {
        let allowances = allow("in = and");
        assert!(heard_phrase("He sat, and the boat.", "in the boat", &allowances));
        assert!(!heard_phrase("He sat, and the boat.", "in the boat", &none()));
        assert!(!heard_phrase("He sat, in the boat.", "and the boat", &allowances));
        assert!(!heard_phrase("He sat, and the boat.", "", &allowances));
    }
}
