//! Compares what was heard in a take with the text it should say, word by word,
//! forgiving the differences in spelling a transcriber makes.

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

fn common_subsequence(a: &[String], b: &[String]) -> usize {
    let mut row = vec![0usize; b.len() + 1];
    for word in a {
        let mut diagonal = 0;
        for (j, other) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = if word == other {
                diagonal + 1
            } else {
                above.max(row[j])
            };
            diagonal = above;
        }
    }
    row[b.len()]
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

/// How much of `text` was heard in `transcript`, and how much more besides.
pub fn compare(text: &str, transcript: &str) -> Accuracy {
    let (expected, heard) = (words(text), words(transcript));
    let expected = join_split_words(&expected, &heard);
    let heard = join_split_words(&heard, &expected);
    let matched = common_subsequence(&expected, &heard);
    let length = expected.len().max(1) as f32;
    Accuracy {
        recall: matched as f32 / length,
        extra: heard.len().saturating_sub(matched) as f32 / length,
        transcript: transcript.to_string(),
    }
}

/// Whether `text` has the phrase's words in a row, ignoring case and punctuation.
pub fn contains_phrase(text: &str, phrase: &str) -> bool {
    let phrase = words(phrase);
    !phrase.is_empty() && words(text).windows(phrase.len()).any(|window| window == phrase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_ignores_case_punctuation_and_digit_style() {
        let accuracy = compare("Chapter 12. Down the Rabbit-Hole!", "chapter twelve, down the rabbit hole");
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
            let accuracy = compare(text, heard);
            assert_eq!((accuracy.recall, accuracy.extra), (1.0, 0.0), "{text} / {heard}");
        }
        // Joining is only forgiven when the joined word is really there.
        assert!(compare("the quarter deck", "the quarter").recall < 1.0);
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
        assert!(!compare(text, "in what trade setups and cost free workers").acceptable());
        let repeated = format!("{text} {text}");
        let accuracy = compare(text, &repeated);
        assert_eq!(accuracy.recall, 1.0);
        assert!(!accuracy.acceptable());
    }
}
