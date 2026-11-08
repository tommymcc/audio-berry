//! Compares what was heard in a take with the text it should say, word by word,
//! forgiving the differences in spelling a transcriber makes, and the words a
//! voice's allowances say it hears as others.

use crate::numbers;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// Passages the user approved by ear scored 0.98-1.00 recall with 0.00-0.02 extra.
const MIN_RECALL: f32 = 0.92;
const MAX_EXTRA: f32 = 0.12;

/// The fewest words in a row of the text that count as a moved phrase when
/// heard somewhere else. See `Alignment::moved_runs`. On 5,053 stored Whisper
/// transcripts, one word flagged 75 chunks with 80 moves, 71 of them one of the
/// texts' 50 commonest words, such as "and", missed in one place and heard in
/// another; two words flagged one chunk, a real move.
const MIN_MOVED_WORDS: usize = 2;

pub struct Accuracy {
    /// Share of the text's words heard, in order.
    pub recall: f32,
    /// Words heard that aren't in the text, relative to the text's length.
    pub extra: f32,
    /// Each phrase of the text heard in another place, in the text's order. A
    /// phrase that traded places with the words beside it in the text is named
    /// with them in the order heard, as `first / before / second`; any other is
    /// named alone. Words are as compared: lowercase, numbers under 100 spelled
    /// out, and numbers of 100 or more in digits where the other side has them
    /// in digits.
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

    /// The moved phrases joined by ` | `, as `report.tsv` shows them, or nothing.
    pub fn moved_list(&self) -> String {
        self.moved.join(" | ")
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

    /// Whether `heard` counts as the text's `written` word: the same word, or
    /// one it may be heard as.
    pub fn matches(&self, written: &str, heard: &str) -> bool {
        written == heard || self.allows(written, heard)
    }

    /// How many pairs there are.
    pub fn len(&self) -> usize {
        self.heard_as.values().map(HashSet::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.heard_as.is_empty()
    }
}

/// A word as compared, and the bytes of the text it came from. A number spelled
/// out gives several words with the same bytes, and words joined into one have
/// the bytes from the first to the last.
#[derive(Clone, Debug, PartialEq)]
struct Spanned {
    word: String,
    bytes: Range<usize>,
}

/// Lowercases into words, with commas between thousands dropped and numbers
/// under 100 spelled out, as `numbers` writes them.
fn words(text: &str) -> Vec<String> {
    plain(spanned_words(text))
}

fn plain(words: Vec<Spanned>) -> Vec<String> {
    words.into_iter().map(|spanned| spanned.word).collect()
}

/// The words of `text` as `words` gives them, each with its bytes in `text`.
fn spanned_words(text: &str) -> Vec<Spanned> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let dropped = numbers::thousands_separators(&chars.iter().map(|&(_, c)| c).collect::<Vec<_>>());
    // Runs of letters, digits and apostrophes, lowercased.
    let mut runs: Vec<Spanned> = Vec::new();
    let mut current: Option<Spanned> = None;
    for (&(byte, c), dropped) in chars.iter().zip(dropped) {
        if dropped {
            continue;
        }
        if c.is_alphanumeric() || c == '\'' {
            let end = byte + c.len_utf8();
            match &mut current {
                Some(run) => {
                    run.word.extend(c.to_lowercase());
                    run.bytes.end = end;
                }
                None => current = Some(Spanned { word: c.to_lowercase().collect(), bytes: byte..end }),
            }
        } else {
            runs.extend(current.take());
        }
    }
    runs.extend(current);
    let mut out = Vec::new();
    for run in runs {
        let word = run.word.trim_matches('\'');
        let at = |word: &str| Spanned { word: word.to_string(), bytes: run.bytes.clone() };
        match numbers::spell_under_100(word) {
            Some(spelled) => out.extend(spelled.into_iter().map(at)),
            None if !word.is_empty() => out.push(at(word)),
            None => {}
        }
    }
    out
}

/// The words of two texts, with each number of 100 or more that one spells out
/// and the other has in digits put in digits.
fn words_of_both(first: &str, second: &str) -> (Vec<String>, Vec<String>) {
    let (first, second) = spanned_words_of_both(first, second);
    (plain(first), plain(second))
}

/// `words_of_both`, with each word's bytes in its text.
fn spanned_words_of_both(first: &str, second: &str) -> (Vec<Spanned>, Vec<Spanned>) {
    let (first, second) = (spanned_words(first), spanned_words(second));
    (in_digits(&first, &second), in_digits(&second, &first))
}

/// `words` with each number spelled out there, of 100 or more, that `other` has
/// in digits, put in digits.
fn in_digits(words: &[Spanned], other: &[Spanned]) -> Vec<Spanned> {
    let plain = |words: &[Spanned]| words.iter().map(|spanned| spanned.word.clone()).collect::<Vec<_>>();
    let mut index = 0;
    numbers::match_digits(&plain(words), &plain(other))
        .into_iter()
        .map(|(word, count)| {
            let bytes = words[index].bytes.start..words[index + count - 1].bytes.end;
            index += count;
            Spanned { word, bytes }
        })
        .collect()
}

/// Joins neighbouring words where `other` has them as one word, so "quarter
/// deck" matches "quarterdeck" and "e book" matches "ebook".
fn join_split_words(words: &[Spanned], other: &[Spanned]) -> Vec<Spanned> {
    let mut out = Vec::with_capacity(words.len());
    let mut index = 0;
    while index < words.len() {
        if let Some(next) = words.get(index + 1) {
            let joined = format!("{}{}", words[index].word, next.word);
            if other.iter().any(|spanned| spanned.word == joined) {
                out.push(Spanned { word: joined, bytes: words[index].bytes.start..next.bytes.end });
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
fn aligned_words(text: &str, transcript: &str) -> (Vec<Spanned>, Vec<Spanned>) {
    let (expected, heard) = spanned_words_of_both(text, transcript);
    let expected = join_split_words(&expected, &heard);
    let heard = join_split_words(&heard, &expected);
    (expected, heard)
}

/// A text's words lined up with the words heard, by the longest common
/// subsequence of the two, where a word matches itself or a word the
/// allowances let it be heard as.
struct Alignment<'a> {
    expected: Vec<String>,
    heard: Vec<String>,
    /// The bytes of the text each of `expected` came from.
    expected_bytes: Vec<Range<usize>>,
    /// The bytes of the transcript each of `heard` came from.
    heard_bytes: Vec<Range<usize>>,
    allowances: &'a Allowances,
    /// How many words, in order, `expected[i..]` and `heard[j..]` have in
    /// common, as `table[i][j]`.
    table: Vec<Vec<usize>>,
}

/// One step through an alignment, in the order of both texts.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Step {
    /// A word of the text and the word heard for it.
    Paired(usize, usize),
    /// A word of the text heard nowhere in its place.
    Missed(usize),
    /// A word heard that isn't in the text in its place.
    Added(usize),
}

/// A run of words of the text heard in the same order, but not where the
/// common subsequence has them: `expected[text]` heard as `heard[transcript]`.
#[derive(Clone, Debug)]
struct Run {
    text: std::ops::Range<usize>,
    transcript: std::ops::Range<usize>,
}

impl<'a> Alignment<'a> {
    fn new(text: &str, transcript: &str, allowances: &'a Allowances) -> Alignment<'a> {
        let split = |words: Vec<Spanned>| -> (Vec<String>, Vec<Range<usize>>) {
            words.into_iter().map(|spanned| (spanned.word, spanned.bytes)).unzip()
        };
        let (expected, heard) = aligned_words(text, transcript);
        let ((expected, expected_bytes), (heard, heard_bytes)) = (split(expected), split(heard));
        let mut table = vec![vec![0usize; heard.len() + 1]; expected.len() + 1];
        for i in (0..expected.len()).rev() {
            for j in (0..heard.len()).rev() {
                table[i][j] = if allowances.matches(&expected[i], &heard[j]) {
                    table[i + 1][j + 1] + 1
                } else {
                    table[i + 1][j].max(table[i][j + 1])
                };
            }
        }
        Alignment { expected, heard, expected_bytes, heard_bytes, allowances, table }
    }

    /// How many words of the text were heard in order.
    fn matched(&self) -> usize {
        self.table[0][0]
    }

    fn matches(&self, i: usize, j: usize) -> bool {
        self.allowances.matches(&self.expected[i], &self.heard[j])
    }

    /// The common subsequence as steps. Words that match are paired as soon as
    /// they meet. Otherwise the step keeps the longer common subsequence ahead,
    /// and on a tie counts the text's word missed before the heard word added.
    fn steps(&self) -> Vec<Step> {
        let (expected, heard) = (self.expected.len(), self.heard.len());
        let mut steps = Vec::with_capacity(expected + heard);
        let (mut i, mut j) = (0, 0);
        while i < expected && j < heard {
            if self.matches(i, j) {
                steps.push(Step::Paired(i, j));
                i += 1;
                j += 1;
            } else if self.table[i + 1][j] >= self.table[i][j + 1] {
                steps.push(Step::Missed(i));
                i += 1;
            } else {
                steps.push(Step::Added(j));
                j += 1;
            }
        }
        steps.extend((i..expected).map(Step::Missed));
        steps.extend((j..heard).map(Step::Added));
        steps
    }

    /// Each phrase of the text heard in another place, described as
    /// `Accuracy::moved` says, in the text's order. See `moved_runs`.
    fn moved(&self) -> Vec<String> {
        let pairs: Vec<(usize, usize)> = self
            .steps()
            .into_iter()
            .filter_map(|step| match step {
                Step::Paired(i, j) => Some((i, j)),
                _ => None,
            })
            .collect();
        self.moved_runs(&pairs).iter().map(|run| self.describe(run, &pairs)).collect()
    }

    /// The runs of at least `MIN_MOVED_WORDS` words that `pairs` leaves out on
    /// both sides, but that match word for word between them: words heard in
    /// the order written, in another place relative to the words around them.
    ///
    /// The longest run is taken first; of runs as long, the one that ends first
    /// in the text, then in the transcript. Each word belongs to one run at
    /// most. A word of a moved phrase that the subsequence pairs elsewhere, as a
    /// "the" can be, splits the run there, so a phrase of two such words isn't
    /// found. A run heard right beside the same words heard again is a repeat,
    /// not a move, and isn't counted. Runs one word apart on both sides, as when
    /// a word in the middle of a moved phrase was misheard, are joined. They
    /// come in the text's order.
    fn moved_runs(&self, pairs: &[(usize, usize)]) -> Vec<Run> {
        let (mut expected_used, mut heard_used) = (vec![false; self.expected.len()], vec![false; self.heard.len()]);
        for &(i, j) in pairs {
            expected_used[i] = true;
            heard_used[j] = true;
        }
        let mut runs: Vec<Run> = Vec::new();
        loop {
            // run[i + 1][j + 1]: how many unused words in a row match, ending at
            // `expected[i]` and `heard[j]`.
            let mut run = vec![vec![0usize; self.heard.len() + 1]; self.expected.len() + 1];
            let (mut longest, mut end) = (0, (0, 0));
            for i in 0..self.expected.len() {
                for j in 0..self.heard.len() {
                    if !expected_used[i] && !heard_used[j] && self.matches(i, j) {
                        run[i + 1][j + 1] = run[i][j] + 1;
                        if run[i + 1][j + 1] > longest {
                            longest = run[i + 1][j + 1];
                            end = (i + 1, j + 1);
                        }
                    }
                }
            }
            if longest < MIN_MOVED_WORDS {
                break;
            }
            let found = Run { text: end.0 - longest..end.0, transcript: end.1 - longest..end.1 };
            expected_used[found.text.clone()].fill(true);
            heard_used[found.transcript.clone()].fill(true);
            if !self.repeated(&found.transcript) {
                runs.push(found);
            }
        }
        runs.sort_by_key(|run| run.text.start);
        let mut joined: Vec<Run> = Vec::new();
        for run in runs {
            match joined.last_mut() {
                Some(last)
                    if run.text.start <= last.text.end + 1
                        && run.transcript.start >= last.transcript.end
                        && run.transcript.start <= last.transcript.end + 1 =>
                {
                    last.text.end = run.text.end;
                    last.transcript.end = run.transcript.end;
                }
                _ => joined.push(run),
            }
        }
        joined
    }

    /// Whether the words heard in `range` are heard again right before or after.
    fn repeated(&self, range: &std::ops::Range<usize>) -> bool {
        let words = &self.heard[range.clone()];
        let before = range.start.checked_sub(words.len()).map(|start| &self.heard[start..range.start]);
        let after = self.heard.get(range.end..range.end + words.len());
        before == Some(words) || after == Some(words)
    }

    /// The moved run's words, or when it traded places with the words next to
    /// it in the text, both phrases in the order heard: `first / before / second`.
    ///
    /// The run swapped with the paired words it was heard on the other side of,
    /// when those come straight after it in the text, for a run heard late, or
    /// straight before it, for a run heard early, with nothing between but
    /// words missed.
    fn describe(&self, run: &Run, pairs: &[(usize, usize)]) -> String {
        let phrase = |range: std::ops::Range<usize>| self.expected[range].join(" ");
        let moved = phrase(run.text.clone());
        let paired = |range: std::ops::Range<usize>| range.into_iter().any(|i| pairs.iter().any(|&(p, _)| p == i));
        // How many pairs come before the run in the text, and in the transcript.
        let in_text = pairs.partition_point(|&(i, _)| i < run.text.start);
        let in_transcript = pairs.partition_point(|&(_, j)| j < run.transcript.start);
        if in_transcript > in_text {
            let (first, last) = (pairs[in_text].0, pairs[in_transcript - 1].0);
            if !paired(run.text.end..first) {
                return format!("{} / before / {moved}", phrase(first..last + 1));
            }
        } else if in_transcript < in_text {
            let (first, last) = (pairs[in_transcript].0, pairs[in_text - 1].0);
            if !paired(last + 1..run.text.start) {
                return format!("{moved} / before / {}", phrase(first..last + 1));
            }
        }
        moved
    }
}

/// How much of `text` was heard in `transcript`, and how much more besides. A
/// word heard as another that `allowances` allow counts as heard.
pub fn compare(text: &str, transcript: &str, allowances: &Allowances) -> Accuracy {
    let alignment = Alignment::new(text, transcript, allowances);
    let matched = alignment.matched();
    let length = alignment.expected.len().max(1) as f32;
    Accuracy {
        recall: matched as f32 / length,
        extra: alignment.heard.len().saturating_sub(matched) as f32 / length,
        moved: alignment.moved(),
        transcript: transcript.to_string(),
    }
}

/// A word of a text lined up with the word heard for it, as the bytes of the
/// text and of the transcript each came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Pair {
    pub written: Range<usize>,
    pub heard: Range<usize>,
}

/// The words of `text` heard in `transcript`, lined up as `compare` lines them
/// up, in order.
pub fn pairs(text: &str, transcript: &str, allowances: &Allowances) -> Vec<Pair> {
    let alignment = Alignment::new(text, transcript, allowances);
    alignment
        .steps()
        .into_iter()
        .filter_map(|step| match step {
            Step::Paired(i, j) => Some(Pair {
                written: alignment.expected_bytes[i].clone(),
                heard: alignment.heard_bytes[j].clone(),
            }),
            _ => None,
        })
        .collect()
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
    let none = Allowances::default();
    let alignment = Alignment::new(text, transcript, &none);
    let mut found = Vec::new();
    // The words between two pairs, on each side.
    let (mut missed, mut instead): (Vec<&String>, Vec<&String>) = (Vec::new(), Vec::new());
    let mut flush = |missed: &mut Vec<&String>, instead: &mut Vec<&String>| {
        if let ([written], [other]) = (missed.as_slice(), instead.as_slice()) {
            found.push(WordPair::new(written, other));
        }
        missed.clear();
        instead.clear();
    };
    for step in alignment.steps() {
        match step {
            Step::Paired(..) => flush(&mut missed, &mut instead),
            Step::Missed(i) => missed.push(&alignment.expected[i]),
            Step::Added(j) => instead.push(&alignment.heard[j]),
        }
    }
    flush(&mut missed, &mut instead);
    Substitutions { text_words: alignment.expected.clone(), found }
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
        phrase.iter().zip(window).all(|(written, heard)| allowances.matches(written, heard))
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

    /// The text and transcript words that `pairs` lines up, as written.
    fn paired<'a>(text: &'a str, transcript: &'a str, allowances: &Allowances) -> Vec<(&'a str, &'a str)> {
        let lined_up = super::pairs(text, transcript, allowances);
        lined_up.into_iter().map(|pair| (&text[pair.written], &transcript[pair.heard])).collect()
    }

    #[test]
    fn pairs_give_each_word_as_written_on_both_sides() {
        assert_eq!(
            paired("The 'Rabbit' sat, quietly.", "the rabbit sat quite", &none()),
            [("The", "the"), ("'Rabbit'", "rabbit"), ("sat", "sat")]
        );
        assert_eq!(
            paired("In 1066 he paid 3,000 pounds", "In ten sixty six he paid 3 ,000 pounds", &none()),
            [("In", "In"), ("1066", "ten sixty six"), ("he", "he"), ("paid", "paid"), ("3,000", "3 ,000"), ("pounds", "pounds")]
        );
        assert_eq!(
            paired("21 guns on the quarter-deck", "twenty-one guns on the quarterdeck", &none()),
            [
                ("21", "twenty"),
                ("21", "one"),
                ("guns", "guns"),
                ("on", "on"),
                ("the", "the"),
                ("quarter-deck", "quarterdeck")
            ]
        );
        assert_eq!(paired("sat in the boat", "sat and the boat", &allow("in = and"))[1], ("in", "and"));
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

    /// A chunk's text and a transcript of a take of it, from the Whisper
    /// transcripts stored for The Coral Island: two phrases said in each other's
    /// place, after a few words that aren't in the text.
    const SWAPPED_TEXT: &str = "Now, while engaged in the coasting trade, I fell in with many seamen who had \
        travelled to almost every quarter of the globe; and I freely confess that my heart glowed ardently within me \
        as they recounted their wild adventures in foreign lands, the dreadful storms they had weathered, the \
        appalling dangers they had escaped, the wonderful creatures they had seen both on the land and in the sea,";
    const SWAPPED_HEARD: &str = "Very tough. Now, while engaged in the coasting trade, I fell in with many seamen \
        who had travelled to almost every quarter of the globe. And I freely confessed that my heart glowed \
        ardently within me as they recounted their wild adventures in foreign lands. The dreadful storms they had \
        weathered, the appalling dangers they had escaped, both on the land and in the sea, the wonderful \
        creatures they had seen,";
    const SWAP: &str = "both on the land and in the sea / before / the wonderful creatures they had seen";

    #[test]
    fn phrases_heard_in_each_other_s_place_are_both_named() {
        let accuracy = compare(SWAPPED_TEXT, SWAPPED_HEARD, &none());
        assert_eq!(accuracy.moved, [SWAP]);
        assert_eq!(accuracy.moved_list(), SWAP);
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
        assert_eq!(accuracy.moved, ["for a long while / before / across the water"]);
        assert!(!accuracy.acceptable());
    }

    #[test]
    fn a_swap_is_named_the_same_whichever_half_the_alignment_keeps() {
        // Halves of the same length, so either could be the one left in place.
        let text = "at last the old man went down to the river to fish";
        let heard = "the old man at last went down to the river to fish";
        assert_eq!(compare(text, heard, &none()).moved, ["the old man / before / at last"]);
        let heard = "at last went down the old man to the river to fish";
        assert_eq!(compare(text, heard, &none()).moved, ["went down / before / the old man"]);
    }

    #[test]
    fn a_phrase_moved_far_is_named_with_what_it_was_heard_across() {
        let text = "one day the old man went down to the river and there he sat in the sun until evening came";
        let heard = "the old man one day went down to the river and there he sat until evening came in the sun";
        let accuracy = compare(text, heard, &none());
        assert_eq!(accuracy.moved, ["the old man / before / one day", "until evening came / before / in the sun"]);
        assert_eq!(
            accuracy.moved_list(),
            "the old man / before / one day | until evening came / before / in the sun"
        );
    }

    #[test]
    fn a_swap_across_words_missed_names_only_the_words_heard() {
        // "at dusk" isn't heard, so the halves are the words either side of it.
        let text = "into the woods at dusk they ran and the night came down";
        let heard = "they ran into the woods and the night came down";
        assert_eq!(compare(text, heard, &none()).moved, ["they ran / before / into the woods"]);
    }

    #[test]
    fn words_heard_in_order_move_nothing() {
        for (text, heard) in [
            ("the quick brown fox jumps over the lazy dog", "the quick brown fox jumps over the lazy dog"),
            // Words repeated and dropped, as some stored transcripts have them.
            ("the quick brown fox jumps over the lazy dog", "the quick brown the quick brown fox jumps over the dog"),
            ("the quick brown fox jumps over the lazy dog", "the quick brown fox jumps over the lazy dog the lazy dog"),
            ("the quick brown fox jumps over the lazy dog", "quick fox over lazy"),
            // A single word heard out of place is too little to call a phrase.
            ("he said that the boat was gone", "said he that the boat was gone"),
        ] {
            let accuracy = compare(text, heard, &none());
            assert!(accuracy.in_order(), "{heard}: {:?}", accuracy.moved);
            assert_eq!(accuracy.moved_list(), "");
        }
    }

    #[test]
    fn a_stutter_beside_words_missed_elsewhere_is_no_move() {
        // "of the" is missed after "top" and said twice after "end": a repeat
        // and a drop, which recall and extra already count.
        let text = "He climbed to the top of the mast and tied the end of the rope to it.";
        let heard = "He climbed to the top mast and tied the end of the of the rope to it.";
        let accuracy = compare(text, heard, &none());
        assert!(accuracy.in_order(), "{:?}", accuracy.moved);
        assert!(!accuracy.acceptable(), "the words missed still fail it");
    }

    #[test]
    fn a_moved_phrase_with_a_word_misheard_is_named_whole() {
        let heard = SWAPPED_HEARD.replace("wonderful creatures", "wonderful creature");
        assert_eq!(compare(SWAPPED_TEXT, &heard, &none()).moved, [SWAP]);
    }

    #[test]
    fn a_common_word_paired_elsewhere_splits_a_moved_phrase() {
        // The subsequence pairs the moved "the" with the text's second "the",
        // which wasn't said, leaving only "net" out of place: too little to
        // call a phrase. Taking such words into a run flagged two transcripts
        // stored for other books that had words missed, not moved.
        let text = "He laid the net beside the boat and went home.";
        let heard = "He laid beside the net boat and went home.";
        assert!(compare(text, heard, &none()).in_order());
    }

    #[test]
    fn a_moved_phrase_is_matched_through_allowances_and_numbers() {
        let text = "At the end of the day he paid 45 pounds for the old grey horse and rode it home.";
        let heard = "At the end of the day he paid for the old grey horse forty-five pounds and rode it home.";
        let accuracy = compare(text, heard, &none());
        assert_eq!(accuracy.moved, ["for the old grey horse / before / forty five pounds"]);
        let text = "He sat quite still all through the night in the boat with his hands upon the oars.";
        let heard = "He sat quite still and the boat all through the night with his hands upon the oars.";
        assert_eq!(compare(text, heard, &allow("in = and")).moved, ["in the boat / before / all through the night"]);
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
