//! Writes numbers the same way on both sides of a comparison, since a book and
//! a transcript may write the same number in digits or in words: "21st" or
//! "twenty-first", "250" or "two hundred and fifty", "1066" or "ten sixty-six".
//!
//! Numbers under 100 are always spelled out. A number of 100 or more spelled out
//! on one side is put in digits only where the other side has it in digits, so
//! words that only look like a number, such as "ten twenty men", stay as they are.

use std::collections::HashSet;

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 8] = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const ORDINAL_ONES: [&str; 20] = [
    "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth",
];
const ORDINAL_TENS: [&str; 8] =
    ["twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth"];

/// The most words one spelled-out number is looked for in, as in "two thousand
/// five hundred and forty one".
const LONGEST_NUMBER: usize = 8;

/// Which of `chars` to drop so that numbers with commas between thousands
/// read as one number: each such comma, and any spaces before it, so "3,000" is
/// one number, as is "3 ,000", as Parakeet sometimes writes it.
pub fn thousands_separators(chars: &[char]) -> Vec<bool> {
    let mut dropped = vec![false; chars.len()];
    for (index, &c) in chars.iter().enumerate() {
        if c != ',' {
            continue;
        }
        // The spaces kept just before the comma, and the character before them.
        let mut before = index;
        while before > 0 && (dropped[before - 1] || chars[before - 1].is_whitespace()) {
            before -= 1;
        }
        let after_digit = before > 0 && chars[before - 1].is_ascii_digit();
        let three_digits_after = chars.get(index + 1..index + 4).is_some_and(|next| next.iter().all(char::is_ascii_digit))
            && !chars.get(index + 4).is_some_and(char::is_ascii_digit);
        if after_digit && three_digits_after {
            dropped[before..=index].fill(true);
        }
    }
    dropped
}

/// A lowercase word that is a number under 100, in digits and maybe an ordinal,
/// such as "21" or "21st", spelled out: "twenty one" or "twenty first".
pub fn spell_under_100(word: &str) -> Option<Vec<&'static str>> {
    let (digits, suffix) = word.split_at(word.find(|c: char| !c.is_ascii_digit()).unwrap_or(word.len()));
    let ordinal = matches!(suffix, "st" | "nd" | "rd" | "th");
    let number: usize = digits.parse().ok().filter(|&number| number < 100)?;
    if !suffix.is_empty() && !ordinal {
        return None;
    }
    let (ones, tens) = if ordinal { (&ORDINAL_ONES, &ORDINAL_TENS) } else { (&ONES, &TENS) };
    Some(match number {
        0..20 => vec![ones[number]],
        _ if number % 10 == 0 => vec![tens[number / 10 - 2]],
        _ => vec![TENS[number / 10 - 2], ones[number % 10]],
    })
}

/// `words` with each number spelled out there, of 100 or more, that `other`
/// has in digits, put in digits, each with how many of `words` it stands for.
/// Both are lowercase words, with numbers under 100 already spelled out.
pub fn match_digits(words: &[String], other: &[String]) -> Vec<(String, usize)> {
    let wanted: HashSet<u64> = other.iter().filter_map(|word| word.parse().ok()).filter(|&number| number >= 100).collect();
    let mut out = Vec::with_capacity(words.len());
    let mut index = 0;
    while index < words.len() {
        let rest = &words[index..];
        let longest = rest.len().min(LONGEST_NUMBER);
        let found = (1..=longest).rev().find_map(|length| {
            value(&rest[..length]).filter(|number| wanted.contains(number)).map(|number| (number, length))
        });
        match found {
            Some((number, length)) => {
                out.push((number.to_string(), length));
                index += length;
            }
            None => {
                out.push((words[index].clone(), 1));
                index += 1;
            }
        }
    }
    out
}

/// The value of `words` if together they spell one number: a cardinal such as
/// "a hundred", "two thousand five hundred and one" or "eighteen hundred", or a
/// year said in pairs of digits, such as "ten sixty six" or "nineteen oh five".
fn value(words: &[String]) -> Option<u64> {
    cardinal(words).or_else(|| year(words))
}

/// The value of a number under 100 spelled out at the start of `words`, such as
/// "forty five", and how many words it takes.
fn under_100(words: &[String]) -> Option<(u64, usize)> {
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

/// `words` as a cardinal number, if they're exactly one.
fn cardinal(words: &[String]) -> Option<u64> {
    let (mut total, mut current) = (0, 0);
    // Whether a number under 100 has been read since the last hundred or thousand.
    let mut small = false;
    let mut index = 0;
    while index < words.len() {
        let scale_next = words.get(index + 1).is_some_and(|next| next == "hundred" || next == "thousand");
        match words[index].as_str() {
            "a" if index == 0 && scale_next => {
                current = 1;
                small = true;
            }
            "hundred" if current < 100 => {
                current = current.max(1) * 100;
                small = false;
            }
            "thousand" if total == 0 => {
                total = current.max(1) * 1000;
                current = 0;
                small = false;
            }
            // Only between a hundred or thousand and what follows it.
            "and" if !small && total + current >= 100 && under_100(&words[index + 1..]).is_some() => {}
            _ => {
                let (number, length) = under_100(&words[index..]).filter(|_| !small)?;
                current += number;
                small = true;
                index += length;
                continue;
            }
        }
        index += 1;
    }
    (index > 0).then_some(total + current)
}

/// `words` as a year said in two pairs of digits, such as "eighteen twenty",
/// or "nineteen oh five".
fn year(words: &[String]) -> Option<u64> {
    let (century, length) = under_100(words).filter(|(century, _)| *century >= 10)?;
    let rest = &words[length..];
    let (pair, used) = match rest.first().map(String::as_str) {
        Some("oh") => under_100(&rest[1..]).filter(|&(ones, used)| ones < 10 && used == 1).map(|(ones, _)| (ones, 2))?,
        _ => under_100(rest).filter(|(pair, _)| *pair >= 10)?,
    };
    (length + used == words.len()).then_some(century * 100 + pair)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(String::from).collect()
    }

    /// `heard` with its numbers matched to `written`'s digits.
    fn matched(heard: &str, written: &str) -> String {
        let matched: Vec<String> = match_digits(&words(heard), &words(written)).into_iter().map(|(word, _)| word).collect();
        matched.join(" ")
    }

    /// `text` without the characters `thousands_separators` drops.
    fn join_thousands(text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let dropped = thousands_separators(&chars);
        chars.iter().zip(dropped).filter(|(_, dropped)| !dropped).map(|(c, _)| c).collect()
    }

    #[test]
    fn each_number_put_in_digits_says_how_many_words_it_stands_for() {
        let groups = match_digits(&words("paid two hundred and fifty pounds"), &words("paid 250 pounds"));
        let wanted = [("paid", 1), ("250", 4), ("pounds", 1)].map(|(word, count)| (word.to_string(), count));
        assert_eq!(groups, wanted);
    }

    #[test]
    fn numbers_under_100_are_spelled_out_and_ordinals_stay_ordinals() {
        assert_eq!(spell_under_100("7"), Some(vec!["seven"]));
        assert_eq!(spell_under_100("40"), Some(vec!["forty"]));
        assert_eq!(spell_under_100("45"), Some(vec!["forty", "five"]));
        assert_eq!(spell_under_100("2nd"), Some(vec!["second"]));
        assert_eq!(spell_under_100("21st"), Some(vec!["twenty", "first"]));
        assert_eq!(spell_under_100("30th"), Some(vec!["thirtieth"]));
        assert_eq!(spell_under_100("100"), None);
        assert_eq!(spell_under_100("4x4"), None);
        assert_eq!(spell_under_100("word"), None);
    }

    #[test]
    fn commas_between_thousands_are_dropped() {
        assert_eq!(join_thousands("3,000 barrels, 12,500 men"), "3000 barrels, 12500 men");
        assert_eq!(join_thousands("there were 3 ,000"), "there were 3000");
        assert_eq!(join_thousands("1, 2, 3"), "1, 2, 3");
        assert_eq!(join_thousands("in 1820,1821 and 1822"), "in 1820,1821 and 1822");
    }

    #[test]
    fn spelled_numbers_are_put_in_digits_where_the_other_side_has_them() {
        for (heard, written, wanted) in [
            ("paid two hundred and fifty", "paid 250", "paid 250"),
            ("two thousand five hundred men", "2500 men", "2500 men"),
            ("one hundred and one guns", "101 guns", "101 guns"),
            ("eighteen hundred", "1800", "1800"),
            ("three thousand barrels", "3000 barrels", "3000 barrels"),
        ] {
            assert_eq!(matched(heard, written), wanted, "{heard} / {written}");
        }
    }

    #[test]
    fn a_hundred_and_a_thousand_are_one_hundred_and_one_thousand() {
        assert_eq!(matched("a hundred and fifty men", "150 men"), "150 men");
        assert_eq!(matched("a hundred men", "100 men"), "100 men");
        assert_eq!(matched("a thousand men", "1000 men"), "1000 men");
        assert_eq!(matched("hundreds of men", "100 men"), "hundreds of men");
    }

    #[test]
    fn years_said_in_pairs_are_put_in_digits_where_the_other_side_has_them() {
        assert_eq!(matched("in ten sixty six", "in 1066"), "in 1066");
        assert_eq!(matched("in eighteen twenty", "in 1820"), "in 1820");
        assert_eq!(matched("in nineteen oh five", "in 1905"), "in 1905");
    }

    #[test]
    fn small_numbers_in_a_row_stay_as_they_are_without_digits_to_match() {
        for heard in ["ten twenty men", "eleven fifteen", "one two three", "twenty one guns", "two hundred and fifty"] {
            assert_eq!(matched(heard, "no digits here"), heard);
        }
        // Each number is matched only as the other side has it.
        assert_eq!(matched("eleven fifteen", "1115"), "1115");
        assert_eq!(matched("ten twenty men", "1020 men"), "1020 men");
    }

    #[test]
    fn part_of_a_spelled_number_matches_when_only_that_part_is_in_digits() {
        // The book's "seventeen 100-pound bags", heard as "seventeen hundred pound bags".
        assert_eq!(matched("seventeen hundred pound bags", "seventeen 100 pound bags"), "seventeen 100 pound bags");
        assert_eq!(matched("seventeen hundred pound bags", "1700 pound bags"), "1700 pound bags");
    }
}
