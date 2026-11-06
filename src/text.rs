//! Turns book text into something a speech model reads well, in small chunks.

/// Longest chunk sent to the model in one request, in characters.
pub const MAX_CHUNK_CHARS: usize = 500;

const ABBREVIATIONS: [(&str, &str); 4] = [
    ("Mr.", "Mister"),
    ("Mrs.", "Missus"),
    ("Ms.", "Miz"),
    ("Dr.", "Doctor"),
];
const NON_FINAL_ABBREVIATIONS: [&str; 6] = ["St.", "vs.", "etc.", "e.g.", "i.e.", "No."];

pub struct Paragraph {
    pub text: String,
    pub heading: bool,
}

fn is_roman(word: &str) -> bool {
    !word.is_empty() && word.chars().all(|c| "IVXLCDM".contains(c))
}

fn roman_value(word: &str) -> Option<u32> {
    if !is_roman(word) {
        return None;
    }
    let value = |c| match c {
        'I' => 1,
        'V' => 5,
        'X' => 10,
        'L' => 50,
        'C' => 100,
        'D' => 500,
        _ => 1000,
    };
    let digits: Vec<u32> = word.chars().map(value).collect();
    let total = digits.iter().enumerate().fold(0i64, |total, (i, &digit)| {
        match digits.get(i + 1) {
            Some(&next) if next > digit => total - i64::from(digit),
            _ => total + i64::from(digit),
        }
    });
    u32::try_from(total).ok()
}

/// "Chapter XII" reads as "Chapter 12", which models say more reliably.
fn expand_heading_numeral(text: &str) -> String {
    let words: Vec<&str> = text.split(' ').collect();
    let mut out = Vec::with_capacity(words.len());
    for (i, word) in words.iter().enumerate() {
        let stem = word.trim_end_matches(|c: char| c.is_ascii_punctuation());
        let follows_label = i > 0
            && ["chapter", "part", "book", "section"]
                .contains(&words[i - 1].to_lowercase().as_str());
        match roman_value(stem).filter(|_| follows_label) {
            Some(value) => out.push(format!("{value}{}", &word[stem.len()..])),
            None => out.push((*word).to_string()),
        }
    }
    out.join(" ")
}

/// `--` is an old way of typing a dash, unless it stretches a word out, as in
/// "Soo--oop", where a comma would break the word in two.
fn replace_double_hyphens(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '-' || chars.get(i + 1) != Some(&'-') {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let end = i + chars[i..].iter().take_while(|&&c| c == '-').count();
        let before = i.checked_sub(1).map(|index| chars[index].to_ascii_lowercase());
        let after = chars.get(end).map(char::to_ascii_lowercase);
        let doubled = after.is_some_and(|c| "aeiou".contains(c)) && chars.get(end + 1) == chars.get(end);
        let stretched = before.is_some_and(char::is_alphabetic)
            && after.is_some_and(char::is_alphabetic)
            && (before == after || doubled);
        out.push_str(if stretched { "-" } else { ", " });
        i = end;
    }
    out
}

/// Rejoins words a typesetter broke across lines, like "looking- glass".
fn rejoin_broken_words(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    chars
        .iter()
        .enumerate()
        .filter(|&(i, &c)| {
            let broken = c == ' '
                && i >= 2
                && chars[i - 1] == '-'
                && chars[i - 2].is_lowercase()
                && chars.get(i + 1).is_some_and(|next| next.is_lowercase());
            !broken
        })
        .map(|(_, &c)| c)
        .collect()
}

fn caps_letters(word: &str) -> Option<usize> {
    let letters = word.chars().filter(|c| c.is_alphabetic()).count();
    let upper = !word.chars().any(char::is_lowercase);
    (letters >= 2 && upper && !is_roman(word.trim_matches(|c: char| !c.is_alphabetic())))
        .then_some(letters)
}

fn title_case(word: &str) -> String {
    let mut seen_letter = false;
    word.chars()
        .flat_map(|c| {
            let keep_upper = c.is_alphabetic() && !seen_letter;
            seen_letter |= c.is_alphabetic();
            if keep_upper {
                c.to_uppercase().collect::<Vec<_>>()
            } else {
                c.to_lowercase().collect::<Vec<_>>()
            }
        })
        .collect()
}

/// Words stressed by their first letters, like "UNimportant".
fn has_shouted_prefix(word: &str) -> bool {
    let letters: Vec<char> = word.chars().filter(|c| c.is_alphabetic()).collect();
    let upper = letters.iter().take_while(|c| c.is_uppercase()).count();
    let lower = letters[upper..].iter().filter(|c| c.is_lowercase()).count();
    upper >= 2 && lower >= 3 && upper + lower == letters.len()
}

/// Shouted words get spelled out by models, so "DRINK ME" becomes "Drink Me".
/// Lone short capitals stay as they are, since they're often acronyms.
fn soften_capitals(text: &str) -> String {
    let words: Vec<&str> = text.split(' ').collect();
    let caps: Vec<Option<usize>> = words.iter().map(|word| caps_letters(word)).collect();
    words
        .iter()
        .enumerate()
        .map(|(i, word)| {
            let neighbour = (i > 0 && caps[i - 1].is_some())
                || caps.get(i + 1).is_some_and(Option::is_some);
            match caps[i] {
                Some(letters) if letters >= 4 || neighbour => title_case(word),
                None if has_shouted_prefix(word) => title_case(word),
                _ => (*word).to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Cleans one paragraph for reading aloud.
fn normalize(paragraph: &Paragraph) -> String {
    let mut text = String::with_capacity(paragraph.text.len());
    for c in replace_double_hyphens(&paragraph.text).chars() {
        match c {
            '\u{2018}' | '\u{2019}' | '\u{02BC}' => text.push('\''),
            '\u{201C}' | '\u{201D}' => text.push('"'),
            '\u{2014}' | '\u{2013}' | '\u{2015}' => text.push_str(", "),
            '\u{2026}' => text.push_str("..."),
            '\u{00A0}' => text.push(' '),
            '_' | '*' => {}
            c => text.push(c),
        }
    }
    for (abbreviation, word) in ABBREVIATIONS {
        text = text.replace(abbreviation, word);
    }

    let mut text = rejoin_broken_words(&text.split_whitespace().collect::<Vec<_>>().join(" "));
    while text.contains(", ,") {
        text = text.replace(", ,", ",");
    }
    for (from, to) in [
        (" ,", ","),
        (",,", ","),
        (",.", "."),
        (",!", "!"),
        (",?", "?"),
        (".,", "."),
        ("!,", "!"),
        ("?,", "?"),
        (":,", ":"),
        (";,", ";"),
        // A dash before a closing quote leaves `Well, " said`.
        (", \" ", ",\" "),
        (", ' ", ",' "),
        // A dash after an opening quote leaves `' , change`.
        (" ', ", " '"),
        (" \", ", " \""),
    ] {
        text = text.replace(from, to);
    }
    if let Some(stem) = text.strip_suffix(", \"") {
        text = format!("{stem},\"");
    }
    if let Some(stem) = text.strip_suffix(", '") {
        text = format!("{stem},'");
    }
    // A paragraph can open on a quote and then a dash, leaving `', change`.
    for quote in ["'", "\""] {
        if let Some(rest) = text.strip_prefix(quote).and_then(|rest| rest.strip_prefix(", ")) {
            text = format!("{quote}{rest}");
        }
    }
    let mut text = text
        .trim_start_matches(|c: char| c == ',' || c == ' ')
        .trim_end_matches(|c: char| c == ',' || c == ' ')
        .to_string();

    if paragraph.heading {
        text = expand_heading_numeral(&text);
    }
    let mut text = soften_capitals(&text);
    if paragraph.heading && !text.ends_with(|c: char| ".!?:".contains(c)) {
        text.push('.');
    }
    text
}

/// Splits text at sentence ends, keeping closing quotes with their sentence.
fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if !".!?".contains(chars[i]) {
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < chars.len() && ".!?\"')]".contains(chars[end]) {
            end += 1;
        }
        let at_space = end == chars.len() || chars[end].is_whitespace();
        let next = chars[end..].iter().find(|c| !c.is_whitespace());
        // `"Oh!" said Alice` carries on after the quote.
        let continues = next.is_some_and(|c| c.is_lowercase());
        let word: String = chars[start..end]
            .iter()
            .collect::<String>()
            .rsplit(' ')
            .next()
            .unwrap_or_default()
            .to_string();
        let initial = word.len() == 2 && word.starts_with(|c: char| c.is_uppercase());
        let abbreviation = NON_FINAL_ABBREVIATIONS.contains(&word.as_str());
        if at_space && !continues && !initial && !abbreviation {
            let sentence: String = chars[start..end].iter().collect();
            sentences.push(sentence.trim().to_string());
            start = end;
        }
        i = end;
    }
    let rest: String = chars[start..].iter().collect();
    if !rest.trim().is_empty() {
        sentences.push(rest.trim().to_string());
    }
    sentences
}

/// Breaks an overlong sentence at the last clause or word boundary that fits.
fn split_long(sentence: &str, max: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut rest = sentence.trim();
    while rest.chars().count() > max {
        let limit = rest
            .char_indices()
            .nth(max)
            .map_or(rest.len(), |(index, _)| index);
        let head = &rest[..limit];
        let in_second_half = |index: &usize| *index > limit / 2;
        let cut = head
            .rfind(|c| ";:".contains(c))
            .filter(in_second_half)
            .or_else(|| head.rfind(',').filter(in_second_half))
            .map(|index| index + 1)
            .or_else(|| head.rfind(' '))
            .unwrap_or(limit);
        parts.push(rest[..cut].trim().to_string());
        rest = rest[cut..].trim();
    }
    if !rest.is_empty() {
        parts.push(rest.to_string());
    }
    parts
}

/// Groups paragraphs into chunks that end on sentence boundaries. Headings get
/// their own chunk, and a paragraph end closes a chunk that's mostly full.
pub fn chunk(paragraphs: &[Paragraph], max: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut close = |current: &mut String| {
        if !current.is_empty() {
            chunks.push(std::mem::take(current));
        }
    };
    for paragraph in paragraphs {
        let text = normalize(paragraph);
        if text.is_empty() {
            continue;
        }
        if paragraph.heading {
            close(&mut current);
            current = text;
            close(&mut current);
            continue;
        }
        for sentence in split_sentences(&text) {
            for part in split_long(&sentence, max) {
                if !current.is_empty() && current.chars().count() + 1 + part.chars().count() > max {
                    close(&mut current);
                }
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(&part);
            }
        }
        if current.chars().count() * 10 >= max * 6 {
            close(&mut current);
        }
    }
    close(&mut current);
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(text: &str) -> Paragraph {
        Paragraph {
            text: text.to_string(),
            heading: false,
        }
    }

    fn heading(text: &str) -> Paragraph {
        Paragraph {
            text: text.to_string(),
            heading: true,
        }
    }

    #[test]
    fn long_sentences_fall_back_to_a_comma_after_an_early_semicolon() {
        let sentence = format!("Early clause; {} late clause, and some more words here", "word ".repeat(10));
        let parts = split_long(&sentence, 90);
        assert!(parts[0].ends_with("late clause,"), "{parts:?}");
    }

    #[test]
    fn headings_expand_numerals_and_get_a_stop() {
        assert_eq!(normalize(&heading("Chapter XII")), "Chapter 12.");
        assert_eq!(normalize(&heading("Chapter IV")), "Chapter 4.");
        assert_eq!(normalize(&heading("Down the Rabbit-Hole")), "Down the Rabbit-Hole.");
    }

    #[test]
    fn body_numerals_and_the_word_i_are_left_alone() {
        assert_eq!(normalize(&body("Part I said so")), "Part I said so");
        assert_eq!(normalize(&body("I think so.")), "I think so.");
    }

    #[test]
    fn punctuation_becomes_speakable() {
        assert_eq!(
            normalize(&body("\u{201C}Well\u{2014}\u{201D} said _Alice_\u{2019}s  cat.")),
            "\"Well,\" said Alice's cat."
        );
        assert_eq!(normalize(&body("Dr. Who met Mrs. Hudson.")), "Doctor Who met Missus Hudson.");
    }

    #[test]
    fn typed_dashes_become_speakable() {
        assert_eq!(
            normalize(&body("'important--unimportant-- along the sea-shore--' 'Two!' it said:-- 'Well.")),
            "'important, unimportant, along the sea-shore,' 'Two!' it said: 'Well."
        );
        assert_eq!(normalize(&body("no.--Yes")), "no. Yes");
        assert_eq!(normalize(&body("partners--' '--change")), "partners,' 'change");
        assert_eq!(normalize(&body("'--change lobsters")), "'change lobsters");
        assert_eq!(
            normalize(&body("Beau--ootiful Soo--oop of the e--e--evening")),
            "Beau-ootiful Soo-oop of the e-e-evening"
        );
    }

    #[test]
    fn broken_words_rejoin() {
        assert_eq!(normalize(&body("the looking- glass and well-known")), "the looking-glass and well-known");
        assert_eq!(normalize(&body("will prose- cute you.\u{2014}Come")), "will prose-cute you. Come");
    }

    #[test]
    fn stressed_prefixes_soften() {
        assert_eq!(normalize(&body("'UNimportant,' he said")), "'Unimportant,' he said");
        assert_eq!(normalize(&body("the PCs and PhDs and McCoy")), "the PCs and PhDs and McCoy");
    }

    #[test]
    fn shouted_words_soften_but_acronyms_survive() {
        assert_eq!(normalize(&body("DRINK ME, it said.")), "Drink Me, it said.");
        assert_eq!(normalize(&body("ORANGE marmalade")), "Orange marmalade");
        assert_eq!(normalize(&body("the BBC said")), "the BBC said");
    }

    #[test]
    fn sentences_keep_quotes_and_continuations_together() {
        assert_eq!(
            split_sentences("\"Oh!\" said Alice. \"Why?\" She left. W. Rabbit came."),
            vec!["\"Oh!\" said Alice.", "\"Why?\"", "She left.", "W. Rabbit came."]
        );
    }

    #[test]
    fn chunks_respect_the_limit_and_keep_every_word() {
        let long = "word, ".repeat(300);
        let paragraphs = vec![heading("Chapter I"), body("One two."), body(&long), body("Last one.")];
        let chunks = chunk(&paragraphs, 100);
        assert_eq!(chunks[0], "Chapter 1.");
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 100));
        let joined = chunks.join(" ");
        assert_eq!(joined.matches("word").count(), 300);
        assert!(joined.starts_with("Chapter 1. One two."));
        assert!(joined.ends_with("Last one."));
    }
}
