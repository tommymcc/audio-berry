//! Writes commands for the user to copy into a shell, as advice prints them.

use std::path::Path;

/// A path as one shell word, quoted if it has anything a shell would split or
/// expand, and starting `./` if it would otherwise be read as an option.
pub fn word(path: &Path) -> String {
    let mut text = path.display().to_string();
    if text.starts_with('-') {
        text.insert_str(0, "./");
    }
    let plain = |c: char| c.is_ascii_alphanumeric() || "/._-+,:@%".contains(c);
    if !text.is_empty() && text.chars().all(plain) {
        text
    } else {
        format!("'{}'", text.replace('\'', r"'\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_paths_that_need_it_are_quoted() {
        assert_eq!(word(Path::new("voices/dad-raw-1.wav")), "voices/dad-raw-1.wav");
        assert_eq!(word(Path::new("Dad's tape.mp3")), r"'Dad'\''s tape.mp3'");
    }

    #[test]
    fn a_path_that_looks_like_an_option_is_made_relative() {
        assert_eq!(word(Path::new("-take 2.wav")), "'./-take 2.wav'");
        assert_eq!(word(Path::new("-take.wav")), "./-take.wav");
    }
}
