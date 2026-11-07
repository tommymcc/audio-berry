//! Reading files that may not have been written yet.

use color_eyre::eyre::Result;
use fs_err as fs;
use std::io::ErrorKind;
use std::path::Path;

/// Reads a file as text, or `None` if there's no such file. Any other failure
/// says which file it was.
pub fn read_if_present(path: &Path) -> Result<Option<String>> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_dir::TempDir;

    #[test]
    fn a_missing_file_is_none_and_a_present_one_is_its_text() {
        let dir = TempDir::new("read-if-present");
        assert_eq!(read_if_present(&dir.join("absent.txt")).unwrap(), None);
        std::fs::write(dir.join("present.txt"), "words").unwrap();
        assert_eq!(read_if_present(&dir.join("present.txt")).unwrap().as_deref(), Some("words"));
    }

    #[test]
    fn other_failures_name_the_file() {
        let dir = TempDir::new("read-if-present-dir");
        // Reading a directory as text fails, but not because it's missing.
        let error = read_if_present(&dir.join("")).unwrap_err().to_string();
        assert!(error.contains("read-if-present-dir"), "{error}");
    }
}
