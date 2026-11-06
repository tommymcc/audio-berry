//! A scratch directory for tests that read and write files.

use std::fs;
use std::path::{Path, PathBuf};

/// A new, empty directory that's removed when the test ends, even if it fails.
/// Named `audio-berry-<process id>-<name>` in the system's temporary directory,
/// so `name` must differ between tests.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("audio-berry-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    /// A path inside the directory.
    pub fn join(&self, path: impl AsRef<Path>) -> PathBuf {
        self.0.join(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
