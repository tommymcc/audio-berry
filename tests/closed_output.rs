//! What the program does when whatever reads its output stops early, as `head` does.

#![cfg(unix)]

use std::os::unix::process::ExitStatusExt;
use std::process::{Command, Stdio};

const SIGPIPE: i32 = 13;

/// Like other Unix tools, it should stop quietly, ended by SIGPIPE, rather than panic.
#[test]
fn a_closed_output_ends_the_program_quietly() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_audio-berry"))
        .args(["chapters", "tests/fixtures/tiny.epub"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());

    let output = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.is_empty(), "expected no error output, got: {stderr}");
    assert_eq!(output.status.signal(), Some(SIGPIPE));
}
