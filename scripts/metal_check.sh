#!/usr/bin/env bash
# Builds audio-berry, narrates a short test, and writes a report to send back.
#
#   scripts/metal_check.sh <voice-sample>
#
# Everything goes in a new folder under output/: the report, the audio to listen
# to, and the book it downloads. A voice sample with no .txt beside it also
# downloads the 3 GB transcription model.
set -u

if [ $# -ne 1 ] || [ ! -f "$1" ]; then
    echo "usage: $0 <voice-sample>" >&2
    exit 2
fi
voice=$1
cd "$(dirname "$0")/.." || exit 1

BOOK_URL=https://www.gutenberg.org/ebooks/11.epub3.images
CHAPTER=1
binary=target/release/audio-berry
out=output/metal-check-$(date +%Y%m%d-%H%M%S)
report=$out/report.txt
mkdir -p "$out"

if [ "$(uname -s)" = Darwin ]; then
    build_args=(--no-default-features --features metal)
    time_args=(-l)
else
    build_args=()
    time_args=(-v)
fi

heading() { printf '\n=== %s\n' "$1"; }

# Runs a command with its time and memory use, and keeps going if it fails.
run() {
    local name=$1 status
    shift
    heading "$name"
    RUST_BACKTRACE=1 /usr/bin/time "${time_args[@]}" "$@"
    status=$?
    echo "--- $name exit code: $status"
    statuses+=("$name=$status")
    return "$status"
}

levels() {
    ffmpeg -hide_banner -i "$1" -af volumedetect -f null - 2>&1 |
        grep -E 'Duration|mean_volume|max_volume'
}

check() {
    statuses=()

    heading "System"
    git rev-parse --short HEAD
    git status --short
    uname -srm
    if [ "$(uname -s)" = Darwin ]; then
        sw_vers -productVersion
        sysctl -n machdep.cpu.brand_string
        echo "memory bytes: $(sysctl -n hw.memsize)"
    fi
    rustc --version
    ffmpeg -version | head -1
    echo "voice sample transcript present: $([ -f "${voice%.*}.txt" ] && echo yes || echo no)"

    if ! run build cargo build --release "${build_args[@]}"; then
        heading "Stopped: the build failed"
        return
    fi

    run speak "$binary" speak "$voice" "Twinkle, twinkle, little star." "$out/speak.wav"
    if [ -f "$out/speak.wav" ]; then
        heading "speak.wav levels"
        levels "$out/speak.wav"
    fi

    heading "Downloading the test book"
    curl -fsSL -o "$out/alice.epub" "$BOOK_URL"
    echo "curl exit code: $?"

    run chapters "$binary" chapters "$out/alice.epub"
    run book "$binary" book "$voice" "$out/alice.epub" "$out/book" "$CHAPTER"

    heading "Book results"
    cat "$out/book/narration.txt" 2>/dev/null | grep -E '^(precision|speed)='
    cat "$out/book/report.tsv" 2>/dev/null
    for wav in "$out"/book/chapters/*.wav; do
        [ -f "$wav" ] && echo "$wav" && levels "$wav"
    done

    heading "Summary"
    echo "steps: ${statuses[*]}"
    echo "listen to: $out/speak.wav and $out/book/chapters/"
}

# The report leaves out your home folder's path, but it does include whatever
# the run prints, such as what was heard in the voice sample.
check 2>&1 | perl -pe 'BEGIN { $| = 1 } s/\Q$ENV{HOME}\E/~/g' | tee "$report"

printf '\nSend back: %s\n' "$report"
