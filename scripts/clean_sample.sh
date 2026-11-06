#!/usr/bin/env bash
# Makes a voice sample from a noisy recording, such as one with birds or wind.
#
#   scripts/clean_sample.sh <recording> <start-seconds> <end-seconds> <output.wav> [reduction-dB]
#
# Cleans the whole recording with DeepFilterNet 3, then cuts start to end into a
# 24 kHz mono sample. Pick start and end in pauses between words, 10 to 15
# seconds apart.
#
# reduction-dB caps how much noise is removed, and defaults to 15. Cleaning harder
# makes takes quieter but makes the clone sound less like the speaker. Tested on
# one outdoor recording, voice ID against the speaker's own recording was:
#
#   reduction   takes' floor   voice ID
#   15 dB       about -53 dB   0.85   (the speaker's own recording: 0.91)
#   20 dB       about -56 dB   0.80
#   30 dB       about -59 dB   0.75
#   100 (all)   about -79 dB   0.74
#
# At 15 dB, takes keep faint room tone that fails the default -55 dB noise check,
# so narrate with AUDIO_BERRY_NOISE_LIMIT_DB=-50 if the takes sound clean (see
# "Noise limit" in the README).
#
# DeepFilterNet runs on the CPU in its own Python environment, made with uv on
# first use (about 800 MB, in ~/.cache/audio-berry). Cleaning takes seconds.
set -euo pipefail

if [ $# -lt 4 ] || [ $# -gt 5 ] || [ ! -f "$1" ]; then
    echo "usage: $0 <recording> <start-seconds> <end-seconds> <output.wav> [reduction-dB]" >&2
    exit 2
fi
recording=$1 start=$2 end=$3 output=$4 reduction=${5:-15}

# Pinned because DeepFilterNet 0.5.6 breaks with newer torchaudio. The CPU build
# of torch is a fraction of the size of the CUDA one.
venv=${XDG_CACHE_HOME:-$HOME/.cache}/audio-berry/deepfilternet-0.5.6
if [ ! -x "$venv/bin/deepFilter" ]; then
    echo "Setting up DeepFilterNet in $venv"
    uv venv -q -p 3.11 "$venv"
    VIRTUAL_ENV=$venv uv pip install -q \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.1.2+cpu" "torchaudio==2.1.2+cpu" "deepfilternet==0.5.6" "numpy<2" \
        "soundfile==0.12.1"
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# Clean all of it rather than just the cut, so the model has context at the edges.
ffmpeg -hide_banner -loglevel error -i "$recording" -ac 1 "$work/recording.wav"
if ! "$venv/bin/deepFilter" "$work/recording.wav" -a "$reduction" -o "$work" >"$work/deepfilter.log" 2>&1; then
    cat "$work/deepfilter.log" >&2
    echo "DeepFilterNet failed; its output is above" >&2
    exit 1
fi
ffmpeg -hide_banner -loglevel error -y -i "$work/recording_DeepFilterNet3.wav" \
    -ac 1 -ar 24000 -c:a pcm_s16le "$work/cleaned.wav"

mkdir -p "$(dirname "$output")"
ffmpeg -hide_banner -loglevel error -y -ss "$start" -to "$end" -i "$work/cleaned.wav" \
    -c:a pcm_s16le "$output"

seconds=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$output")
peak=$(ffmpeg -hide_banner -i "$output" -af volumedetect -f null - 2>&1 | grep -oE 'max_volume: -?[0-9.]+ dB')
echo "Wrote $output: ${seconds}s, $peak, cleaned with up to $reduction dB of reduction"
echo "The first narration transcribes it to ${output%.*}.txt. Check that file's words."
