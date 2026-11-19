# AMD GPUs and CPUs without AVX2

How audio-berry was made to run on a machine with an AMD Radeon RX 6600 and an
AMD FX-8350, what had to be patched or worked around, and why. It records the
reasoning, including what was tried and dropped, so the choices can be revisited
when upstream changes.

## The machine

| | |
|---|---|
| CPU | AMD FX-8350 (Piledriver, 2012): AVX, FMA, F16C, but **no AVX2** |
| GPU | AMD Radeon RX 6600, 8 GB (Navi 23, RDNA2, **gfx1032**) |
| OS | Ubuntu 26.04 LTS |
| ROCm | Ubuntu's own ROCm 7.1 packages (`hipcc` 7.1.1+dfsg, whose `hipconfig` reports HIP 7.1.52801), plus AMD's ROCm 7.1.1 libraries in `/opt/rocm` |

## What runs where

| Build | Speech (OmniVoice) | Checker (Parakeet) |
|---|---|---|
| default (`cuda`) | NVIDIA only | ONNX Runtime, CUDA |
| `--no-default-features` | CPU, candle | ONNX Runtime, CPU: **crashes on CPUs without AVX2** ([1](#1-onnx-runtime-needs-avx2)) |
| `--no-default-features --features vulkan` | CPU, candle | burn on Vulkan (any GPU) |
| `--no-default-features --features rocm` | transformer backbone and audio decoder on the GPU with burn ROCm (F32, or F16 projection weights with `OMNIVOICE_BURN_F16=1`), the backbone's matmuls through hipBLAS where it's installed ([13](#13-burns-matmul-is-slow-on-rdna2-so-omnivoices-matmuls-use-hipblas-fork)); voice preparation, text embeddings (once per chunk) and sampling with a temperature on the CPU with candle | burn on ROCm |

The `cuda` and `metal` builds don't use burn or CubeCL at all (`cargo tree`
shows neither), so nothing here changes them; they haven't been built on this
machine, which has neither toolkit.

## Setup

What another user on Ubuntu 26.04 with an AMD GPU would do.

1. **Rust 1.92 or newer.** `ort` 2.0.0-rc.15 (under parakeet-rs) needs 1.92.
   This machine was upgraded from 1.91.1 to 1.100.0 with `rustup update stable`.
2. **ROCm from Ubuntu's archive**, for `rocm` builds: `sudo apt install rocm`
   (Ubuntu 26.04 ships 7.1). The build runs `hipconfig`, which must be on `PATH`;
   at run time burn loads `libamdhip64` and hipRTC itself. The user must be in
   the `render` and `video` groups to open `/dev/kfd`.
3. **Build:**
   ```sh
   cargo build --release --no-default-features --features rocm
   ```
4. **hipBLAS, optional.** With hipBLAS and rocBLAS installed, OmniVoice's
   matmuls run through them, about a third faster per step
   ([13](#13-burns-matmul-is-slow-on-rdna2-so-omnivoices-matmuls-use-hipblas-fork)).
   This machine uses AMD's ROCm 7.1.1 libraries in `/opt/rocm`, which are on
   the linker's search path (`/etc/ld.so.conf.d`); Ubuntu's `libhipblas3`
   and `librocblas5` should also do, but haven't been tried. Without them,
   burn's own matmul runs, as before.
5. **Run** as usual. `Model loaded in … on rocm:0` says the backbone is on the
   GPU. No environment variables are needed: on an RX 6600 and other
   gfx1031 to gfx1036 cards, the commands that run OmniVoice set
   `HSA_OVERRIDE_GFX_VERSION=10.3.0` themselves and say so
   ([5](#5-hsa_override_gfx_version-dropped-then-set-automatically-for-hipblas)). `OMNIVOICE_BURN_F16=1` stores the
   backbone's projection weights in F16, which is about 19% faster per step and
   took steady VRAM on a chapter from 7.26 to 5.8 GB, at the cost of tokens no
   longer matching F32 exactly ([measurements](#burn-023-f16-and-fixed-shapes)).

The first run downloads OmniVoice (3.1 GB) and Parakeet (2.4 GB) into the Hugging
Face cache. It also compiles every GPU kernel with LLVM, which makes it much
slower than later runs; compiled kernels and autotune results are cached in
`~/.cache/cubecl` ([7](#7-kernels-were-compiled-again-on-every-run)).

## Problems found, and what was done about each

### 1. ONNX Runtime needs AVX2

ort's prebuilt ONNX Runtime binaries need AVX2. On the FX-8350, `sections`
printed ort's warning and stopped with SIGILL (illegal instruction) as soon as
Parakeet started, so every build that checks takes with ONNX Runtime fails on a
CPU without AVX2, including the CPU-only build.

**Done:** the `vulkan` and `rocm` builds run Parakeet on burn through
parakeet-rs's `BurnWgpu`/`BurnRocm` providers (parakeet-rs 0.4 runs every model
on burn from the same `.onnx` files), and `Checker::start` doesn't initialise
ONNX Runtime at all in those builds (`src/checker.rs`, `execution_config`).

**Alternatives, not tried:** build ONNX Runtime from source without AVX2 and
point ort at it (`ORT_LIB_LOCATION`); or use parakeet-rs's `burn` feature (burn on
the CPU) for the CPU-only build.

### 2. gemm 0.19 panics on CPUs with FMA but not AVX2 (fork)

gemm, which candle's CPU backend and burn's CPU backend (`burn-flex`) use, picks
its FMA kernels when the CPU has FMA, but those kernels run through pulp's `V3`
level, which also needs AVX2. On the FX-8350, `V3::try_new()` returned `None` and
gemm panicked on `unwrap` inside `matmul`. Found when Parakeet's decoder, which
parakeet-rs runs on burn's CPU backend, crashed; OmniVoice on the CPU crashes the
same way.

**Done:** [tommymcc/gemm](https://github.com/tommymcc/gemm), branch
`fma-needs-avx2` (dbdcb05): the three dispatch checks also require AVX2, so such
CPUs fall back to the scalar kernels. Patched in `Cargo.toml`
(`[patch.crates-io]`, `gemm-common` and `gemm-f16`).

**Notes:** upstream (sarah-quinones/gemm) has had no release since 0.19.0. gemm
0.18.2, which candle pulls in through `ug` for `cuda`/`metal` builds, has the same
bug but isn't built for CPU, `vulkan` or `rocm`. The scalar fallback is slow:
OmniVoice entirely on this CPU took 691 s to make 3.4 s of audio.

### 3. pliron 0.19 resolves to a hashbrown too old to build with (fork)

pliron (under burn's CubeCL compiler) imports `hashbrown::Equivalent` but asks for
`hashbrown = "0"`. Cargo resolved it to 0.11.2, already in the graph for tract's
`string-interner` (DeepFilterNet), which has no `Equivalent`, and pliron failed to
compile. pliron 0.20 fixed it (upstream PR #302 pins `0.17.1`), but CubeCL 0.11
needs pliron 0.19.

**Done:** [tommymcc/pliron](https://github.com/tommymcc/pliron), branch
`v0.19-hashbrown` (ac2cc0c), from tag v0.19.0 with the requirement backported.
Patched in `Cargo.toml`.

**Hacks tried first and removed:** an optional `hashbrown` dependency with the
`equivalent` feature (didn't help, as pliron wasn't on 0.17), then hand-editing
`Cargo.lock` to point pliron at 0.17.1 (worked, but `cargo update` could undo it).

**Better way later:** a CubeCL release that depends on pliron 0.20.

**No longer needed** from branch `amd-burn-0.23`: burn 0.23's CubeCL 0.12 uses
pliron 0.20 ([10](#10-parakeet-rs-pins-burn-022-fork)), so the patch was
removed (Cargo had been warning that it was unused).

### 4. cubecl-hip-sys has no bindings for Ubuntu's HIP (fork)

burn's ROCm backend runs on CubeCL, whose `cubecl-hip-sys` ships pre-generated
bindings for particular HIP patch versions and picks one from `hipconfig
--version`. Ubuntu 26.04 builds HIP 7.1.1 itself, and its `hipconfig` reports
**52801**, where AMD's own 7.1.1 reports 52802. With no bindings for 52801,
`cubecl-hip-sys` built empty and `cubecl-hip` failed with 87 unresolved imports.

**Done:** [tommymcc/cubecl-hip-sys](https://github.com/tommymcc/cubecl-hip-sys),
branch `hip-52801` (26e8dea), from tag v7.14.6085001, adds `bindings_52801.rs`
generated from Ubuntu's headers in `/usr/include` with the same settings as the
crate's `cargo xtask bindgen`, plus the `hip_52801` feature and module. The
bindings declare the same 1476 HIP and hipRTC items as `bindings_52802.rs`; the
rest of the difference is glibc's. Patched in `Cargo.toml`.

The repository's own xtask couldn't be run (blocked by Claude Code's
permissions), so the bindings were generated with a small tool using the same
bindgen version and settings: `omnivoice-burn-spike/tools/hipbindgen`. The
crate's own tests (`cargo xtask test`) haven't been run on the fork.

**Why this rather than installing AMD's toolchain:** this machine already had
AMD's `hip-dev` 7.1.52802, but Ubuntu's `hipcc` package supplies the
`/usr/bin/hipconfig` on `PATH`. Switching would mean adding AMD's ROCm repo (whose
packages are built for Ubuntu 24.04) and replacing Ubuntu's `hipcc`. Another user
of Ubuntu 26.04 is most likely to have Ubuntu's ROCm (`apt install rocm`), so
supporting 52801 is what they'd need.

**Hack tried first and removed:** a `hipconfig` wrapper earlier on `PATH` that
reported `7.1.52802-spike` (cubecl-hip-sys's parser needs a suffix after the patch
number), followed by `cargo clean -p cubecl-hip-sys`, since its build script only
reruns when `ROCM_PATH` or `HIP_PATH` change. It worked, and showed the 52802
bindings run against Ubuntu's 52801 library, but it isn't something to ask of
users.

**Better way later:** upstream adding 52801, or falling back to the nearest
bindings within a minor version.

### 5. `HSA_OVERRIDE_GFX_VERSION`: dropped, then set automatically for hipBLAS

ROCm's BLAS libraries ship kernels for gfx1030 but not gfx1032, so tools that use
them on an RX 6600 need `HSA_OVERRIDE_GFX_VERSION=10.3.0` (this machine's Ollama
service sets it). any-tts briefly set it automatically, reading
`gfx_target_version` from `/sys/class/kfd/kfd/topology/nodes/*/properties`.

**Removed:** burn doesn't use rocBLAS; CubeCL compiles its own kernels with LLVM
for the GPU it finds, gfx1032 included. The backbone and the checker ran the same
with and without the override, so it was dropped rather than kept as an
unexplained environment change. It's still needed for the
[feasibility test](#why-burn-and-the-alternatives-considered) programs, which use
hipBLAS.

**Set again, automatically, since OmniVoice's matmuls use hipBLAS**
([13](#13-burns-matmul-is-slow-on-rdna2-so-omnivoices-matmuls-use-hipblas-fork)).
Without it rocBLAS stops the program: `Cannot read …/TensileLibrary.dat … for
GPU arch : gfx1032`. Neither AMD's rocBLAS 7.1.1 nor Ubuntu's 7.1.0
(`librocblas5`, checked by listing the package) has gfx1032 kernels; both
have gfx1030's. gfx1031 to gfx1036 share gfx1030's instruction set (they
differ in size), so the override runs gfx1030's kernels on them; it's what
Ollama and most RX 6600 users do. Building rocBLAS for gfx1032 would give the
same kernels for hours of build, again on every ROCm update.

`any_tts::hipblas::apply_gfx_override`, called from `main`, sets it only when
all of these hold, and prints a line when it does
(`GPU gfx1032: using gfx1030's rocBLAS kernels (HSA_OVERRIDE_GFX_VERSION=10.3.0)`):

- the build has `rocm`, and the command runs OmniVoice (`speak`, `book`,
  `retake`, `audition`); not `sections`, `clean`, `play` or `--help`;
- every GPU's `gfx_target_version` in
  `/sys/class/kfd/kfd/topology/nodes/*/properties` is 100301 to 100306
  (gfx1031 to gfx1036). It's read from there because HIP reads the variable
  once, when it starts, so the decision has to come before anything touches
  HIP;
- the variable isn't already set (a user's value wins), and `ANY_TTS_HIPBLAS`
  isn't `0`;
- hipBLAS loads. The variable is set just before loading it, because
  `hipblasCreate` starts HIP, and taken back if loading fails (no HIP call has
  been made then).

It changes the GPU HIP reports to burn as well, so burn compiles and tunes its
own kernels for gfx1030 once more, into its cache: the first run after the
change was slow (decode 56 s, the checker 80 s), later ones not. A first
version checked for hipBLAS before setting the variable; loading it started
HIP, rocBLAS still saw gfx1032, and the run aborted. If AMD adds gfx1032 to
rocBLAS, the range check is the line to remove.

### 6. Smaller things

- **Rust version:** the README said 1.85+; dependencies need 1.92. README updated.
- **Stale Cargo index cache:** resolving burn failed with versions missing that
  crates.io had (`data-encoding` 2.11.1, `uuid` 1.26.1). The local sparse-index
  cache (`~/.cargo/registry/index/*/.cache`) was stale; deleting it fixed
  resolution. It's only a cache.
- **`combine` in any-tts's lockfile:** any-tts had `combine` 4.6.7 locked, and
  pliron needs `Hash` on `SourcePosition`, which 4.6.9 has. `cargo update -p
  combine`.
- **An any-tts test fails on this CPU:**
  `layers::mlp::tests::composite_mlp_runs_sigmoid_silu_and_fused_matches_it`
  finds the fused SiLU MLP 1.53e-5 from the reference against a tolerance of
  1e-5, with or without `rocm`. On this CPU gemm runs its scalar kernels (the fork
  in [2](#2-gemm-019-panics-on-cpus-with-fma-but-not-avx2-fork)), which round
  differently. Left as is; the tolerance may be too tight for scalar kernels.
- **`speak` prints "0.00x realtime"** when generation is far slower than real
  time; the ratio rounds to zero at two decimal places. Not fixed.

### 7. Kernels were compiled again on every run

`perf` showed the CPU in LLVM's AMDGPU code generator during generation: CubeCL
compiled every kernel again in each process. The first backbone pass at a new
length took 15.4 s (484 tokens) and 4.8 s (84 tokens), about 20 s a run.
`~/.cache/cubecl` only held autotune results; CubeCL's cache of compiled kernels
is off unless its runtime configuration sets `compilation.cache`.

**Done:** `any_tts::burn_device::prepare_rocm` loads CubeCL's configuration
(from a `cubecl.toml` in the working directory or a parent, if there is one),
turns on `compilation.cache`, and installs it before burn first touches the GPU.
OmniVoice and the checker both call it. Those first passes then took 0.45 s and
0.11 s, and token generation for the test sentence went from 38.4 s to 18.8 s.

The cache is keyed to the binary's build, so the first run after each rebuild
compiles every kernel again (a `speak` of the test sentence took 79 s rather than
59 s).

**Tried first:** a `cubecl.toml` with `[compilation] cache = true` in the working
directory, which proved the gain but only works when run from that directory.

### 8. burn's attention autotune picks a slow kernel on RDNA2

The backbone took about 0.55 s per generation step in audio-berry but 0.32 s in
the spike, with the same layers. The difference is burn's `autotune` feature,
which parakeet-rs turns on (it takes burn with its default features) and the
spike didn't have. Building the spike with each of burn's defaults alone, batch
1, 484 + 84 tokens:

| burn features | Per step |
|---|---|
| `rocm` | 0.325 s |
| `rocm`, `fusion` | 0.320 s |
| `rocm`, `autotune` | 0.503 s |
| `rocm`, `autotune`, `fusion` | 0.495 s |

The autotune results in `~/.cache/cubecl/default.db` (SQLite, values in CBOR)
show why. For matmul, autotune picks the same kernel burn uses without it. For
attention at 512 tokens it picked the flash `unit` kernel at 7.9 ms per layer
and **skipped** the fallback (Q Kᵀ matmul, softmax, matmul with V), which takes
about 0.8 ms. This is by design in burn-cubecl 0.22.1
(`kernel/attention/tune.rs`): the fallback materialises the whole score matrix,
so it only competes when `seq_q <= head_dim` (128 here). Longer sequences get the
flash kernels, and on a GPU without matrix cores only the `unit` one runs.
Without `autotune`, burn always uses the fallback, which is why the spike was
faster. No CubeCL runtime setting changes this: the autotune level scales how
many candidates are timed, not which priority group is allowed.

**Done:** any-tts's burn backbone no longer calls burn's `attention`. It computes
softmax(Q Kᵀ / √d) V itself, as burn's fallback does, with the causal mask when
asked for. The query heads that share a KV head are stacked along the rows (Q
reshaped to (batch, kv_heads, group × seq, head_dim)), so K and V aren't copied
for each query head as `repeat_interleave` did. A test checks it against
burn's `attention`, causal and not (skipped without a ROCm device). Result in
`speak`: backbone 17.6 s → 11.4 s for 32 steps (0.55 → 0.36 s per step); the
generated audio transcribes to the same words. any-tts commit `ad03f0a`.

**Not done:** turning burn's `autotune` off for audio-berry. It would need
parakeet-rs forked to take burn without default features, the checker may gain
from autotune elsewhere (convolutions), and the fix above doesn't depend on it.
This matters to anything else run through burn's `attention` on RDNA2 with long
sequences; the checker's Parakeet encoder may be one, which hasn't been checked.

### 9. burn's convolutions are slow on RDNA2, so the audio decoder doesn't use them

OmniVoice's audio decoder (codes to waveform: codebook lookups, `fc2`, then a
DAC decoder of five upsampling blocks, each a transposed conv and three dilated
kernel-7 residual convs) ran on candle's CPU device at about 1.3 s per second
of audio. It is now in any-tts on burn (`burn_audio_decoder.rs`, `rocm`
feature), used instead of candle's whenever the backbone runs on ROCm.

burn's conv ops were the obvious way to write it, and they're slow on this GPU.
On a GPU without matrix cores burn-cubecl runs a direct convolution kernel
(its implicit-GEMM kernels need CMMA), with or without autotune. Decoding 3 s of
audio (75 frames), after kernels are compiled:

| How the convolutions run | Time |
|---|---|
| candle, CPU (before) | 4.2 s |
| burn `conv1d` / `conv_transpose1d`, no `autotune` | 0.52 s |
| burn `conv1d` / `conv_transpose1d`, `autotune` (as audio-berry builds burn) | 0.37 to 0.62 s |
| **as matmuls (what's used)** | **0.037 s** |

So the convolutions are written as matmuls, which run on cubek's matmul at its
measured rate:

- **conv1d** (stride 1, "same" padding): the kernel's k shifted views of the
  zero-padded input are stacked along the channels, (batch, k × in, len), and
  multiplied by the weight laid out as (out, k × in). This costs a copy of the
  input k times over (about 320 MB at the last block for a 15 s chunk), which the
  8 GB card has room for.
- **conv_transpose1d** (DAC's kernels are twice the stride s): one matmul gives
  each input frame's 2s output samples; output frame n is the first half of
  input frame n's plus the second half of frame n − 1's (overlap-add), and the
  padding crops the ends.

Tests check both against burn's conv ops at DAC's settings, and an ignored test
checks the whole decoder against candle's with the real weights (max difference
3e-6 for 3 s, 7e-6 for 15 s; `cargo test --release --features rocm --lib
decoder_matches_candle -- --include-ignored --nocapture`, with
`OMNIVOICE_DECODER_TEST_FRAMES` for other lengths). `OMNIVOICE_BURN_CONV=burn`
switches the decoder to burn's conv ops, for comparison. any-tts commit
`f59fd94`.

Longer chunks: 15 s of audio decodes in 0.21 s (candle: 22.9 s). The first
decode at a new length can take from 0.3 s to 34 s while kernels compile and
matmul autotune runs for the new shapes; both are cached on disk
(`~/.cache/cubecl`), so a book run pays this only for lengths it hasn't seen.

### 10. parakeet-rs pins burn 0.22 (fork)

F16 matmuls only launch on RDNA2 from burn 0.23 (CubeCL 0.12, cubek 0.4):
on 0.22 the F16 `Auto` matmul fails to launch, and on 0.23.0-pre.2 it runs and
is 10 to 15% faster than F32 ([profile](#where-the-gpu-time-goes-rocprofv3)).
parakeet-rs 0.4.4 depends on `burn = "0.22"`, and the checker and OmniVoice
must share one burn: two would mean two copies of CubeCL and its runtimes in
one binary.

**Done:** [tommymcc/parakeet-rs](https://github.com/tommymcc/parakeet-rs),
branch `burn-0.23` (33445ef), from tag v0.4.4: `burn` and `burn-store` at
`=0.23.0-pre.2`, and its generated model code (`src/burn_backend/generated`)
adapted to 0.23's `attention` signature, which gained a KV-cache argument and
`AttentionModuleOptions` fields (a script added `None` and
`..Default::default()` to the 320 calls). Patched in `Cargo.toml` with
`[patch.crates-io]`, as the fork keeps version 0.4.4. any-tts moved to burn
0.23 on its branch `rocm-burn-0.23` the same way. The cubecl-hip-sys fork
([4](#4-cubecl-hip-sys-has-no-bindings-for-ubuntus-hip-fork)) works unchanged
with CubeCL 0.12.

The checker's output is unchanged: `sections` on the 120 s LibriVox clip gives
the same transcripts as on 0.22.

**Why a pre-release:** 0.23.0-pre.2 is what has the F16 fix; `=` pins it so a
later pre-release can't arrive unannounced. **Better way later:** burn 0.23
final, and a parakeet-rs release on it.

### 11. burn tuned kernels again for nearly every chunk

The F16 chapter ran at about 58 s per 20 s chunk, though one 21 s chunk took
about 21 s once its kernels were tuned. burn autotunes each operation per
shape and keeps the results in `~/.cache/cubecl/default.db` (an SQLite table
keyed by operation and sizes). Several of OmniVoice's shapes depend on a
chunk's exact length (the sequence, the number of keys, the output window,
the decoder's frames), and chunks are all different lengths, so nearly every
chunk paid for tuning.

**Done** (any-tts `9a85c8c`):

- **Passes padded to a multiple of 16 rows.** Padded positions get the
  codebooks' first embedding row; padded keys repeat the last real key, with
  V zero there, so they can't change a softmax; padded query rows are thrown
  away. Each pass's output window (the target audio) is also padded, so the
  audio head and token picking see few shapes too.
- **Decoder input padded to a multiple of 32 frames,** with the padding zeroed
  before every convolution and transposed convolution. That is what the
  convolutions' own zero padding gives the unpadded input, so the output is
  the same to 3e-6 against candle over the whole waveform. A first version
  repeated the last frame instead and changed the last 0.5 s of every chunk
  (by up to 0.63).
- **The two passes as one stack of rows,** the conditional pass's rows then
  the unconditional's, apart only in attention (each segment attends within
  itself, causally or not). The projections then read each weight once per
  step instead of twice, and the 521-row unconditional pass no longer runs
  its own badly occupied matmuls.
- **q, k and v merged into one weight, and gate and up into another,** so
  each layer runs 4 projection matmuls instead of 7.

F32 tokens for the test chunk are identical to before (4170 of 4170 with
`OMNIVOICE_SEED=7`). Bucket size, F32, per step for the 21 s chunk:

| Rows padded to a multiple of | Per step |
|---|---|
| (no padding) | 0.699 s |
| **16** | **0.701 s** |
| 32 | 0.751 s |
| 64 | 0.753 s |
| 128 | 0.830 s |

Before stacking and merging the step took 0.719 s, so those two together
gained about 3%, much less than hoped: matmul speed here depends more on how
a shape falls on the tiles than on reading weights once. 64 rows of padding
added 11% to the rows and lost more than stacking gained.

Measured on the chapter: 94 minutes to 44 ([`book`](#book-one-chapter-metamorphosis-chapter-i)).

### 12. Unexplained: one garbled F16 chunk, and one machine freeze

- **A garbled chunk.** Once, the first F16 run of the new shapes produced
  0.4 s of audio: the first step tuned kernels (25 s) and every step after it
  gave wrong tokens. The next run, with the kernels tuned, was right. Three
  runs from an empty cache (`XDG_CACHE_HOME` pointed at a new directory), with
  both the new and the previous build, all said the right words, so it hasn't
  been reproduced. A guess: a candidate kernel tried during tuning wrote out
  of bounds. In `book` the checker would catch such a take and retry it.
- **A machine freeze.** The machine hung, with nothing in the kernel log,
  30 to 60 s into an F16 chapter run, and stayed hung until reset. It was
  still loading or preparing the voice on the CPU, as far as the logs show.
  The same evening LLVM segfaulted during an LTO build, and the boot of
  20 October also ended without a shutdown, before any of this work, so the
  hardware is as likely a cause as the GPU code. After the reset, F32 and
  F16 test chunks and the whole chapter ran without a problem. Long runs now
  `sync` every 2 s so that another freeze leaves its logs.

### 13. burn's matmul is slow on RDNA2, so OmniVoice's matmuls use hipBLAS (fork)

[Profiled](#where-the-gpu-time-goes-rocprofv3) after fixed shapes, a 21 s
chunk's step (0.58 s, F16) was 58% projections, at about 3.8 TFLOP/s in
cubek's register-tiled kernel, and 20% attention matmuls (F32). hipBLAS at the
same shapes (`omnivoice-burn-spike/feasibility/hipblas-shapes`):

| Per layer | burn | hipBLAS F32 | hipBLAS F16 |
|---|---|---|---|
| Projections (q/k/v, o, gate/up, down; 1504 rows) | 12.2 ms (F16) | 7.0 ms | 4.9 ms |
| Attention matmuls (16 heads, 992 and 528 tokens) | 4.3 ms (F32) | 2.5 ms | 1.3 ms |

Writing a faster kernel in burn was ruled out: the blueprint sweep found at
most 20%, and F16 accumulation gave no speedup, so CubeCL's code may not be
using RDNA2's packed F16 instructions at all. Calling hipBLAS from inside burn
(a fork of burn) would speed up the checker too, but means carrying a fork of
a fast-moving pre-release.

**Done** (any-tts `627e9da`, `src/hipblas.rs`):

- **A burn backend extension, `BlasMatmul`.** burn 0.23's
  `#[backend_extension(Cube, Fusion)]` adds an operation that takes its place
  in burn's lazily fused queue. Its `CubeBackend` implementation gets each
  buffer's device address from CubeCL (`client.get_resource::<HipServer>`),
  allocates the output, and calls `hipblasGemmStridedBatchedEx` (F32, or F16
  in and out with F32 accumulation). A row-major (…, m, k) · (…, k, n) is
  computed as the column-major Cᵀ = Bᵀ Aᵀ; each operand is taken row- or
  column-major as it lies (the weights are transposed views), and batch
  dimensions collapse to one stride or broadcast. Anything else (another
  dtype, a layout that doesn't fit, another runtime) falls back to burn's
  `float_matmul`, as does the whole thing with `ANY_TTS_HIPBLAS=0` or no
  hipBLAS. The backbone's projections, attention matmuls and audio head go
  through `hipblas::matmul`.
- **hipBLAS loaded at run time** with `libloading`, as CubeCL loads HIP, so
  it isn't a build dependency. It loads `libamdhip64.so` by the same name
  CubeCL does, so both share one HIP runtime (here `/opt/rocm`'s 7.1.1, which
  is first on the search path).
- **burn's own HIP stream (fork of cubecl-hip).** The first version ran
  hipBLAS on HIP's default stream, with burn's queue synchronised before each
  call and the device after it: 0.507 s per step, only 13% faster, because
  the GPU sat idle for about 6 of 17.6 s of generation while the CPU caught
  up at every one of about 170 barriers a step. CubeCL keeps its stream
  private (`Stream::sys` is `pub(crate)`), so a fork of cubecl-hip
  0.12.0-pre.2 hands it out: `GpuResource` gains a `stream` field, which
  `HipServer::get_resource` fills with the stream it has just ordered the
  buffer on (16 lines; `~/workspace/forks/cubecl-hip-0.12`, a git repo
  started from the crates.io source). hipBLAS is then set to that stream and
  runs in order with burn's kernels, with no waiting. A freed buffer that
  burn reuses is only used by later work on the same stream, so it can't
  overwrite one hipBLAS hasn't read.
- **burn's `fusion` and `extension` features** are now on in any-tts's
  `rocm` (fusion was on already through parakeet-rs). Building any-tts alone
  then needed tokio 1.53 or later (burn-remote, pulled in by the lock).

Tested against burn's matmul (`hipblas::tests`, with a count of calls that
actually went through hipBLAS): projections, batched attention, F16, and a
sliced batch that falls back. The test chunk's words were right.

Measured: see [hipBLAS](#hipblas).

## Why burn, and the alternatives considered

The question was how to run OmniVoice, written for candle, on an AMD GPU in a way
that stays maintainable.

| Approach | Outcome |
|---|---|
| **candle HIP backend in a fork** | Feasible: candle's CUDA kernels compile and run on the RX 6600 after hipify and about 120 lines of fixes (`omnivoice-burn-spike/feasibility/candle-hip`). Rejected: a whole GPU backend (~3,900 lines plus a HIP runtime layer) to maintain alone against candle's changes. |
| **burn** (chosen) | burn's CubeCL backends (ROCm, CUDA, Vulkan/wgpu, Metal, CPU) are maintained upstream, burn 0.22 picks the backend at run time from the `Device`, and parakeet-rs already runs on burn 0.22, so both models share one framework. |
| ONNX Runtime with MIGraphX | Vendor-maintained, but needs an ONNX export of OmniVoice, a source build of ONNX Runtime, and MIGraphX recompiles for every input shape. Not tried. |
| ggml (llama.cpp's library) | Widest backend support and proven on this GPU through Ollama, but C/C++ through FFI. Not tried. |
| Python sidecar with PyTorch ROCm | Quickest to work, but leaves the single Rust binary. Not tried. |

## Measurements (RX 6600, FX-8350)

### How to measure

- **Phase timings:** `OMNIVOICE_TIMINGS=1` makes any-tts print how long
  `synthesize` spent preparing the voice, generating tokens (with the backbone's
  share), and decoding audio, and with `rocm`, each backbone pass.
- **CPU profiles:** `perf` needs `kernel.perf_event_paranoid` at 2 or lower
  (`sudo sysctl kernel.perf_event_paranoid=2`, until reboot); Ubuntu's default
  is 4. Then `perf record -F 199 -g audio-berry …` and `perf report --time
  <from>%-<to>%` for one phase.
- **GPU timing pitfall:** burn queues GPU work lazily. Timing a pass by reading
  its output with `into_data()` and starting the next can leave work from one
  pass running into the next, which made back-to-back passes look 3 to 10 times
  faster than they are. Time many passes as the real loop runs them, with
  `Device::sync()` at the end.

### OmniVoice, end to end (`speak`, 3.3 s of audio, 14.3 s voice sample)

| Phase | CPU build | `rocm`, first run | `rocm`, kernels cached | `rocm`, attention written out ([8](#8-burns-attention-autotune-picks-a-slow-kernel-on-rdna2)) | `rocm`, audio decoder on the GPU ([9](#9-burns-convolutions-are-slow-on-rdna2-so-the-audio-decoder-doesnt-use-them)) |
|---|---|---|---|---|---|
| Load model | 62 s | 6 s | 5 s | 5 s | 5 s |
| Prepare voice (CPU) | | 35.7 s | 35.5 s | 35.5 s | 35.6 s |
| Generate tokens, 32 steps | | 38.4 s | 18.8 s | 12.5 s | 12.5 s |
| …of which the backbone | | 37.3 s | 17.6 s | 11.4 s | 11.4 s |
| Decode audio | | 4.2 s (CPU) | 4.6 s (CPU) | 4.4 s (CPU) | **0.11 s** (GPU) |
| **Total** | **691 s** | **78 s** | **59 s** | **53 s** | **48 s** |

The last column is the second run of a new build: the first recompiles and
autotunes (decode took 43.5 s, the backbone 19.5 s). Its output transcribes to
the sentence. In `book` the voice is prepared once per run, so most of the 35 s
isn't paid per chunk; per chunk, the GPU backbone is what's left.

### `book`, one chapter (*Metamorphosis*, chapter I)

Chapter I of Kafka's *Metamorphosis* (the public-domain Jackson translation,
38,001 characters; the copy in the `epub` crate's test files, as Project
Gutenberg was down), narrated in the user's voice (`voices/me.wav`, 14.3 s used)
with the `rocm` build at any-tts `f59fd94`, kernels cached:

```sh
cd ~/workspace/audio-samples
audio-berry book voices/me.wav metamorphosis.epub out-metamorphosis 4
```

| | |
|---|---|
| Chunks | 110, 1.5 to 26 s (mean 20.5 s) |
| Audio | 37.3 min (2237 s) |
| Wall clock, including loading and the voice | 87.4 min: **0.43× real time**, 2.3 s per second of audio |
| Takes | 110 of 110 passed on the first take; 0 marked `CHECK` |
| Checker recall | mean 0.995, lowest 0.94 (one dropped "Just a moment."); extra words at most 0.05 |
| VRAM (OmniVoice + Parakeet, sampled every 0.5 s) | 7.26 GB steady, one sample at 8.10 GB, of 8.57 GB |
| RAM (peak RSS) | 3.7 GB |

Per chunk, about 47 s for 20 s of audio: almost all of it the backbone (32
steps at these longer sequences); decoding is about 0.2 s and the check well
under a second. Faster narration now depends on the backbone's matmuls (see
below). VRAM is the tighter limit: both models together leave about 0.5 GB of
headroom on an 8 GB card at chunks of up to 28 s.

### Backbone alone (`omnivoice-burn-spike`, 28 layers, F32)

Per generation step at the test sentence's shapes (a 484-token conditional pass
and an 84-token unconditional pass, batch 1), wall clock over 10 steps with one
sync: **0.32 s** with `rocm` alone, and with any burn features once attention is
written out ([8](#8-burns-attention-autotune-picks-a-slow-kernel-on-rdna2)).
That is about 1.7 TFLOP/s (about 555 GFLOP per step). audio-berry's backbone now
takes 0.36 s per step. Layer 0 matches candle to within 2.4e-6 relative on
burn's CPU, Vulkan and ROCm backends.

**Correction:** an earlier version of this section gave 0.65 s (`rocm`) and
0.82 s (with burn's defaults). The spike read its batch size from `SPIKE_batch`,
so the documented `SPIKE_BATCH=1` was ignored and those runs used batch 2. Fixed
in the spike; batch 2 does take 0.65 s.

### Where the GPU time goes (rocprofv3)

`rocprofv3 --kernel-trace --stats` on the spike (`rocm` only), over all its
passes:

| Kernels | Share of GPU time |
|---|---|
| matmul (projections and attention) | 84% |
| reductions (RMSNorm, softmax) | 6% |
| element-wise, copies, slices | 10% |

The GPU was busy for nearly all of the wall-clock time, so launch overhead isn't
what's slow: the matmul kernel is. On a GPU without matrix cores burn runs
cubek's register-tiled `SimpleUnit` matmul: each of 256 threads computes 8×8
outputs, so a workgroup covers 128×128 of the output. At 484 tokens the k, v, o
and down projections are 4 × 8 = 32 workgroups on 28 compute units (HIP reports
14, counting WGPs), and the 84-token pass is 8 to 24 workgroups, so much of the
GPU idles.

Measured directly, without burn (`omnivoice-burn-spike/src/bin/matmul_bench.rs`,
right-hand side column-major as the weights are):

| Projection | M = 484 | M = 84 |
|---|---|---|
| q (1024 → 2048) | 3.15 TFLOP/s | 1.25 |
| k, v (1024 → 1024) | 2.20 | 0.79 |
| o (2048 → 1024) | 2.15 | 0.85 |
| gate, up (1024 → 3072) | 3.25 | 1.72 |
| down (3072 → 1024) | 2.41 | 0.88 |

Every other routine cubek has for this GPU (double-buffered unit, min tile size,
`gemm`, naive) was slower at every shape, which is why matmul autotune
changes nothing. A sweep of `SimpleUnit` blueprints (tile, partition and stage
sizes; 353 that ran, `src/bin/matmul_sweep.rs`) found at best 4 to 20% per shape
(partition 4×1×4, stage 8×32). So cubek's register-tiled kernel tops out around
2 to 3 TFLOP/s on RDNA2 whatever its tiling, against 7.9 for hipBLAS. Closing
that gap needs a different kernel, not tuning; forking cubek-matmul for the
sweep's best blueprint isn't worth the upkeep for at most 20%.

**F16 doesn't help either.** `matmul_bench --dtype f16` runs f16 inputs and
output with an f32 accumulator, as burn launches f16 matmuls, and `--dtype
f16acc` runs f16 throughout, the only way RDNA2's double-rate packed f16 FMA
could be used. `matmul_sweep` takes the same choice (`MATMUL_SWEEP_DTYPE`, with
`MATMUL_SWEEP_QUICK=1` for a handful of blueprints). At the q projection:

| | M = 484 | M = 84 |
|---|---|---|
| f32, burn's default | 2.8 to 3.0 TFLOP/s | 1.3 |
| f16, burn's default (`Auto`) | doesn't launch: "shared memory budget exceeded" | same |
| f16, best routine `Auto` doesn't pick (`gemm`, double-buffered unit) | 0.9 | 0.8 |
| f16, best forced `SimpleUnit` blueprint (8×8×8 tiles) | 2.65 | 1.35 |
| f16 accumulator too | the same as f16 to within 1% | |

f16's vector width is 8, so the unit kernel needs 8-wide tiles, and with burn's
default partition and stage sizes those don't fit in 64 KB of shared memory. The
best blueprint that does is at best as fast as f32 at M = 484, and 5 to 40%
faster at M = 84 (a seventh of the work). Accumulating in f16 changes nothing,
so the kernel doesn't get double-rate f16 math. f16 weights would also need
burn's default blueprint replaced (a fork) and would lose precision, for
little or nothing. Not pursued.

**The next cubek doesn't change f32 either.** burn 0.23.0-pre.2 builds cubecl
0.12.0-pre.2 and cubek 0.4.0-pre.2. The same benchmark against them
(`omnivoice-burn-spike/matmul-next`, its own crate, with the same
cubecl-hip-sys fork) gives the same f32 speed at every shape, to within 3%: the
register-tiled kernel RDNA2 gets hasn't changed. f16 is different there: `Auto`
launches it, and it's faster than f32 at most shapes:

| Projection | M = 484, f32 → f16 | M = 84, f32 → f16 |
|---|---|---|
| q | 2.76 → 3.00 TFLOP/s (+9%) | 1.27 → 1.29 (+2%) |
| k, v | 2.24 → 2.33 (+4%) | 0.79 → 1.12 (+42%) |
| o | 2.18 → 2.63 (+21%) | 0.85 → 1.27 (+49%) |
| gate, up | 3.08 → 3.38 (+10%) | 1.81 → 1.19 (−34%) |
| down | 2.37 → 2.68 (+13%) | 0.88 → 1.31 (+49%) |

So f16 weights would be worth trying once audio-berry is on burn 0.23 (for
about 10 to 15% of the matmul time, and half the backbone's VRAM), but not on
0.22. audio-berry stays on 0.22 because parakeet-rs 0.4.4 pins it.

### Per chunk, after the attention and sampling changes

A 21 s chunk (981-token conditional and 521-token unconditional passes, 32
steps), `speak`, kernels cached, any-tts `f85b678`:

| Part | Runs on | Before | After |
|---|---|---|---|
| Backbone, 32 steps | burn, GPU | 40.6 s | 32.0 s |
| …conditional pass | | 0.76 s | 0.63 s |
| …unconditional pass | | 0.395 s | 0.369 s |
| Picking tokens (log-softmax ×3, guidance, argmax) | candle, CPU → burn, GPU | 7.2 s | inside the backbone figure |
| Embeddings before each pass | candle, CPU | | 3.6 s |
| Generate tokens, in all | | 47.9 s | **35.9 s** |
| Decode audio | burn, GPU | 0.28 s | 0.27 s |

What changed (any-tts `1f2cf68`, `f85b678`):

- **Attention's softmax.** At 981 tokens attention was 36% of a layer and the
  softmax most of that: scale, max, exp, sum and divide, each a pass over 15.4M
  scores, and element by element, as 981 isn't a multiple of 4. Q is now scaled
  before Q Kᵀ; the keys are padded to a multiple of 8 by repeating the last key,
  which can't change a row's max; and V gets a column of ones (zero in the
  padded rows), so P V also gives each row's sum, replacing a reduction and a
  full-size divide. The 981-token layer's attention went from 9.1 to 5.8 ms.
- **Picking tokens on the GPU.** Each step brought both passes' logits back
  (17 MB each) and ran three log-softmaxes, the guidance and an argmax in candle
  on the CPU. With greedy sampling (the default) this now runs on burn, and only
  the picks and their scores come back. With a fixed seed the 4168 tokens are
  identical to the CPU path's. Sampling with a temperature, and the first-step
  debug dump, still take the CPU path.

The GPU time (`rocprofv3`) now splits as: projections 74%, attention matmuls
13%, softmax 6.5%, norms, RoPE, SiLU and copies 6%, audio head 1%. Of the
softmax, the row max is the largest part (1.3 ms per 981-token layer, a
reduction over 62 workgroups).

### Embeddings on the GPU

After that, the largest CPU cost per step was building each pass's input in
candle: looking up and summing the text and eight audio-codebook embeddings
(0.11 s a step, 10% of a chunk). any-tts `61b1caf` splits it:

- **The text part** doesn't change between steps (only audio positions get new
  tokens), so it's looked up once per chunk on the CPU, zeroed at audio
  positions, and copied to the GPU once. The 620 MB text table stays in system
  memory, as VRAM has about 0.5 GB to spare.
- **The audio part** changes every step. The audio embedding tables (8 × 1025
  rows, 33 MB) now live on the GPU next to the backbone; each step uploads only
  the token ids (offset per codebook on the CPU), gathers the rows with
  `select`, sums over codebooks and masks them.

Same 21 s chunk, kernels cached: generate 35.9 s → **31.7 s**, tokens
identical with `OMNIVOICE_SEED=7`. Almost all of generation is now the
backbone's matmuls on the GPU.

**Correction:** earlier spike runs reported 0.375 s per pass at 1024 tokens on
ROCm and 1.89 s on Vulkan, timed with the pitfall above. The ROCm figure was
wrong by several times; the Vulkan figures were measured the same way and
haven't been re-measured. The candle CPU figures (87 s per pass at 1024 tokens)
are unaffected.

### burn 0.23, F16, and fixed shapes

Same 21 s chunk, `speak`, `OMNIVOICE_SEED=7`, kernels tuned (second run):

| Build | Generate tokens | Tokens vs. the row above |
|---|---|---|
| any-tts `61b1caf` (burn 0.22, embeddings on the GPU) | 31.7 s | |
| `672546b`: burn 0.23.0-pre.2, F32 | 23.0 s | identical |
| `672546b`, `OMNIVOICE_BURN_F16=1` | 18.8 s | 1631 of 4168 differ |
| `9a85c8c`: fixed shapes, stacked passes, merged weights, F32 | 22.4 s | identical to F32 above |
| `9a85c8c`, F16 | **18.65 s** | differ (F16 rounding varies with the tuned kernels) |

Decode is 0.25 s either way. burn 0.23 alone took a quarter off F32 (cubek
0.4's matmuls), F16 about another fifth. F16's differing tokens still say the
words: three F16 runs from an empty kernel cache, with 1548 to 3849 tokens
different, all transcribed to the sentence.

Whole chapter, *Metamorphosis* chapter I as [above](#book-one-chapter-metamorphosis-chapter-i):

| | F32, burn 0.22 (`f59fd94`) | F16, exact shapes (`672546b`) | **F16, fixed shapes (`9a85c8c`)** |
|---|---|---|---|
| Wall clock for 37.3 min of audio | 87.4 min (0.43× real time) | 94.6 min (0.39×) | **44.5 min (0.84×)** |
| First takes | 110 of 110 | 110 of 110 | 110 of 110 |
| Recall | mean 0.995, lowest 0.94 | lowest 0.94 | mean 0.995, lowest 0.94 |
| VRAM, steady / peak | 7.26 / 8.10 GB | 5.8 / 7.44 GB | 5.8 / 6.97 GB |
| RAM, peak RSS | 3.7 GB | 4.2 GB | 3.6 GB |

The exact-shape F16 run was slower than F32 despite faster steps because it
tuned again for nearly every chunk ([11](#11-burn-tuned-kernels-again-for-nearly-every-chunk)).
With fixed shapes, tuning stops early: chunks 60 to 110 took 17 minutes, about
20 s each, for a mean of 20.5 s of audio, so a warm cache runs at about real
time. That run is any-tts at the commit shown with audio-berry branch
`amd-burn-0.23`.

### hipBLAS

Same 21 s chunk, `speak`, F16 (`OMNIVOICE_BURN_F16=1`), `OMNIVOICE_SEED=7`,
kernels tuned (second run of each build):

| Build | Per step | Generate tokens | GPU busy during generation |
|---|---|---|---|
| any-tts `9a85c8c` (burn's matmul) | 0.58 s | 18.7 s | nearly all of it |
| hipBLAS on the default stream, barriers around each call | 0.507 s | 17.0 s | about two thirds (11.8 of 17.6 s, under `rocprofv3`) |
| **any-tts `627e9da`, hipBLAS on burn's stream** | **0.381 s** | **13.1 s** | |

Per layer under `rocprofv3` with barriers, hipBLAS's projections took 5.3 ms
(burn's 12.2) and attention's F32 matmuls 3.3 ms (burn's 4.3). The checker
heard the take's words right.

Whole chapter, *Metamorphosis* chapter I, as [above](#book-one-chapter-metamorphosis-chapter-i),
audio-berry `7007d48`:

| | F16, burn's matmul (`9a85c8c`) | **F16, hipBLAS (`627e9da`)** |
|---|---|---|
| Wall clock | 44.5 min (0.84× real time) | **36.8 min (1.02×)** |
| Chunks 60 to 110 | 17 min | 13 min (about 15.6 s per chunk for 20.5 s of audio: 1.3× real time once warm) |
| First takes | 110 of 110 | 110 of 110 |
| Recall | mean 0.995, lowest 0.94 | mean 0.995, lowest 0.94; extra words at most 0.05 |
| VRAM, steady / peak | 5.8 / 6.97 GB | 6.09 / 7.67 GB (of 8.57) |
| RAM, peak RSS | 3.6 GB | 3.7 GB |

This run also paid for burn tuning its kernels again for gfx1030 (the
override changes the GPU burn sees, [5](#5-hsa_override_gfx_version-dropped-then-set-automatically-for-hipblas)),
mostly in the first 20 chunks (12 minutes), so a later run should be quicker
than 36.8 minutes. hipBLAS and rocBLAS take about 0.3 GB more VRAM at steady
state and 0.7 GB more at peak, leaving about 0.9 GB.

### Checker (`sections`, 120 s LibriVox clip)

| Backend | Hearing 120 s | Transcribing 5 stretches |
|---|---|---|
| ONNX Runtime, CPU | crashes (no AVX2) | |
| burn Vulkan | 18 to 26 s | 2.5 to 13.8 s |
| burn ROCm, first run | 61.8 s | 10.7 s |
| burn ROCm, kernels cached | 9.5 s | 3.9 s |

The words of all 5 stretches matched the book on Vulkan and ROCm, identically.

## What's left

- **Voice preparation runs on the CPU** in candle with the scalar gemm kernels:
  35 s per voice (once per `book` run). Porting OmniVoice's audio encoder to
  burn, as the decoder was ([9](#9-burns-convolutions-are-slow-on-rdna2-so-the-audio-decoder-doesnt-use-them)),
  would move it to the GPU.
- **The matmuls now run through hipBLAS** ([13](#13-burns-matmul-is-slow-on-rdna2-so-omnivoices-matmuls-use-hipblas-fork)):
  a step is 0.38 s (F16) for a 21 s chunk, from 0.58. Attention's matmuls
  are still F32; hipBLAS runs them about twice as fast in F16 (F16 inputs
  with F32 scores would keep the softmax precise). The cubecl-hip fork is
  local only so far.
- **F16 hasn't been checked for likeness to the voice**, only for words.
  `audition` pairs of F32 and F16 takes would show whether it matters.
- **burn 0.23 is a pre-release** (`0.23.0-pre.2`), pinned exactly in both forks.
- **The two unexplained events** ([12](#12-unexplained-one-garbled-f16-chunk-and-one-machine-freeze))
  haven't recurred.
- **VRAM:** with F16, 6.97 GB at peak of 8.57 with both models loaded and
  chunks up to 26 s. F32 left about 0.5 GB.
- None of the forks has been offered upstream.

## Where things are

| What | Where |
|---|---|
| audio-berry changes | branch `amd-burn-checker` (burn 0.22); branch `amd-burn-0.23` (burn 0.23, F16, fixed shapes); branch `amd-hipblas` (hipBLAS; any-tts and cubecl-hip as path dependencies, local only) |
| any-tts `rocm` feature | `tommymcc/any-tts`, branch `rocm-burn` (61b1caf, burn 0.22); branch `rocm-burn-0.23` (9a85c8c), pinned in `Cargo.toml` on `amd-burn-0.23`; branch `rocm-hipblas` (627e9da, local only) |
| cubecl-hip handing out its stream | `~/workspace/forks/cubecl-hip-0.12` (local git repo from the crates.io source of 0.12.0-pre.2, not yet on GitHub) |
| hipBLAS benchmarks at OmniVoice's shapes | `omnivoice-burn-spike/feasibility/hipblas-shapes` |
| parakeet-rs on burn 0.23 | `tommymcc/parakeet-rs`, branch `burn-0.23` (33445ef); clone in `~/workspace/forks/parakeet-rs` (its `origin` is upstream; the fork is the remote `fork`) |
| Matmul benchmark and blueprint sweep (f32 and f16) | `omnivoice-burn-spike/src/bin/matmul_{bench,sweep}.rs` (`--features rocm`); against the next cubek: `omnivoice-burn-spike/matmul-next` |
| Spike, feasibility test, bindgen tool | `~/workspace/omnivoice-burn-spike` (local git repo) |
| Fork clones | `~/workspace/forks/{gemm,pliron,cubecl-hip}` |
| Test audio | `~/workspace/test-audio` (LibriVox, public domain) |

No pull requests or issues have been opened upstream for any of the forks.
