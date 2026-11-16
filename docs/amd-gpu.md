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
| `--no-default-features --features rocm` | transformer backbone on the GPU with burn ROCm; voice preparation, embeddings, sampling and audio decoding on the CPU with candle | burn on ROCm |

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
4. **Run** as usual. `Model loaded in … on rocm:0` says the backbone is on the
   GPU. No environment variables are needed.

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

### 5. `HSA_OVERRIDE_GFX_VERSION` isn't needed (tried and removed)

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

| Phase | CPU build | `rocm`, first run | `rocm`, kernels cached |
|---|---|---|---|
| Load model | 62 s | 6 s | 5 s |
| Prepare voice (CPU in both) | | 35.7 s | 35.5 s |
| Generate tokens, 32 steps | | 38.4 s | 18.8 s |
| …of which the backbone | | 37.3 s | 17.6 s |
| Decode audio (CPU in both) | | 4.2 s | 4.6 s |
| **Total** | **691 s** | **78 s** | **59 s** |

In `book` the voice is prepared once per run, so most of the 35 s isn't paid per
chunk. Per chunk, the GPU backbone and the CPU audio decoder are what's left.

### Backbone alone (`omnivoice-burn-spike`, 28 layers, F32)

Per generation step at the test sentence's shapes (a 484-token conditional pass
and an 84-token unconditional pass), wall clock over 10 steps with one sync:

| burn features | Per step |
|---|---|
| `rocm` only | 0.65 s |
| `rocm` plus burn's defaults (fusion, autotune; what parakeet-rs brings in) | 0.82 s |

audio-berry's steady state, about 0.55 s per step, matches. Layer 0 matches
candle to within 2.4e-6 relative on burn's CPU, Vulkan and ROCm backends.

This is about 1 TFLOP/s; hipBLAS reached 7.9 TFLOP/s in F32 on the same card. So
burn's kernels, not the hardware, set the pace here, and there's a lot of
headroom.

**Correction:** earlier spike runs reported 0.375 s per pass at 1024 tokens on
ROCm and 1.89 s on Vulkan, timed with the pitfall above. The ROCm figure was
wrong by several times; the Vulkan figures were measured the same way and
haven't been re-measured. The candle CPU figures (87 s per pass at 1024 tokens)
are unaffected.

### Checker (`sections`, 120 s LibriVox clip)

| Backend | Hearing 120 s | Transcribing 5 stretches |
|---|---|---|
| ONNX Runtime, CPU | crashes (no AVX2) | |
| burn Vulkan | 18 to 26 s | 2.5 to 13.8 s |
| burn ROCm, first run | 61.8 s | 10.7 s |
| burn ROCm, kernels cached | 9.5 s | 3.9 s |

The words of all 5 stretches matched the book on Vulkan and ROCm, identically.

## What's left

- **Voice preparation and audio decoding run on the CPU** in candle with the
  scalar gemm kernels: 35 s per voice and about 1.3 s per second of audio. Porting
  OmniVoice's audio encoder and decoder to burn would move them to the GPU.
- **burn's ROCm kernels reach about 1 TFLOP/s** against 7.9 for hipBLAS.
  Profiling the GPU kernels (rocprofv3) would show whether matmul, attention or
  the many small elementwise kernels dominate.
- None of the forks has been offered upstream.

## Where things are

| What | Where |
|---|---|
| audio-berry changes | branch `amd-burn-checker` |
| any-tts `rocm` feature | `tommymcc/any-tts`, branch `rocm-burn` (cc2105b), pinned in `Cargo.toml` |
| Spike, feasibility test, bindgen tool | `~/workspace/omnivoice-burn-spike` (local git repo) |
| Fork clones | `~/workspace/forks/{gemm,pliron,cubecl-hip}` |
| Test audio | `~/workspace/test-audio` (LibriVox, public domain) |

No pull requests or issues have been opened upstream for any of the forks.
