# Vortex Fuzz

This crate contains general fuzzing infrastructure and tooling for all public components of Vortex.

## Setup

Currently, the only thing required to run the fuzzing targets is [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz)

## Reproduce crash from CI

In the case of a crash in the nightly run, you can download the crash artifact and run `cargo-fuzz` with the exact same
input with the command `cargo fuzz run array_ops <path/to/artifact>` or `cargo fuzz run file_io <path/to/artifact>`

## Scheduled fuzzing

Exploration runs in libFuzzer fork mode, including a single worker for the GPU target. The parent
replays the saved seeds to establish coverage, then replaces workers that crash, time out, or run
out of memory until the exploration time budget expires. Failing inputs are saved as artifacts;
the workflow reports failures after exploration and corpus persistence finish. `-keep_seed=1` is
deliberately omitted so new inputs are compared against coverage from the saved seeds.

Exploration and minimization share a concurrency group for each corpus. Different targets can
progress independently. Compiled binaries are cached by commit, toolchain, architecture, target,
and features, and the CPU targets share one build.

Daily minimization uses eight CPU shards (one for GPU), followed by a merge of the reduced outputs.
It reports replay throughput and estimated remaining time every 30 seconds. Only a complete merge
with no failing inputs replaces the stored corpus; partial outputs are discarded on the next run.
Minimization does not save resumable state. An initially oversized corpus can still make the first
exploration replay expensive until minimization succeeds.

### ASAN

If there are any linking (on macOS) then run `cargo fuzz run --dev --sanitizer=none ...`. `--dev` runs the fuzzer in dev
profile.
