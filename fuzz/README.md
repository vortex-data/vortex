# Vortex Fuzz

This crate contains general fuzzing infrastructure and tooling for all public components of Vortex.

## Setup

In order to run the fuzzer you'll need (up to) three things:

1. A nightly toolchain
1. [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz)
1. If you wish to run/build some of the code directly, setting `RUSTFLAGS="--cfg fuzzing"`, which is automatically set when running `cargo fuzz`.

## Reproduce crash from CI

In the case of a crash in the nightly run, you can download the crash artifact and run `cargo-fuzz` with the exact same
input with the command `cargo +nightly fuzz run array_ops/file_io <path/to/artifact>`

### ASAN

If there are any linking (on macOS) then run `cargo +nightly fuzz run --dev --sanitizer=none ...`. `--dev` runs the fuzzer in dev profile.
