# 0009: Cargo workspace

> **Status:** Accepted ADR.
> **Progress:** The repository is a Cargo workspace; new
> infrastructure crates are added as workspace members.
> **Open questions:** none.

## Status

Accepted.

## Context

The engine ships infrastructure crates that need to compile
independently of the engine itself — schema crates that other
tools depend on, replay libraries that target both native and
wasm, and viewer bindings.

A single-crate layout cannot accommodate a wasm-friendly
schema crate that does not pull the whole engine into its
dependency graph. A workspace layout solves that without
requiring the engine to be split into smaller pieces today.

## Decision

The repository is a Cargo workspace. Members today:

- `.` — the `vortex-engine` package at the repo root; no source
  moves.
- `crates/vortex-trace-format` — `no_std`-friendly schemas plus
  framing helpers.
- `crates/vortex-trace-replay` — opens trace files and
  reconstructs scheduler state. Compiles native and to
  `wasm32-unknown-unknown`.
- `viewer/crate` — wasm bindings for the viewer.

The root manifest gains a `[workspace]` section listing those
members. The engine's `[package]` section is unchanged.

## Consequences

- `cargo build`, `cargo test`, and `cargo check` are run with
  `--workspace` in CI to cover all members.
- New crates depend on engine schemas via path dependencies on
  `vortex-trace-format`; they do not depend on `vortex-engine`.
- The `vortex-engine` crate gains `vortex-trace-format` as a
  dependency for its recorder code.
- Future crate splits land as additional workspace members
  without further architectural debate.

## Out of scope

- Splitting the engine itself into multiple crates is a separate
  decision. The engine continues to live as a single crate at the
  repo root.
- Publishing crates to crates.io: not yet, none of these are
  publish-ready.
