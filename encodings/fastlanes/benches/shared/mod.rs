// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared helpers for the fastlanes benchmark binaries.
//!
//! Pulled into a bench via `mod shared;`. The macros here give every benchmark
//! a single source of truth for *which* CPU feature set / architecture it is
//! measured under, driven by the compile-time `BENCH_VARIANT` environment
//! variable:
//!
//! * A plain `cargo bench` leaves `BENCH_VARIANT` at its `local` default (set in
//!   `.cargo/config.toml`), so every benchmark runs once on the host.
//! * The CodSpeed workflow sets `BENCH_VARIANT` to exactly one variant per CI
//!   leg (e.g. `simulation`, `x86_64`, `aarch64`), which both prefixes the
//!   benchmark name and gates whether the benchmark runs.

/// Prefix a benchmark name with the active build variant.
///
/// `BENCH_VARIANT` is read at compile time. The prefix keeps measurements taken
/// under different builds or architectures from colliding in CodSpeed (most
/// importantly the architecture-neutral scalar benchmarks, which run on every
/// leg).
#[macro_export]
macro_rules! variant {
    ($name:literal) => {
        concat!(env!("BENCH_VARIANT"), "::", $name)
    };
}

/// Map a known variant identifier to its string tag.
///
/// Adding a new variant? Add an arm here *and* a matching CI leg. An unknown
/// identifier fails to compile, so benchmark tags can't silently typo into a
/// variant that never runs.
#[macro_export]
macro_rules! variant_tag {
    (simulation) => {
        "simulation"
    };
    (x86_64) => {
        "x86_64"
    };
    (aarch64) => {
        "aarch64"
    };
}

/// divan `ignore` expression: skip this benchmark *unless* we are running
/// locally (`BENCH_VARIANT=local`, the default) or the active variant is one of
/// the listed feature sets. CI sets `BENCH_VARIANT` to exactly one variant per
/// leg; locally it defaults to `local`, so every benchmark runs.
///
/// The gate is an OR-chain of `==` rather than `matches!` because
/// [`variant_tag!`] expands to a string literal, which is not valid in
/// `matches!` pattern position.
#[macro_export]
macro_rules! ignore_unless_variant {
    ($($v:ident),+ $(,)?) => {{
        let active = env!("BENCH_VARIANT");
        !(active == "local" $(|| active == $crate::variant_tag!($v))+)
    }};
}
