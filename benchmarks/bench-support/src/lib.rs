// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared support code for Vortex benchmark binaries.
//!
//! # Benchmark variants
//!
//! A *variant* names the build a benchmark binary was produced for. A benchmark that only
//! makes sense under one CPU architecture, or that is too noisy for one instrument but not
//! another, declares the variants it should be measured under; everywhere else it is skipped
//! rather than compiled out, so a single bench source covers every leg.
//!
//! | Variant | Where it runs | Instrument |
//! |---|---|---|
//! | `local` | a plain `cargo bench` on a developer machine | divan walltime |
//! | `simulation` | the sharded `bench-codspeed` CI job | CodSpeed simulation |
//! | `x86_64` | the `bench-codspeed-arch` x86-64 leg, built with `+avx2` | CodSpeed walltime |
//! | `aarch64` | the `bench-codspeed-arch` aarch64 leg, built with `+neon` | CodSpeed walltime |
//!
//! Two compile-time environment variables carry the variant, both defaulted in
//! `.cargo/config.toml` so a plain `cargo bench` needs no setup:
//!
//! * `VORTEX_BENCH_VARIANT` — the active variant tag, read by [`ignore_unless_variant!`].
//!   Defaults to `local`, under which *every* benchmark runs, whatever it is tagged with.
//! * `VORTEX_BENCH_PREFIX` — prepended to benchmark names by [`variant_name!`]. Empty by
//!   default, so local and simulation runs keep bare names (and, for CodSpeed, their
//!   existing measurement history); only the walltime legs set it, to `<variant>::`.
//!
//! The prefix is what makes it safe for one benchmark to run on more than one leg: an
//! architecture-neutral baseline measured on both the x86-64 and aarch64 legs reports as
//! `x86_64::words_gather_scalar` and `aarch64::words_gather_scalar`, instead of two
//! different machines fighting over one name.
//!
//! # Example
//!
//! ```ignore
//! use vortex_bench_support::ignore_unless_variant;
//! use vortex_bench_support::variant_name;
//!
//! // Runs locally, on the simulation shards, and on both walltime legs.
//! #[divan::bench(
//!     name = variant_name!("from_bool_slice"),
//!     ignore = ignore_unless_variant!(simulation, x86_64, aarch64),
//! )]
//! fn from_bool_slice(bencher: divan::Bencher) { /* ... */ }
//!
//! // NEON only: compiled out elsewhere by `cfg`, and skipped on the aarch64 machine
//! // whenever that machine is running some other leg.
//! #[cfg(target_arch = "aarch64")]
//! #[divan::bench(
//!     name = variant_name!("words_gather_neon"),
//!     ignore = ignore_unless_variant!(aarch64),
//! )]
//! fn words_gather_neon(bencher: divan::Bencher) { /* ... */ }
//! ```
//!
//! # Adding a variant
//!
//! Add an arm to [`variant_tag!`] and a matching leg to `.github/workflows/codspeed.yml`.
//! Benchmarks name variants by identifier, not by string, so a tag that does not exist
//! fails to compile instead of silently never running.

/// Map a benchmark variant identifier to its string tag.
///
/// This is the list of variants that exist. An unknown identifier is a compile error, so a
/// typo in an [`ignore_unless_variant!`] tag cannot silently turn into a benchmark that
/// never runs anywhere. A benchmark behind `#[cfg(target_arch = ...)]` is only checked when
/// building for that architecture, which for a NEON-only benchmark means the aarch64 CI leg
/// rather than a developer's x86-64 machine.
#[macro_export]
macro_rules! variant_tag {
    (local) => {
        "local"
    };
    (simulation) => {
        "simulation"
    };
    (x86_64) => {
        "x86_64"
    };
    (aarch64) => {
        "aarch64"
    };
    ($unknown:ident) => {
        compile_error!(concat!(
            "unknown benchmark variant `",
            stringify!($unknown),
            "`; add it to `vortex_bench_support::variant_tag!` and give it a CI leg",
        ))
    };
}

/// Qualify a benchmark name with the active variant's prefix.
///
/// Expands to a string literal, so it is usable as `#[divan::bench(name = ...)]`. The prefix
/// comes from the compile-time `VORTEX_BENCH_PREFIX` and is empty unless the build sets it,
/// leaving local and CodSpeed simulation names untouched.
#[macro_export]
macro_rules! variant_name {
    ($name:literal) => {
        concat!(env!("VORTEX_BENCH_PREFIX"), $name)
    };
}

/// Skip this benchmark unless the active variant is one of the listed ones.
///
/// Expands to a `bool` for `#[divan::bench(ignore = ...)]`. `local` — the default outside
/// CI — always runs, so a developer never has to know which tags a benchmark carries.
///
/// The check is an OR-chain of `==` rather than a `matches!`, because [`variant_tag!`]
/// expands to a string literal, which is not valid in pattern position.
#[macro_export]
macro_rules! ignore_unless_variant {
    ($($variant:ident),+ $(,)?) => {{
        let active = env!("VORTEX_BENCH_VARIANT");
        !(active == $crate::variant_tag!(local) $(|| active == $crate::variant_tag!($variant))+)
    }};
}
