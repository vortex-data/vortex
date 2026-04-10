// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

//! Encoding-agnostic compression framework for Vortex arrays.
//!
//! This crate provides the core compression engine: the [`Scheme`](scheme::Scheme) trait,
//! sampling-based ratio estimation, cascaded compression, and statistics infrastructure for
//! deciding the best encoding scheme for an array.
//!
//! This crate contains no encoding dependencies. Batteries-included compressors are provided by
//! downstream crates like `vortex-btrblocks`, which register different encodings to the compressor.
//!
//! # Observability
//!
//! The compressor emits structured `tracing` spans and events through four independent
//! targets. Pick one with `RUST_LOG` to study a single aspect of the compressor at a time,
//! or combine them. No subscriber is installed by this crate; the caller does that.
//!
//! | Target                          | What it covers                                                                    |
//! |---------------------------------|-----------------------------------------------------------------------------------|
//! | `vortex_compressor::cascade`    | Top-level `compress` and `compress_child` spans — the cascade-tree shape.         |
//! | `vortex_compressor::select`     | Scheme eligibility, per-scheme evaluation, winner, and short-circuit reasons.     |
//! | `vortex_compressor::estimate`   | Sampling: sample sizing, sample compression, and the resulting estimated ratio.   |
//! | `vortex_compressor::encode`     | The winner's encode span and its estimated-vs-actual `scheme.compress_result`.    |
//!
//! ## Recipes
//!
//! Summary of each leaf (which scheme won, estimated vs actual ratio, accepted?):
//!
//! ```text
//! RUST_LOG=vortex_compressor::encode=debug cargo test -p vortex-btrblocks
//! ```
//!
//! Every scheme evaluated for every leaf, with estimate kind and ratio:
//!
//! ```text
//! RUST_LOG=vortex_compressor::select=trace cargo test -p vortex-btrblocks
//! ```
//!
//! Sample sizes and sample compression results:
//!
//! ```text
//! RUST_LOG=vortex_compressor::estimate=trace cargo test -p vortex-btrblocks
//! ```
//!
//! Cascade tree (which scheme cascaded into which child):
//!
//! ```text
//! RUST_LOG=vortex_compressor::cascade=debug cargo test -p vortex-btrblocks
//! ```
//!
//! Everything (firehose):
//!
//! ```text
//! RUST_LOG=vortex_compressor=trace cargo test -p vortex-btrblocks
//! ```
//!
//! Combine targets:
//!
//! ```text
//! RUST_LOG=vortex_compressor::encode=debug,vortex_compressor::estimate=debug cargo run ...
//! ```
//!
//! ## Span inventory
//!
//! | Span                            | Target    | Level | Key fields                                              |
//! |---------------------------------|-----------|-------|---------------------------------------------------------|
//! | `CascadingCompressor::compress` | cascade   | trace | `len`, `nbytes`, `dtype`                                |
//! | `compress_child`                | cascade   | trace | `parent`, `child_index`, `cascade_depth`, `len`         |
//! | `choose_and_compress`           | select    | trace | `dtype`, `len`, `cascade_depth`, `eligible_count`       |
//! | `estimate.sample`               | estimate  | trace | `scheme`, `source_len`                                  |
//! | `scheme.compress`               | encode    | trace | `scheme`, `before_nbytes`                               |
//! | `sample.compress`               | encode    | trace | `scheme`                                                |
//!
//! ## Event inventory
//!
//! | Event                       | Target            | Level | Fields                                                                                          |
//! |-----------------------------|-------------------|-------|-------------------------------------------------------------------------------------------------|
//! | `scheme.evaluated`          | select            | trace | `scheme`, `kind`, `ratio` (Option)                                                              |
//! | `scheme.evaluated.resolved` | select            | trace | `scheme`, `kind`, `resolved_kind`?, `ratio`?                                                    |
//! | `scheme.winner`             | select            | debug | `scheme`, `estimated_ratio`, `candidate_count`                                                  |
//! | `scheme.compress_result`    | encode            | debug | `scheme`, `before_nbytes`, `after_nbytes`, `estimated_ratio`, `actual_ratio`, `accepted`        |
//! | `sample.collected`          | estimate          | trace | `scheme`, `sample_count`, `sample_size`, `sampled_len`, `source_len`                            |
//! | `sample.result`             | estimate          | debug | `scheme`, `sampled_before`, `sampled_after`, `sampled_ratio`                                    |
//! | `short_circuit`             | select / cascade  | debug | `reason` (`cascade_exhausted` \| `no_schemes` \| `empty` \| `all_null` \| `fell_through` \| `larger_output`), scheme?/parent? |
//!
//! An `estimated_ratio` of [`f64::INFINITY`] indicates a scheme that returned
//! [`CompressionEstimate::AlwaysUse`](estimate::CompressionEstimate::AlwaysUse).
//!
//! Field names are considered stable and are meant to be matched directly by downstream
//! observability tooling. This means `tracing-opentelemetry`, `tracing-perfetto`, and
//! `tracing-timing` subscribers work with no adapter code — attach a layer in your binary's
//! subscriber registry and the spans/events will be captured.
//!
//! ## Plugging in a subscriber
//!
//! A minimal stderr-only setup:
//!
//! ```rust,ignore
//! use tracing_subscriber::EnvFilter;
//! use tracing_subscriber::layer::SubscriberExt;
//! use tracing_subscriber::util::SubscriberInitExt;
//!
//! tracing_subscriber::registry()
//!     .with(EnvFilter::from_default_env())
//!     .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
//!     .init();
//! ```
//!
//! To capture timings or export to a collector, add a second layer
//! (`tracing_perfetto::PerfettoLayer`, `tracing_timing::Builder`,
//! `tracing_opentelemetry::layer(...)`) to the registry.

pub mod builtins;
pub mod ctx;
pub mod estimate;
pub mod scheme;
pub mod stats;

mod sample;

mod compressor;
pub use compressor::CascadingCompressor;
