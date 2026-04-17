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
//! The compressor emits a small set of `tracing` events so you can see what it's doing
//! without attaching a profiler. Set `RUST_LOG=vortex_compressor::encode=debug` to see one
//! line per leaf decision; that single event covers the bulk of debugging needs.
//!
//! | Target                        | Emits                                                                     |
//! |-------------------------------|---------------------------------------------------------------------------|
//! | `vortex_compressor::cascade`  | `compress` span (with rollup fields on exit), `cascade_exhausted` event.  |
//! | `vortex_compressor::encode`   | `scheme.compress_result` per leaf, `scheme.compress_failed` on error.     |
//! | `vortex_compressor::estimate` | `sample.result` per sampled scheme, `sample.compress_failed` on error.    |
//!
//! The primary event is `scheme.compress_result`, which carries `scheme`, `before_nbytes`,
//! `after_nbytes`, `estimated_ratio` (absent when the scheme returned `AlwaysUse`),
//! `actual_ratio`, and `accepted`. From those fields you can derive per-scheme savings,
//! rejection counts, and estimator accuracy with a short `jq` pipeline.
//!
//! Event names and field names should be considered unstable for now — expect them to change
//! as we gain experience with what's actually useful.

pub mod builtins;
pub mod ctx;
pub mod estimate;
pub mod scheme;
pub mod stats;

mod sample;

mod compressor;
pub use compressor::CascadingCompressor;
