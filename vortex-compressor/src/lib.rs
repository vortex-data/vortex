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
//! The compressor emits a small set of `tracing` events on a single target so you can see what
//! it's doing without attaching a profiler.
//!
//! For example, set `RUST_LOG=vortex_compressor::encode=debug` to see one line per leaf compression
//! decision. The `vortex_compressor::encode` target carries the main decision events
//! (`scheme.compress_result`, `sample.result`, and both `*.compress_failed`) plus the coarse
//! top-level `compress` span and `cascade_exhausted` event.
//!
//! The primary event is `scheme.compress_result`, which carries `scheme`, `before_nbytes`,
//! `after_nbytes`, `estimated_ratio` (absent when the scheme returned `AlwaysUse` or sampled to 0
//! bytes), `actual_ratio` (absent when the compressed output is 0 bytes), and `accepted`.
//!
//! Failure events additionally carry `cascade_path` and `cascade_depth`, so nested compression
//! errors can be tied back to the ancestor branch that triggered them.
//!
//! From those fields you can derive per-scheme savings, rejection counts, and estimator accuracy
//! with a short `jq` query.

pub mod builtins;
pub mod ctx;
pub mod estimate;
pub mod scheme;
pub mod stats;

mod sample;

mod compressor;
mod trace;
pub use compressor::CascadingCompressor;
