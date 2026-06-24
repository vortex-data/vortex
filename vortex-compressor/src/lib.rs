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
//! # Example
//!
//! A [`CascadingCompressor`] can be created directly with a fixed scheme list. With no schemes it
//! still canonicalizes supported inputs and recursively handles nested structure, but no leaf
//! compression is selected.
//!
//! ```rust
//! use vortex_array::{IntoArray, VortexSessionExecute, array_session};
//! use vortex_array::arrays::PrimitiveArray;
//! use vortex_array::validity::Validity;
//! use vortex_buffer::buffer;
//! use vortex_compressor::CascadingCompressor;
//!
//! # fn example() -> vortex_error::VortexResult<()> {
//! let session = array_session();
//! let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
//! let compressor = CascadingCompressor::new(Vec::new());
//!
//! let result = compressor.compress(&array, &mut session.create_execution_ctx())?;
//! assert_eq!(result.dtype(), array.dtype());
//! assert_eq!(result.len(), array.len());
//! # Ok(())
//! # }
//! ```
//!
//! # Observability
//!
//! The compressor emits a small set of `tracing` spans and events on a single target so you can
//! see what it's doing without attaching a profiler.
//!
//! For example, set `RUST_LOG=vortex_compressor::encode=debug` to see compression decision spans
//! and exceptional events. The `vortex_compressor::encode` target carries the top-level `compress`
//! span, per-scheme evaluation and winning-compression spans, the `cascade_exhausted` event,
//! `sample.result` events for zero-byte sample outputs, and both `*.compress_failed` events.
//!
//! The winning-compression span carries `scheme_chosen`, `input_nbytes`, `compressed_nbytes`,
//! `estimated_ratio` (absent when the scheme returned `AlwaysUse` or sampled to 0 bytes),
//! `achieved_ratio` (absent when the compressed output is 0 bytes), and `accepted`.
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
