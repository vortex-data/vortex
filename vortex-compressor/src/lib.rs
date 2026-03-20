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

pub mod builtins;
pub mod ctx;
pub mod scheme;
pub mod stats;

mod sample;

mod compressor;
pub use compressor::CascadingCompressor;
