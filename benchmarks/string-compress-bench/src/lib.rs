// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! String-compression benchmark suite.
//!
//! Compares the original C++ FSST (8- and 12-bit code variants), the pure-Rust
//! [`fsst-rs`](https://crates.io/crates/fsst-rs) port, and
//! [OnPair / OnPair16](https://github.com/gargiulofrancesco/onpair_rs) on
//! synthetic string datasets. The library exposes a single `Backend` trait so
//! the bench harness and the report binary share the same set of compressors.

pub mod backends;
pub mod datasets;
pub mod harness;

pub use backends::BackendConfig;
pub use harness::{BackendKind, BackendResult, MeasureOpts, run_backend};
