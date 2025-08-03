// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains some experiments into improving the performance of Vortex. It breaks down
//! into a few sections.
//!
//! ## Vectorized Compute
//!
//! Vectors represent mutable views over canonical output arrays. Evaluating an expression over
//! an array first constructs a pipeline, and then output vectors are pushed into it one at a time.
//!
//! ## Scanning
//!
//! The absence of a selection array is what requires us to separate filter from projection.
//! If we can return sparse arrays from an expression evaluation, then we can combine filter and
//! projection expressions together into a single execution plan, thus removing duplicate
//! decompression kernels.
//!
//! What we currently call a FilterLayoutReader would be responsible for splitting the expression
//! into `rounds`. Each round is a single expression evaluation over the split. Any projection
//! columns from a round are saved off to one side, and filter columns are resolved into a mask
//! which is then passed into the next round. If, for example, a filter expression performs a full
//! decompression of a column, then the projection result should simply be saved off to one side.
//!
//! Further, pruning should happen in an up-front round-trip over the full file.
//!
//! ## Scan I/O
//!
//! Currently, I/O orchestrated using Rust futures. I believe this adds per-segment overhead when
//! scheduling (a single segment gets resolved, and the full future tree wakes up to get polled).
//! Instead, scanning should produce tasks to be scheduled by worker threads. Similarly, I/O should
//! be inlined into these work threads such that we end up with a thread-per-core model.

pub mod array;
pub mod buffers;
pub mod encodings;
pub mod expression;
pub mod mask;
pub mod pipeline;
pub mod vector;
