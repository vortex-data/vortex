// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Byte-oriented boolean encoding.
//!
//! Stores boolean values using Rust's `bool` type (one byte per value) rather than packed bits.
//! More memory-intensive than bit-packed `BoolArray`, but useful when data is already byte-aligned
//! or when byte-oriented operations are more efficient.
//!
//! Can be canonicalized to bit-packed `BoolArray` when needed.

pub use array::*;

mod array;
mod compute;
mod rules;
mod slice;
