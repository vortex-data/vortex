// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This crate contains experiments into vectorized data processing within Vortex.
//!
//! Vectors are fixed-size chunks of data in Vortex represented in a canonical form. The size
//! is a compile-time constant [`N`], which is set to 1024 elements by default.

pub mod array;
pub mod bits;
pub mod buffers;
pub mod encodings;
pub mod expression;
pub mod pipeline;
pub mod selection;
pub mod vector;
pub mod view;

/// The number of elements in each step of a Vortex evaluation pipeline.
pub const N: usize = 1024;
