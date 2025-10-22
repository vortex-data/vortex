// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Immutable and mutable decompressed (canonical) vectors for Vortex.
// TODO(connor): More docs

// TODO(connor):
// - Document everything
// - Figure out correct panic propagation
// - Figure out exact semantics of `split_off` w.r.t. length of capacity
// - Fix bugs in implementations
// - Add tests
// - Figure out error semantics on ops traits
// - Implement PartialEq and Eq for vectors
// - Add stubs for remaining vector variants
// - Potentially add `TryFrom<<Type>Vector> for Vector` or some other conversion method

#![deny(missing_docs)]

mod bool;
mod macros;
mod null;
mod ops;
mod primitive;
mod private;
mod vector;
mod vector_mut;

pub use bool::{BoolVector, BoolVectorMut};
pub use null::{NullVector, NullVectorMut};
pub use ops::{VectorMutOps, VectorOps};
pub use primitive::{PVector, PVectorMut, PrimitiveVector, PrimitiveVectorMut};
pub use vector::Vector;
pub use vector_mut::VectorMut;
