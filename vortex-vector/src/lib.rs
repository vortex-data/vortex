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

mod vector;
pub use vector::Vector;

mod vector_mut;
pub use vector_mut::VectorMut;

mod ops;
pub use ops::{VectorMutOps, VectorOps};

mod bool;
mod null;
mod primitive;

pub use bool::{BoolVector, BoolVectorMut};
pub use null::{NullVector, NullVectorMut};
pub use primitive::{PVector, PVectorMut, PrimitiveVector, PrimitiveVectorMut};

mod macros;

mod private;
