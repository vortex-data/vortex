// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TODO(connor): Explain what vectors are, why we need them for the new operator model of arrays,
// differences from Arrow (builders and arrays and scalars), etc.
//! Immutable and mutable decompressed (canonical) vectors for Vortex.

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(clippy::missing_errors_doc)]
#![deny(clippy::missing_panics_doc)]
#![deny(clippy::missing_safety_doc)]

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
