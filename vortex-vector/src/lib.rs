// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Mutable decompressed (canonical) vectors for Vortex.
//!
//! TODO docs.

#![deny(missing_docs)]
// #![warn(clippy::missing_docs_in_private_items)]
// #![warn(clippy::missing_errors_doc)]
// #![warn(clippy::missing_panics_doc)]
// #![warn(clippy::missing_safety_doc)]

mod vector;
pub use vector::ops::{VectorMutOps, VectorOps};
pub use vector::{Vector, VectorMut};

mod bool;
mod null;
mod primitive;

pub use bool::{BoolVector, BoolVectorMut};
pub use null::{NullVector, NullVectorMut};
pub use primitive::{GenericPVector, GenericPVectorMut, PrimitiveVector, PrimitiveVectorMut};

mod private;
