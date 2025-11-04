// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TODO(connor): Explain what vectors are, why we need them for the new operator model of arrays,
// differences from Arrow (builders and arrays and scalars), etc.
//! Immutable and mutable decompressed (canonical) vectors for Vortex.

#![deny(missing_docs)]
#![deny(clippy::missing_errors_doc)]
#![deny(clippy::missing_panics_doc)]
#![deny(clippy::missing_safety_doc)]

mod binaryview;
mod bool;
mod decimal;
mod fixed_size_list;
mod null;
mod primitive;
mod struct_;

pub use binaryview::*;
pub use bool::*;
pub use decimal::*;
pub use fixed_size_list::*;
pub use null::*;
pub use primitive::*;
pub use struct_::*;

mod ops;
mod vector;
mod vector_mut;

pub use ops::{VectorMutOps, VectorOps};
pub use vector::Vector;
pub use vector_mut::VectorMut;

mod macros;
mod private;
