// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of variable-length binary types.
//!
//! All types are specializations of the [`BinaryViewVector`] type, which is represented internally
//! by `BinaryView`s. `BinaryView`s are identical to the `BinaryView` type defined by the Arrow
//! [specification](https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-view-layout),
//! which are inspired by "German" strings.

pub use scalar::*;
pub use types::*;
pub use vector_mut::*;
pub use view::*;
use vortex_error::vortex_panic;

use crate::Vector;

mod scalar;
mod types;
// mod vector;
mod vector_mut;
mod view;

/// Type alias for non-utf8 variable-length binary vectors.
pub type BinaryVector = BinaryViewVector<BinaryType>;
/// Type alias for UTF-8 variable-length string vectors.
pub type StringVector = BinaryViewVector<StringType>;
/// Type alias for non-utf8 variable-length binary scalars.
pub type BinaryScalar = BinaryViewScalar<BinaryType>;

impl BinaryViewDowncast for Vector {
    type Output<T: BinaryViewType> = BinaryViewVector<T>;

    fn into_binary(self) -> Self::Output<BinaryType> {
        if let Vector::Binary(v) = self {
            return v;
        }
        vortex_panic!("Expected BinaryVector, got {self:?}");
    }

    fn into_string(self) -> Self::Output<StringType> {
        if let Vector::String(v) = self {
            return v;
        }
        vortex_panic!("Expected StringVector, got {self:?}");
    }
}

impl BinaryViewTypeUpcast for Vector {
    type Input<T: BinaryViewType> = BinaryViewVector<T>;

    fn from_binary(input: Self::Input<BinaryType>) -> Self {
        Vector::Binary(input)
    }

    fn from_string(input: Self::Input<StringType>) -> Self {
        Vector::String(input)
    }
}
