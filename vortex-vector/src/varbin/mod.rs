// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use types::*;
pub use vector::*;
pub use vector_mut::*;
use vortex_error::vortex_panic;

use crate::{Vector, VectorMut};

mod types;
mod vector;
mod vector_mut;
mod view;

/// Type alias for non-utf8 variable-length binary vectors.
pub type BinaryVector = VarBinVector<BinaryType>;
/// Type alias for mutable non-utf8 variable-length binary vectors.
pub type BinaryVectorMut = VarBinVectorMut<BinaryType>;
/// Type alias for UTF-8 variable-length string vectors.
pub type StringVector = VarBinVector<StringType>;
/// Type alias for mutable UTF-8 variable-length string vectors.
pub type StringVectorMut = VarBinVectorMut<StringType>;

impl VarBinTypeDowncast for Vector {
    type Output<T: VarBinType> = VarBinVector<T>;

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

impl VarBinTypeUpcast for Vector {
    type Input<T: VarBinType> = VarBinVector<T>;

    fn from_binary(input: Self::Input<BinaryType>) -> Self {
        Vector::Binary(input)
    }

    fn from_string(input: Self::Input<StringType>) -> Self {
        Vector::String(input)
    }
}

impl VarBinTypeDowncast for VectorMut {
    type Output<T: VarBinType> = VarBinVectorMut<T>;

    fn into_binary(self) -> Self::Output<BinaryType> {
        if let VectorMut::Binary(v) = self {
            return v;
        }
        vortex_panic!("Expected BinaryVector, got {self:?}");
    }

    fn into_string(self) -> Self::Output<StringType> {
        if let VectorMut::String(v) = self {
            return v;
        }
        vortex_panic!("Expected StringVector, got {self:?}");
    }
}

impl VarBinTypeUpcast for VectorMut {
    type Input<T: VarBinType> = VarBinVectorMut<T>;

    fn from_binary(input: Self::Input<BinaryType>) -> Self {
        VectorMut::Binary(input)
    }

    fn from_string(input: Self::Input<StringType>) -> Self {
        VectorMut::String(input)
    }
}
