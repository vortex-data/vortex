// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod generic;
mod generic_mut;
mod generic_mut_impl;
mod macros;
mod vector;
mod vector_mut;

pub use generic::*;
pub use generic_mut::*;
pub use vector::*;
pub use vector_mut::*;
use vortex_dtype::NativeDecimalType;

use crate::{Vector, VectorMut};

impl From<DecimalVector> for Vector {
    fn from(v: DecimalVector) -> Self {
        Self::Decimal(v)
    }
}

impl<D: NativeDecimalType> From<DVector<D>> for Vector {
    fn from(v: DVector<D>) -> Self {
        Self::Decimal(DecimalVector::from(v))
    }
}

impl<D: NativeDecimalType> From<DVector<D>> for DecimalVector {
    fn from(value: DVector<D>) -> Self {
        D::upcast(value)
    }
}

impl From<DecimalVectorMut> for VectorMut {
    fn from(v: DecimalVectorMut) -> Self {
        Self::Decimal(v)
    }
}

impl<D: NativeDecimalType> From<DVectorMut<D>> for DecimalVectorMut {
    fn from(val: DVectorMut<D>) -> Self {
        D::upcast(val)
    }
}

impl<D: NativeDecimalType> From<DVectorMut<D>> for VectorMut {
    fn from(val: DVectorMut<D>) -> Self {
        Self::Decimal(DecimalVectorMut::from(val))
    }
}
