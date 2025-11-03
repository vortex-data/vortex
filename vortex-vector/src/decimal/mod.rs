// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definitions and implementations of decimal vector types.
//!
//! The types that hold data are [`DVector`] and [`DVectorMut`], which are generic over types `D`
//! that implement [`NativeDecimalType`].
//!
//! [`DecimalVector`] and [`DecimalVectorMut`] are enums that wrap all of the different possible
//! [`DVector`]s. There are several macros defined in this crate to make working with these
//! primitive vector types easier.

mod generic;
pub use generic::DVector;

mod generic_mut;
pub use generic_mut::DVectorMut;

mod vector;
pub use vector::DecimalVector;

mod vector_mut;
pub use vector_mut::DecimalVectorMut;

mod macros;

use vortex_dtype::NativeDecimalType;

use crate::{Vector, VectorMut};

impl From<DecimalVector> for Vector {
    fn from(v: DecimalVector) -> Self {
        Self::Decimal(v)
    }
}

impl<D: NativeDecimalType> From<DVector<D>> for DecimalVector {
    fn from(value: DVector<D>) -> Self {
        D::upcast(value)
    }
}

impl<D: NativeDecimalType> From<DVector<D>> for Vector {
    fn from(v: DVector<D>) -> Self {
        Self::Decimal(DecimalVector::from(v))
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
