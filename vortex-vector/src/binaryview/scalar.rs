// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Scalar;
use crate::ScalarOps;
use crate::VectorMut;
use crate::VectorMutOps;
use crate::binaryview::BinaryType;
use crate::binaryview::BinaryViewType;
use crate::binaryview::BinaryViewTypeUpcast;
use crate::binaryview::BinaryViewVectorMut;
use crate::binaryview::StringType;

/// A scalar value for types that implement [`BinaryViewType`].
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryViewScalar<T: BinaryViewType>(Option<T::Scalar>);

impl<T: BinaryViewType> BinaryViewScalar<T> {
    /// Creates a new binary view scalar with the given value.
    pub fn new(value: Option<T::Scalar>) -> Self {
        Self(value)
    }
}

impl<T: BinaryViewType> BinaryViewScalar<T> {
    /// Returns the scalar value as [`BinaryViewType::Scalar`], or [`None`] if the scalar is null.
    pub fn value(&self) -> Option<&T::Scalar> {
        self.0.as_ref()
    }

    /// Creates a zero (empty) binary view scalar.
    pub fn zero() -> Self {
        Self::new(Some(T::empty_scalar()))
    }

    /// Creates a null binary view scalar.
    pub fn null() -> Self {
        Self::new(None)
    }
}

impl<T: BinaryViewType> ScalarOps for BinaryViewScalar<T> {
    fn is_valid(&self) -> bool {
        self.0.is_some()
    }

    fn mask_validity(&mut self, mask: bool) {
        if !mask {
            self.0 = None;
        }
    }

    fn repeat(&self, n: usize) -> VectorMut {
        let mut vec = BinaryViewVectorMut::<T>::with_capacity(n);
        match self.value() {
            None => vec.append_nulls(n),
            Some(buf) => vec.append_owned_values(buf.clone(), n),
        }
        vec.into()
    }
}

impl BinaryViewTypeUpcast for Scalar {
    type Input<T: BinaryViewType> = BinaryViewScalar<T>;

    fn from_binary(input: Self::Input<BinaryType>) -> Self {
        Scalar::Binary(input)
    }

    fn from_string(input: Self::Input<StringType>) -> Self {
        Scalar::String(input)
    }
}

impl<T: BinaryViewType> From<BinaryViewScalar<T>> for Scalar {
    fn from(val: BinaryViewScalar<T>) -> Self {
        T::upcast(val)
    }
}
