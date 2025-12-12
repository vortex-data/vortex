// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_mask::MaskMut;

use crate::Scalar;
use crate::ScalarOps;
use crate::VectorMut;
use crate::VectorMutOps;
use crate::VectorOps;
use crate::listview::ListViewVector;
use crate::listview::ListViewVectorMut;

/// A scalar value for list view types.
///
/// The inner value is a [`ListViewVector`] with length 1.
#[derive(Clone, Debug, PartialEq)]
pub struct ListViewScalar(ListViewVector);

impl ListViewScalar {
    /// Create a new [`ListViewScalar`] from a length-1 [`ListViewVector`].
    ///
    /// # Panics
    ///
    /// Panics if the input vector does not have length 1.
    pub fn new(vector: ListViewVector) -> Self {
        assert_eq!(vector.len(), 1);
        Self(vector)
    }

    /// Returns the inner length-1 vector representing the list view scalar.
    pub fn value(&self) -> &ListViewVector {
        &self.0
    }
}

impl ListViewScalar {
    /// Creates a zero (empty list) list view scalar of the given [`DType`].
    ///
    /// # Panics
    ///
    /// Panics if the dtype is not a [`DType::List`].
    pub fn zero(dtype: &DType) -> Self {
        if !matches!(dtype, DType::List(..)) {
            vortex_panic!("Expected List dtype, got {}", dtype);
        }

        let mut vec = VectorMut::with_capacity(dtype, 1);
        vec.append_zeros(1);
        vec.freeze().scalar_at(0).into_list()
    }

    /// Creates a null list view scalar of the given [`DType`].
    ///
    /// # Panics
    ///
    /// Panics if the dtype is not a nullable [`DType::List`].
    pub fn null(dtype: &DType) -> Self {
        match dtype {
            DType::List(_, n) if n.is_nullable() => {}
            DType::List(..) => vortex_panic!("Expected nullable List dtype, got {}", dtype),
            _ => vortex_panic!("Expected List dtype, got {}", dtype),
        }

        let mut vec = VectorMut::with_capacity(dtype, 1);
        vec.append_nulls(1);
        vec.freeze().scalar_at(0).into_list()
    }
}

impl ScalarOps for ListViewScalar {
    fn is_valid(&self) -> bool {
        self.0.validity().value(0)
    }

    fn mask_validity(&mut self, mask: bool) {
        if !mask {
            self.0.mask_validity(&Mask::new_false(1))
        }
    }

    fn repeat(&self, n: usize) -> VectorMut {
        // Grab the scalar elements.
        let elements = self.0.elements.clone();
        // Repeat the offset and size n times.
        let offsets = self.0.offsets.scalar_at(0).repeat(n).into_primitive();
        let sizes = self.0.sizes.scalar_at(0).repeat(n).into_primitive();
        unsafe {
            ListViewVectorMut::new_unchecked(
                Box::new(Arc::unwrap_or_clone(elements).into_mut()),
                offsets,
                sizes,
                MaskMut::new(n, self.is_valid()),
            )
        }
        .into()
    }
}

impl From<ListViewScalar> for Scalar {
    fn from(val: ListViewScalar) -> Self {
        Scalar::List(val)
    }
}
