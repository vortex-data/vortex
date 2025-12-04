// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Scalar;
use crate::ScalarOps;
use crate::VectorMut;
use crate::VectorOps;
use crate::listview::ListViewVector;
use vortex_mask::Mask;

/// A scalar value for list view types.
///
/// The inner value is a ListViewVector with length 1.
#[derive(Clone, Debug)]
pub struct ListViewScalar(ListViewVector);

impl ListViewScalar {
    /// Create a new ListViewScalar from a length-1 ListViewVector.
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

impl ScalarOps for ListViewScalar {
    fn is_valid(&self) -> bool {
        self.0.validity().value(0)
    }

    fn mask_validity(&mut self, mask: bool) {
        if !mask {
            self.0.mask_validity(&Mask::new_false(1))
        }
    }

    fn repeat(&self, _n: usize) -> VectorMut {
        todo!()
    }
}

impl From<ListViewScalar> for Scalar {
    fn from(val: ListViewScalar) -> Self {
        Scalar::List(val)
    }
}
