// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::listview::ListViewVector;
use crate::{Scalar, ScalarOps, VectorMut, VectorOps};

/// A scalar value for list view types.
///
/// The inner value is a ListViewVector with length 1.
#[derive(Debug)]
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

    fn repeat(&self, _n: usize) -> VectorMut {
        todo!()
    }
}

impl From<ListViewScalar> for Scalar {
    fn from(val: ListViewScalar) -> Self {
        Scalar::List(val)
    }
}
