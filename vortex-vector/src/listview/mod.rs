// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`ListViewVector`] and [`ListViewVectorMut`].
//!
//! A [`ListViewVector`] represents a collection of variable-width lists, where each list can
//! contain a different number of elements.
//!
//! The structure uses separate offset and size vectors to track the boundaries of each list
//! within the flat elements array. This allows for efficient access to individual lists without
//! copying data.  This is similar to Apache Arrow's `ListView` type.

// TODO(connor): More docs and examples.

mod vector;
pub use vector::ListViewVector;

mod vector_mut;
pub use vector_mut::ListViewVectorMut;

mod scalar;
pub use scalar::ListViewScalar;

use crate::Vector;
use crate::VectorMut;

impl From<ListViewVector> for Vector {
    fn from(v: ListViewVector) -> Self {
        Self::List(v)
    }
}

impl From<ListViewVectorMut> for VectorMut {
    fn from(v: ListViewVectorMut) -> Self {
        Self::List(v)
    }
}

#[cfg(test)]
mod tests;
