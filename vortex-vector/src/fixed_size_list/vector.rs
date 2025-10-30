// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`FixedSizeListVector`].

use crate::{FixedSizeListVectorMut, VectorOps};

/// An immutable vector of fixed-size lists.
///
/// `FixedSizeListVector` can be considered a borrowed / frozen version of
/// [`FixedSizeListVectorMut`], which is created via the [`freeze`](crate::VectorMutOps::freeze)
/// method.
///
/// See the documentation for [`FixedSizeListVectorMut`] for more information.
#[derive(Debug, Clone)]
pub struct FixedSizeListVector;

impl VectorOps for FixedSizeListVector {
    type Mutable = FixedSizeListVectorMut;

    fn len(&self) -> usize {
        todo!()
    }

    fn validity(&self) -> &vortex_mask::Mask {
        todo!()
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        todo!()
    }
}
