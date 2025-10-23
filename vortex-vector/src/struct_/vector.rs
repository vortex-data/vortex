// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVector`].

use vortex_mask::Mask;

use crate::{StructVectorMut, VectorOps};

/// An immutable vector of boolean values.
///
/// `StructVector` can be considered a borrowed / frozen version of [`StructVectorMut`], which is
/// created via the [`freeze`](crate::VectorMutOps::freeze) method.
///
/// See the documentation for [`StructVectorMut`] for more information.
#[derive(Debug, Clone)]
pub struct StructVector {
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: Mask,
}

impl VectorOps for StructVector {
    type Mutable = StructVectorMut;

    fn len(&self) -> usize {
        todo!()
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        todo!()
    }
}
