// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::varbin::vector_mut::VarBinVectorMut;
use crate::varbin::view::BinaryView;
use crate::varbin::VarBinType;
use crate::VectorOps;
use std::sync::Arc;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_mask::Mask;

/// A variable-length binary vector.
#[derive(Debug, Clone)]
pub struct VarBinVector<T: VarBinType> {
    views: Buffer<BinaryView>,
    validity: Mask,
    buffers: Arc<Box<[ByteBuffer]>>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: VarBinType> VarBinVector<T> {
    /// Creates a new [`VarBinVector`] from the provided components.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate the consistency of the provided
    /// components.
    ///
    /// The caller must ensure that:
    /// - The length of the `validity` mask matches the length of the `views` buffer.
    /// - The `views` buffer correctly references the data in the `buffers`.
    pub unsafe fn new_unchecked(
        views: Buffer<BinaryView>,
        validity: Mask,
        buffers: Arc<Box<[ByteBuffer]>>,
    ) -> Self {
        Self {
            views,
            validity,
            buffers,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: VarBinType> VectorOps for VarBinVector<T> {
    type Mutable = VarBinVectorMut<T>;

    fn len(&self) -> usize {
        self.views.len()
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
