// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Variable-length binary vector implementation.

use std::sync::Arc;

use vortex_buffer::{Buffer, ByteBuffer};
use vortex_mask::Mask;

use crate::VectorOps;
use crate::varbin::VarBinType;
use crate::varbin::vector_mut::VarBinVectorMut;
use crate::varbin::view::BinaryView;

/// A variable-length binary vector.
#[derive(Debug, Clone)]
pub struct VarBinVector<T: VarBinType> {
    /// Views into the binary data.
    views: Buffer<BinaryView>,
    /// Buffers holding the referenced binary data.
    buffers: Arc<Box<[ByteBuffer]>>,
    /// Validity mask for the vector.
    validity: Mask,
    /// Marker trait for the [`VarBinType`].
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
        buffers: Arc<Box<[ByteBuffer]>>,
        validity: Mask,
    ) -> Self {
        Self {
            views,
            buffers,
            validity,
            _marker: std::marker::PhantomData,
        }
    }

    /// Decomposes the vector into its constituent parts.
    pub fn into_parts(self) -> (Buffer<BinaryView>, Arc<Box<[ByteBuffer]>>, Mask) {
        (self.views, self.buffers, self.validity)
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
