// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Mutable variable-length binary vector.

use vortex_buffer::{BufferMut, ByteBuffer, ByteBufferMut};
use vortex_mask::MaskMut;

use crate::VectorMutOps;
use crate::varbin::VarBinType;
use crate::varbin::vector::VarBinVector;
use crate::varbin::view::BinaryView;

/// Mutable variable-length binary vector.
#[allow(dead_code)] // FIXME(ngates): remove after implementing the methods
#[derive(Clone, Debug)]
pub struct VarBinVectorMut<T: VarBinType> {
    /// Views into the binary data.
    views: BufferMut<BinaryView>,
    /// Validity mask for the vector.
    validity: MaskMut,

    /// The completed buffers holding referenced binary data.
    buffers: Vec<ByteBuffer>,
    /// The current buffer being appended to, if any.
    open_buffer: Option<ByteBufferMut>,

    /// Marker trait for the [`VarBinType`].
    _marker: std::marker::PhantomData<T>,
}

impl<T: VarBinType> VectorMutOps for VarBinVectorMut<T> {
    type Immutable = VarBinVector<T>;

    fn len(&self) -> usize {
        self.views.len()
    }

    fn capacity(&self) -> usize {
        self.views.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.views.reserve(additional);
    }

    fn extend_from_vector(&mut self, _other: &Self::Immutable) {
        todo!()
    }

    fn append_nulls(&mut self, n: usize) {
        self.views.push_n(BinaryView::empty_view(), n);
        self.validity.append_n(false, n);
    }

    fn freeze(self) -> Self::Immutable {
        todo!()
    }

    fn split_off(&mut self, _at: usize) -> Self {
        todo!()
    }

    fn unsplit(&mut self, _other: Self) {
        todo!()
    }
}
