// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::varbin::vector::VarBinVector;
use crate::varbin::view::BinaryView;
use crate::varbin::VarBinType;
use crate::VectorMutOps;
use vortex_buffer::{BufferMut, ByteBuffer};
use vortex_mask::MaskMut;

/// Mutable variable-length binary vector.
#[derive(Clone, Debug)]
pub struct VarBinVectorMut<T: VarBinType> {
    views: BufferMut<BinaryView>,
    validity: MaskMut,

    buffers: Vec<ByteBuffer>,
    open_buffer: Option<ByteBuffer>,

    _marker: std::marker::PhantomData<T>,
}

impl<T: VarBinType> VarBinVectorMut<T> {
    pub(super) fn new(
        views: BufferMut<BinaryView>,
        validity: MaskMut,
        buffers: Vec<ByteBuffer>,
    ) -> Self {
        Self {
            views,
            validity,
            buffers,
            open_buffer: None,
            _marker: std::marker::PhantomData,
        }
    }
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

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        todo!()
    }

    fn append_nulls(&mut self, n: usize) {
        self.views.push_n(BinaryView::empty_view(), n);
        self.validity.append_n(false, n);
    }

    fn freeze(self) -> Self::Immutable {
        todo!()
    }

    fn split_off(&mut self, at: usize) -> Self {
        todo!()
    }

    fn unsplit(&mut self, other: Self) {
        todo!()
    }
}
