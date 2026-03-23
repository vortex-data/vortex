// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;

use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutRef;
use crate::v2::layout::vtable::LayoutVTable;

pub struct Layout<V: LayoutVTable> {
    vtable: V,
    metadata: V::Metadata,
    dtype: DType,
    row_count: u64,
    children: Vec<LayoutChild>,
}

impl<V: LayoutVTable> Layout<V> {
    /// Returns the ID of the layout.
    fn id(&self) -> LayoutId {
        DynLayout::id(self)
    }

    /// Returns the dtype of the layout.
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the nth child of the layout.
    fn child(&self, idx: usize) -> VortexResult<LayoutRef> {
        DynLayout::child(self, idx)
    }
}

pub(super) trait DynLayout: 'static + Send + Sync + super::sealed::Sealed {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> LayoutId;
    fn metadata_any(&self) -> &dyn Any;

    fn dtype(&self) -> &DType;
    fn child(&self, idx: usize) -> VortexResult<LayoutRef>;
}

impl<V: LayoutVTable> DynLayout for Layout<V> {
    #[inline(always)]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline(always)]
    fn id(&self) -> LayoutId {
        self.vtable.id()
    }

    #[inline(always)]
    fn metadata_any(&self) -> &dyn Any {
        &self.metadata
    }

    #[inline(always)]
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn child(&self, idx: usize) -> VortexResult<LayoutRef> {
        assert!(idx < self.children.len(), "Child idx out of bounds");
        self.children[idx].resolve(self.vtable.child_dtype(idx))
    }
}
