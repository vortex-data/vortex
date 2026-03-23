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
        V::id(&self.vtable)
    }

    #[inline(always)]
    fn metadata_any(&self) -> &dyn Any {
        &self.metadata
    }

    /// Returns the un-projected DType of the layout.
    #[inline(always)]
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the nth child of the layout.
    fn child(&self, idx: usize) -> VortexResult<LayoutRef> {
        assert!(idx < self.children.len(), "Child idx out of bounds");
        self.children[idx].resolve(self.vtable.child_dtype(idx))
    }
}
