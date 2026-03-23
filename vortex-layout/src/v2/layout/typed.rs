// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_array::dtype::DType;

use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
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

    fn dtype(&self) -> &DType {
        &self.dtype
    }
}
