// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::vtable::DynLayoutVTable;
use std::any::Any;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::VortexResult;

pub type LayoutRef = Arc<Layout>;

pub struct Layout {
    vtable: Arc<dyn DynLayoutVTable>,
    instance: Box<dyn Any + Send>,
    row_count: u64,
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
}

impl Layout {
    pub fn vtable(&self) -> &Arc<dyn DynLayoutVTable> {
        &self.vtable
    }

    pub fn instance(&self) -> &dyn Any {
        self.instance.as_ref()
    }

    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn children(&self) -> &Arc<dyn LayoutChildren> {
        &self.children
    }

    pub fn child(&self, idx: usize) -> VortexResult<Layout> {
        // Grab the child dtype from the layout vtable.
        self.children.child(idx, &DType::Null)
    }
}

/// Abstract way of accessing the children of a layout.
///
/// This allows us to abstract over the lazy flatbuffer-based layouts, as well as the in-memory
/// layout trees.
pub trait LayoutChildren: 'static + Send + Sync {
    fn to_arc(&self) -> Arc<dyn LayoutChildren>;

    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<Layout>;

    fn child_row_count(&self, idx: usize) -> u64;

    fn nchildren(&self) -> usize;
}
