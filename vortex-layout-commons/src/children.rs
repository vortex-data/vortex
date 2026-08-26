// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::LayoutRef;

/// Abstract way of accessing the children of a layout.
///
/// This allows layout trees to use lazy serialized children as well as in-memory children.
pub trait LayoutChildren: 'static + Send + Sync {
    fn to_arc(&self) -> Arc<dyn LayoutChildren>;

    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef>;

    fn child_row_count(&self, idx: usize) -> u64;

    fn nchildren(&self) -> usize;

    /// Returns `true` if the child at `idx` is known, without materializing it, to be indivisible.
    ///
    /// Implementations must conservatively return `false` when answering would require
    /// materializing the child.
    fn child_is_indivisible(&self, _idx: usize) -> bool {
        false
    }
}

impl Debug for dyn LayoutChildren {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayoutChildren")
            .field("nchildren", &self.nchildren())
            .finish()
    }
}

impl LayoutChildren for Arc<dyn LayoutChildren> {
    fn to_arc(&self) -> Arc<dyn LayoutChildren> {
        Arc::clone(self)
    }

    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef> {
        self.as_ref().child(idx, dtype)
    }

    fn child_row_count(&self, idx: usize) -> u64 {
        self.as_ref().child_row_count(idx)
    }

    fn nchildren(&self) -> usize {
        self.as_ref().nchildren()
    }

    fn child_is_indivisible(&self, idx: usize) -> bool {
        self.as_ref().child_is_indivisible(idx)
    }
}

/// In-memory owned layout children.
#[derive(Clone)]
pub struct OwnedLayoutChildren(Vec<LayoutRef>);

impl OwnedLayoutChildren {
    pub fn layout_children(children: Vec<LayoutRef>) -> Arc<dyn LayoutChildren> {
        Arc::new(Self(children))
    }
}

/// Create an in-memory child adapter from owned layout references.
pub fn layout_children(children: Vec<LayoutRef>) -> Arc<dyn LayoutChildren> {
    OwnedLayoutChildren::layout_children(children)
}

impl LayoutChildren for OwnedLayoutChildren {
    fn to_arc(&self) -> Arc<dyn LayoutChildren> {
        Arc::new(self.clone())
    }

    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef> {
        if idx >= self.0.len() {
            vortex_bail!("Child index out of bounds: {} of {}", idx, self.0.len());
        }
        let child = &self.0[idx];
        if child.dtype() != dtype {
            vortex_bail!("Child dtype mismatch: {} != {}", child.dtype(), dtype);
        }
        Ok(Arc::clone(child))
    }

    fn child_row_count(&self, idx: usize) -> u64 {
        self.0[idx].row_count()
    }

    fn nchildren(&self) -> usize {
        self.0.len()
    }

    fn child_is_indivisible(&self, idx: usize) -> bool {
        self.0[idx].dyn_is_indivisible()
    }
}
