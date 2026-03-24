// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutRef;
use crate::v2::layout::Selection;
use crate::v2::layout::vtable::LayoutVTable;
use crate::v2::scan::planner::PlanBuilder;
use crate::v2::scan::planner::SplitPlannerRef;

pub struct Layout<V: LayoutVTable> {
    vtable: V,
    metadata: V::Metadata,
    dtype: DType,
    row_count: u64,
    children: Vec<LayoutChild>,
    segments: Vec<SegmentId>,
    segment_source: Arc<dyn SegmentSource>,
}

#[allow(clippy::same_name_method)]
impl<V: LayoutVTable> Layout<V> {
    /// Returns the ID of the layout.
    pub fn id(&self) -> LayoutId {
        self.vtable.id()
    }

    /// Returns the dtype of the layout.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the row count of the layout.
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    /// Returns the segment IDs referenced by this layout.
    pub fn segments(&self) -> &[SegmentId] {
        &self.segments
    }

    /// Returns the segment source backing this layout.
    pub fn segment_source(&self) -> &Arc<dyn SegmentSource> {
        &self.segment_source
    }

    /// Returns the nth child of the layout.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of bounds.
    pub fn child(&self, idx: usize) -> VortexResult<LayoutRef> {
        assert!(idx < self.children.len(), "Child idx out of bounds");
        self.children[idx].resolve(V::child_dtype(self, idx))
    }

    /// Returns the metadata for this layout.
    pub fn metadata(&self) -> &V::Metadata {
        &self.metadata
    }

    /// Returns the number of children.
    pub fn num_children(&self) -> usize {
        self.children.len()
    }
}

pub(super) trait DynLayout: 'static + Send + Sync + super::sealed::Sealed {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> LayoutId;
    fn metadata_any(&self) -> &dyn Any;

    fn dtype(&self) -> &DType;
    fn row_count(&self) -> u64;
    fn segments(&self) -> &[SegmentId];
    fn segment_source(&self) -> &Arc<dyn SegmentSource>;
    fn child(&self, idx: usize) -> VortexResult<LayoutRef>;

    fn prepare(
        &self,
        expr: &Expression,
        selection: &Selection,
        row_splits: &mut BTreeSet<u64>,
    ) -> VortexResult<SplitPlannerRef>;
}

/// Blanket impl: thin forwarder to `Layout<V>` inherent methods.
///
/// Every method here delegates to the corresponding inherent method on `Layout<V>`.
/// Rust's method resolution picks inherent methods over trait methods, so `self.id()` etc.
/// call the inherent impl, not this trait impl (no infinite recursion).
impl<V: LayoutVTable> DynLayout for Layout<V> {
    #[inline(always)]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline(always)]
    fn id(&self) -> LayoutId {
        self.id()
    }

    #[inline(always)]
    fn metadata_any(&self) -> &dyn Any {
        &self.metadata
    }

    #[inline(always)]
    fn dtype(&self) -> &DType {
        self.dtype()
    }

    #[inline(always)]
    fn segments(&self) -> &[SegmentId] {
        self.segments()
    }

    #[inline(always)]
    fn row_count(&self) -> u64 {
        self.row_count()
    }

    #[inline(always)]
    fn segment_source(&self) -> &Arc<dyn SegmentSource> {
        self.segment_source()
    }

    #[inline(always)]
    fn child(&self, idx: usize) -> VortexResult<LayoutRef> {
        self.child(idx)
    }

    fn prepare(
        &self,
        expr: &Expression,
        selection: &Selection,
        row_splits: &mut BTreeSet<u64>,
    ) -> VortexResult<SplitPlannerRef> {
        V::prepare(self, expr, selection, row_splits)
    }
}
