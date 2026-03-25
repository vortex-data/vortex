// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::segments::SegmentSource;
use crate::v2::layout::LayoutId;
use crate::v2::layout::typed::DynLayout;
use crate::v2::scan::planner::SplitPlannerRef;
use crate::v2::selection::Selection;

#[derive(Clone)]
pub struct LayoutRef(pub(super) Arc<dyn DynLayout>);

impl LayoutRef {
    /// Returns the ID of the layout.
    pub fn id(&self) -> LayoutId {
        self.0.id()
    }

    /// Returns the DType of the layout.
    pub fn dtype(&self) -> &DType {
        self.0.dtype()
    }

    /// Returns the row count of the layout.
    pub fn row_count(&self) -> u64 {
        self.0.row_count()
    }

    /// Returns the nth child of the layout.
    ///
    /// May fail if the deferred deserialization of the layout tree fails.
    ///
    /// # Panics
    ///
    /// Panics on out-of-bounds error.
    pub fn child(&self, idx: usize) -> VortexResult<LayoutRef> {
        self.0.child(idx)
    }

    /// Returns the segment source backing this layout.
    pub fn segment_source(&self) -> &Arc<dyn SegmentSource> {
        self.0.segment_source()
    }

    /// Prepares a split planner for the given expression and row selection.
    ///
    /// `row_offset` is the global row offset of this layout in the main row space.
    /// `None` means the layout is auxiliary and should not register split boundaries.
    pub fn prepare(
        &self,
        expr: &Expression,
        selection: &Selection,
        row_offset: Option<u64>,
        row_splits: &mut BTreeSet<u64>,
        session: &VortexSession,
    ) -> VortexResult<SplitPlannerRef> {
        self.0
            .prepare(expr, selection, row_offset, row_splits, session)
    }
}
