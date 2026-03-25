// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::ops::Range;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::v2::layout::Layout;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::scan::planner::SplitPlannerRef;
use crate::v2::selection::Selection;

/// The vtable for a pluggable layout.
pub trait LayoutVTable: 'static + Sized + Clone + Send + Sync {
    /// Any metadata that configures this instance of the layout.
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;
    /// Returns the ID of the layout.
    fn id(&self) -> LayoutId;

    /// Deserialize the layout metadata from raw bytes.
    ///
    /// Additional context (dtype, row_count, child row counts) is provided for layouts that
    /// derive metadata from the layout tree structure rather than storing it explicitly.
    fn deserialize_metadata(
        metadata: &[u8],
        dtype: &DType,
        row_count: u64,
        children: &[LayoutChild],
        array_ctx: &ReadContext,
    ) -> VortexResult<Self::Metadata>;

    /// Returns the DType of the given child.
    fn child_dtype(layout: &Layout<Self>, child_idx: usize) -> &DType;

    /// Returns the relationship of the given child.
    fn child_relationship(layout: &Layout<Self>, child_idx: usize) -> ChildRelationship;

    /// Create a planner for the given expression and row selection.
    ///
    /// Implementations should perform expression partitioning once here, rather than doing it
    /// once per split later.
    ///
    /// `row_offset` is the global row offset of this layout. `Some(offset)` means the layout
    /// is in the main row space and should register split boundaries at global coordinates
    /// (`offset + local`). `None` means the layout is auxiliary and should not register splits.
    fn prepare(
        layout: &Layout<Self>,
        expr: &Expression,
        selection: &Selection,
        row_offset: Option<u64>,
        row_splits: &mut BTreeSet<u64>,
        session: &VortexSession,
    ) -> VortexResult<SplitPlannerRef>;
}

/// The positional relationship of a layout to its parent.
pub enum ChildRelationship {
    /// A row offset from the current layout.
    RowOffset(u64),
    /// A child field of the current layout.
    FieldName(FieldName),
    /// Auxiliary data that is positionally unrelated to the parent's row space.
    /// The row range specifies the parent's row range that this auxiliary data covers,
    /// used to determine the lifetime scope for nodes in this subtree.
    Auxiliary(Range<u64>),
}

impl ChildRelationship {
    /// Derives the child's global row offset from the parent's offset and this relationship.
    ///
    /// - `RowOffset(n)`: child is at `parent + n`.
    /// - `FieldName`: same row space, offset unchanged.
    /// - `Auxiliary`: not in the parent's row space, returns `None`.
    pub fn child_row_offset(&self, parent_offset: Option<u64>) -> Option<u64> {
        match self {
            Self::RowOffset(n) => parent_offset.map(|o| o + n),
            Self::FieldName(_) => parent_offset,
            Self::Auxiliary(_) => None,
        }
    }
}
