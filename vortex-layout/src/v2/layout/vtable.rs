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

use crate::v2::layout::Layout;
use crate::v2::layout::LayoutId;
use crate::v2::planner::PlanBuilder;
use crate::v2::planner::SplitPlannerRef;

/// The vtable for a pluggable layout.
pub trait LayoutVTable: 'static + Sized + Clone + Send + Sync {
    /// Any metadata that configures this instance of the layout.
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;
    type Plan: 'static + Send;

    /// Returns the ID of the layout.
    fn id(&self) -> LayoutId;

    /// Returns the DType of the given child.
    fn child_dtype(layout: &Layout<Self>, child_idx: usize) -> &DType;

    /// Returns the relationship of the given child.
    fn child_relationship(layout: &Layout<Self>, child_idx: usize) -> ChildRelationship;

    /// Create a planner for the given expression and row selection.
    ///
    /// Implementations should perform expression partitioning once here, rather than doing it
    /// once per split later.
    ///
    /// Row splits should be registered into the BTreeSet.
    ///
    /// The [`PlanBuilder`] can be used to construct nodes that will be shared across many splits.
    fn prepare(
        layout: &Layout<Self>,
        expr: &Expression,
        selection: &RowSelection,
        row_splits: &mut BTreeSet<u64>,
        builder: &mut PlanBuilder,
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

/// A set of rows to include in the scan.
pub enum RowSelection {
    All,
    IncludeRanges(Vec<Range<u64>>),
}
