// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

    /// Planning phase (invoked once per expression).
    ///
    /// Implementations should inspect and partition the expression, and then recursively call
    /// `plan` on their children while pruning as many unnecessary children as possible.
    ///
    /// The resulting plan is used to construct per-split I/O graphs in the next phase.
    fn plan(
        layout: &Layout<Self>,
        expr: &Expression,
        selection: &RowSelection,
    ) -> VortexResult<Self::Plan>;

    /// Return the I/O graph for a single split of the plan.
    fn split(plan: &Self::Plan) -> VortexResult<()>;
}

/// The positional relationship of a layout to its parent.
pub enum ChildRelationship {
    /// A row offset from the current layout.
    RowOffset(u64),
    /// A child field of the current layout.
    FieldName(FieldName),
    /// Auxiliary data that is positionally unrelated to the layout.
    Auxiliary,
}

/// A set of rows to include in the scan.
pub enum RowSelection {
    All,
    IncludeRanges(Vec<Range<u64>>),
}
