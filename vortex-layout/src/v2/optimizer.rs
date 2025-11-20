// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::layout::LayoutRef;
use crate::v2::view::LayoutView;
use crate::v2::vtable::VTable;
use vortex_error::VortexResult;

/// An optimizer rule that tries to reduce/replace a parent layout where the implementer is a
/// child layout in the `CHILD_IDX` position of the parent layout.
pub trait ReduceParent<Parent: VTable, const CHILD_IDX: usize>: VTable {
    /// Try to reduce/replace the given parent layout based on this child layout.
    ///
    /// If no reduction is possible, return None.
    fn reduce_parent(
        layout: &LayoutView<Self>,
        parent: &LayoutView<Parent>,
    ) -> VortexResult<Option<LayoutRef>>;
}

/// A generic optimizer rule that can be applied to a layout to try to optimize it.
pub trait OptimizerRule {
    /// Try to optimize the given layout, returning a replacement if successful.
    ///
    /// If no optimization is possible, return None.
    fn optimize(&self, layout: &LayoutRef) -> VortexResult<Option<LayoutRef>>;
}
