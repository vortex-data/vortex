// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;

use crate::v2::layout::LayoutId;
use crate::v2::layout::RowSelection;
use crate::v2::layout::SplitIterator;
use crate::v2::layout::typed::DynLayout;

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

    pub fn plan(
        &self,
        expr: &Expression,
        selection: &RowSelection,
        builder: &PlanBuilder,
    ) -> VortexResult<SplitIterator> {
        todo!()
    }
}
