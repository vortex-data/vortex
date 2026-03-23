// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::dtype::DType;

use crate::v2::layout::LayoutId;
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
}
