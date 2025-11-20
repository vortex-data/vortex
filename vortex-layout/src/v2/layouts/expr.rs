// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::view::LayoutView;
use crate::v2::vtable::{ChildName, VTable};
use crate::LayoutId;
use vortex_array::expr::Expression;

pub struct ExprLayout;

impl VTable for ExprLayout {
    type Instance = Expression;

    fn id(&self) -> LayoutId {
        LayoutId::from("vortex.expr")
    }

    fn child_name(&self, _view: &LayoutView<Self>, _child_idx: usize) -> ChildName {
        ChildName::from("scope")
    }
}
