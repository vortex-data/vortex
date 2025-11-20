// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::view::LayoutView;
use crate::v2::vtable::{ChildName, VTable};
use crate::LayoutId;

/// A layout that combines one child layout per field into an aligned stream of struct arrays.
pub struct StructLayout;

impl VTable for StructLayout {
    type Instance = ();

    fn id(&self) -> LayoutId {
        LayoutId::from("vortex.struct")
    }

    fn child_name(&self, view: &LayoutView<Self>, child_idx: usize) -> ChildName {
        let fields = view.dtype().as_struct_fields();
        ChildName::from(fields.names()[child_idx].inner().clone())
    }
}
