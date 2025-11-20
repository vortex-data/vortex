// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::layout::{Layout, LayoutRef};
use crate::v2::vtable::VTable;
use std::ops::Deref;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_session::SessionVar;

pub struct LayoutView<'a, V: VTable> {
    layout: &'a LayoutRef,
    vtable: &'a V,
    instance: &'a V::Instance,
}

impl<V: VTable> Deref for LayoutView<'_, V> {
    type Target = LayoutRef;

    fn deref(&self) -> &Self::Target {
        self.layout
    }
}

impl<'a, V: VTable> LayoutView<'a, V> {
    pub fn new(layout: &'a Layout) -> Self {
        Self::try_new(layout).vortex_expect("Layout type mismatch")
    }

    pub fn try_new(layout: &'a Layout) -> VortexResult<Self> {
        let vtable = layout
            .vtable()
            .as_any()
            .downcast_ref::<V>()
            .ok_or_else(|| vortex_err!("Layout vtable type mismatch"))?;
        let instance = layout
            .instance()
            .as_any()
            .downcast_ref::<V::Instance>()
            .ok_or_else(|| vortex_err!("Layout vtable type mismatch"))?;
        Ok(Self {
            layout,
            vtable,
            instance,
        })
    }

    pub fn vtable(&self) -> &'a V {
        self.vtable
    }

    pub fn instance(&self) -> &'a V::Instance {
        self.instance
    }
}
