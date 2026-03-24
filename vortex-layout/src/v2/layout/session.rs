// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::v2::layout::LayoutPluginRef;
use crate::v2::layout::LayoutVTable;

pub type LayoutRegistry = Registry<LayoutPluginRef>;

/// Session state for layout encodings.
#[derive(Debug)]
pub struct LayoutSession {
    registry: LayoutRegistry,
}

impl LayoutSession {
    /// Register a layout vtable in the session, replacing any existing vtable with the same ID.
    pub fn register<V: LayoutVTable>(&self, vtable: V) {
        self.registry
            .register(vtable.id(), Arc::new(vtable) as LayoutPluginRef);
    }

    /// Returns the layout encoding registry.
    pub fn registry(&self) -> &LayoutRegistry {
        &self.registry
    }
}

impl Default for LayoutSession {
    fn default() -> Self {
        use crate::v2::layouts::chunked::Chunked;
        use crate::v2::layouts::dict::Dict;
        use crate::v2::layouts::flat::Flat;
        use crate::v2::layouts::struct_::Struct;
        use crate::v2::layouts::zoned::Zoned;

        let session = Self {
            registry: LayoutRegistry::default(),
        };

        session.register(Chunked);
        session.register(Dict);
        session.register(Flat);
        session.register(Struct);
        session.register(Zoned);

        session
    }
}

/// Extension trait for accessing layout session data.
pub trait LayoutSessionExt: SessionExt {
    /// Returns the layout encoding registry.
    fn layouts2(&self) -> Ref<'_, LayoutSession> {
        self.get::<LayoutSession>()
    }
}
impl<S: SessionExt> LayoutSessionExt for S {}
