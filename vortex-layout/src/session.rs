// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::Registry;
use vortex_session::{Ref, SessionExt};

use crate::layouts::chunked::ChunkedLayoutVTable;
use crate::layouts::dict::DictLayoutVTable;
use crate::layouts::flat::FlatLayoutVTable;
use crate::layouts::struct_::StructLayoutVTable;
use crate::layouts::zoned::ZonedLayoutVTable;
use crate::LayoutEncodingRef;

pub type LayoutRegistry = Registry<LayoutEncodingRef>;

/// Session state for layout encodings.
#[derive(Debug)]
pub struct LayoutSession {
    registry: LayoutRegistry,
}

impl LayoutSession {
    /// Register a layout encoding in the session, replacing any existing encoding with the same ID.
    pub fn register(&self, layout: LayoutEncodingRef) {
        self.registry.register(layout);
    }

    /// Register layout encodings in the session, replacing any existing encodings with the same IDs.
    pub fn register_many(&self, layouts: impl IntoIterator<Item = LayoutEncodingRef>) {
        self.registry.register_many(layouts);
    }

    /// Returns the layout encoding registry.
    pub fn registry(&self) -> &LayoutRegistry {
        &self.registry
    }
}

impl Default for LayoutSession {
    fn default() -> Self {
        let layouts = LayoutRegistry::default();

        // Register the built-in layout encodings.
        layouts.register_many([
            LayoutEncodingRef::new_ref(ChunkedLayoutVTable.as_ref()),
            LayoutEncodingRef::new_ref(FlatLayoutVTable.as_ref()),
            LayoutEncodingRef::new_ref(StructLayoutVTable.as_ref()),
            LayoutEncodingRef::new_ref(ZonedLayoutVTable.as_ref()),
            LayoutEncodingRef::new_ref(DictLayoutVTable.as_ref()),
        ]);

        Self { registry: layouts }
    }
}

/// Extension trait for accessing layout session data.
pub trait LayoutSessionExt: SessionExt {
    /// Returns the layout encoding registry.
    fn layouts(&self) -> Ref<'_, LayoutSession> {
        self.get::<LayoutSession>()
    }
}
impl<S: SessionExt> LayoutSessionExt for S {}
