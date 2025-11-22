// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::Registry;
use vortex_session::{Ref, SessionExt};

use crate::LayoutEncodingRef;
use crate::layouts::chunked::ChunkedLayoutEncoding;
use crate::layouts::dict::DictLayoutEncoding;
use crate::layouts::flat::FlatLayoutEncoding;
use crate::layouts::struct_::StructLayoutEncoding;
use crate::layouts::zoned::ZonedLayoutEncoding;

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
            LayoutEncodingRef::new_ref(ChunkedLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(FlatLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(StructLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(ZonedLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(DictLayoutEncoding.as_ref()),
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
