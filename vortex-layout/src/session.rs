// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::LayoutEncodingRef;
use crate::layouts::chunked::ChunkedLayoutEncoding;
use crate::layouts::dict::DictLayoutEncoding;
use crate::layouts::flat::FlatLayoutEncoding;
use crate::layouts::list::ListLayoutEncoding;
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
        self.registry.register(layout.id(), layout);
    }

    /// Register layout encodings in the session, replacing any existing encodings with the same IDs.
    pub fn register_many(&self, layouts: impl IntoIterator<Item = LayoutEncodingRef>) {
        for layout in layouts {
            self.registry.register(layout.id(), layout);
        }
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
        layouts.register(ChunkedLayoutEncoding.id(), ChunkedLayoutEncoding.as_ref());
        layouts.register(FlatLayoutEncoding.id(), FlatLayoutEncoding.as_ref());
        layouts.register(ListLayoutEncoding.id(), ListLayoutEncoding.as_ref());
        layouts.register(StructLayoutEncoding.id(), StructLayoutEncoding.as_ref());
        layouts.register(ZonedLayoutEncoding.id(), ZonedLayoutEncoding.as_ref());
        layouts.register(DictLayoutEncoding.id(), DictLayoutEncoding.as_ref());

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
