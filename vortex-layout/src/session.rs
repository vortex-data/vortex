// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::registry::Registry;

use crate::LayoutEncodingRef;
use crate::layouts::array_tree::ArrayTreeFlatLayoutEncoding;
use crate::layouts::array_tree::ArrayTreeLayoutEncoding;
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
        layouts.register(StructLayoutEncoding.id(), StructLayoutEncoding.as_ref());
        layouts.register(ZonedLayoutEncoding.id(), ZonedLayoutEncoding.as_ref());
        layouts.register(DictLayoutEncoding.id(), DictLayoutEncoding.as_ref());
        layouts.register(
            ArrayTreeLayoutEncoding.id(),
            ArrayTreeLayoutEncoding.as_ref(),
        );
        layouts.register(
            ArrayTreeFlatLayoutEncoding.id(),
            ArrayTreeFlatLayoutEncoding.as_ref(),
        );

        Self { registry: layouts }
    }
}

impl SessionVar for LayoutSession {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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
