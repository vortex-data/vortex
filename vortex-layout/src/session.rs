// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Registry;

use crate::LayoutEncoding;
use crate::LayoutEncodingRef;
use crate::layouts::chunked::Chunked;
use crate::layouts::dict::Dict;
use crate::layouts::flat::Flat;
use crate::layouts::list::List;
use crate::layouts::struct_::Struct;
use crate::layouts::zoned::LegacyStats;
use crate::layouts::zoned::Zoned;

pub type LayoutRegistry = Registry<LayoutEncodingRef>;

/// Session state for layout encodings.
#[derive(Clone, Debug)]
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
        layouts.register(Chunked.id(), &Chunked as &dyn LayoutEncoding);
        layouts.register(Flat.id(), &Flat as &dyn LayoutEncoding);
        layouts.register(Struct.id(), &Struct as &dyn LayoutEncoding);
        layouts.register(Zoned.id(), &Zoned as &dyn LayoutEncoding);
        layouts.register(LegacyStats.id(), &LegacyStats as &dyn LayoutEncoding);
        layouts.register(Dict.id(), &Dict as &dyn LayoutEncoding);
        layouts.register(List.id(), &List as &dyn LayoutEncoding);

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
    fn layouts(&self) -> SessionGuard<'_, LayoutSession> {
        self.get::<LayoutSession>()
    }
}
impl<S: SessionExt> LayoutSessionExt for S {}
