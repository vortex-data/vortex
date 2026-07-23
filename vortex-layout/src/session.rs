// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::LayoutEncoding;
use crate::LayoutEncodingRef;
use crate::layouts::chunked::ChunkedLayoutEncoding;
use crate::layouts::dict::DictLayoutEncoding;
use crate::layouts::flat::FlatLayoutEncoding;
use crate::layouts::list::ListLayoutEncoding;
use crate::layouts::struct_::StructLayoutEncoding;
use crate::layouts::zoned::{LegacyStatsLayoutEncoding, ZonedLayoutEncoding};

/// Session state for layout encodings.
#[derive(Clone, Debug)]
pub struct LayoutSession {
    registry: ArcSwapMap<Id, LayoutEncodingRef>,
}

impl LayoutSession {
    /// Register a layout encoding in the session, replacing any existing encoding with the same ID.
    pub fn register(&self, layout: LayoutEncodingRef) {
        self.registry.insert(layout.id(), layout);
    }

    /// Register layout encodings in the session, replacing any existing encodings with the same IDs.
    pub fn register_many(&self, layouts: impl IntoIterator<Item = LayoutEncodingRef>) {
        for layout in layouts {
            self.registry.insert(layout.id(), layout);
        }
    }

    /// Returns the layout encoding registry.
    pub fn registry(&self) -> &ArcSwapMap<Id, LayoutEncodingRef> {
        &self.registry
    }
}

impl Default for LayoutSession {
    fn default() -> Self {
        let this = Self {
            registry: ArcSwapMap::default(),
        };

        // Register the built-in layout encodings.
        this.register(ChunkedLayoutEncoding.as_ref().into());
        this.register(FlatLayoutEncoding.as_ref().into());
        this.register(StructLayoutEncoding.as_ref().into());
        this.register(ZonedLayoutEncoding.as_ref().into());
        this.register(LegacyStatsLayoutEncoding.as_ref().into());
        this.register(DictLayoutEncoding.as_ref().into());
        this.register(ListLayoutEncoding.as_ref().into());
        this
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
