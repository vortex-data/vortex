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
use crate::layouts::chunked::Chunked;
use crate::layouts::dict::Dict;
use crate::layouts::flat::Flat;
use crate::layouts::list::List;
use crate::layouts::struct_::Struct;
use crate::layouts::zoned::{LegacyStats, Zoned};

/// Session state for layout encodings.
#[derive(Clone, Debug)]
pub struct LayoutSession {
    registry: ArcSwapMap<Id, LayoutEncodingRef>,
}

impl LayoutSession {
    /// Register a layout encoding in the session, replacing any existing encoding with the same ID.
    pub fn register(&self, layout_ref: impl Into<LayoutEncodingRef>) {
        let layout = layout_ref.into();
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
        this.register(&Chunked as &dyn LayoutEncoding);
        this.register(&Flat as &dyn LayoutEncoding);
        this.register(&Struct as &dyn LayoutEncoding);
        this.register(&Zoned as &dyn LayoutEncoding);
        this.register(&LegacyStats as &dyn LayoutEncoding);
        this.register(&Dict as &dyn LayoutEncoding);
        this.register(&List as &dyn LayoutEncoding);
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
