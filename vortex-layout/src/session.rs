// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Registry;

use crate::LayoutEncodingRef;
use crate::layout_v2;
use crate::layout_v2::VTable as _;
use crate::layouts::chunked::ChunkedLayoutEncoding;
use crate::layouts::dict::DictLayoutEncoding;
use crate::layouts::flat::FlatLayoutEncoding;
use crate::layouts::struct_::StructLayoutEncoding;
use crate::layouts::zoned::LegacyStatsLayoutEncoding;
use crate::layouts::zoned::ZonedLayoutEncoding;
use crate::layouts_v2::chunked::Chunked as ChunkedV2;
use crate::layouts_v2::dict::Dict as DictV2;
use crate::layouts_v2::flat::Flat as FlatV2;
use crate::layouts_v2::struct_::Struct as StructV2;
use crate::layouts_v2::zoned::LegacyStats as LegacyStatsV2;
use crate::layouts_v2::zoned::Zoned as ZonedV2;

pub type LayoutRegistry = Registry<LayoutEncodingRef>;

/// Session state for layout encodings.
#[derive(Clone, Debug)]
pub struct LayoutSession {
    registry: LayoutRegistry,
    v2_registry: layout_v2::LayoutVTableRegistry,
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

    /// Register a v2 layout vtable in the session, replacing any existing vtable with the same ID.
    pub fn register_v2<V: layout_v2::VTable>(&self, vtable: V) {
        self.v2_registry
            .register(vtable.id(), Arc::new(vtable) as layout_v2::LayoutVTableRef);
    }

    /// Returns the v2 layout vtable registry.
    pub fn v2_registry(&self) -> &layout_v2::LayoutVTableRegistry {
        &self.v2_registry
    }
}

impl Default for LayoutSession {
    fn default() -> Self {
        let layouts = LayoutRegistry::default();
        let v2_layouts = layout_v2::LayoutVTableRegistry::default();

        // Register the built-in layout encodings.
        layouts.register(ChunkedLayoutEncoding.id(), ChunkedLayoutEncoding.as_ref());
        layouts.register(FlatLayoutEncoding.id(), FlatLayoutEncoding.as_ref());
        layouts.register(StructLayoutEncoding.id(), StructLayoutEncoding.as_ref());
        layouts.register(ZonedLayoutEncoding.id(), ZonedLayoutEncoding.as_ref());
        layouts.register(
            LegacyStatsLayoutEncoding.id(),
            LegacyStatsLayoutEncoding.as_ref(),
        );
        layouts.register(DictLayoutEncoding.id(), DictLayoutEncoding.as_ref());

        // Register the built-in v2 layout vtables.
        v2_layouts.register(
            ChunkedV2.id(),
            Arc::new(ChunkedV2) as layout_v2::LayoutVTableRef,
        );
        v2_layouts.register(FlatV2.id(), Arc::new(FlatV2) as layout_v2::LayoutVTableRef);
        v2_layouts.register(
            StructV2.id(),
            Arc::new(StructV2) as layout_v2::LayoutVTableRef,
        );
        v2_layouts.register(
            ZonedV2.id(),
            Arc::new(ZonedV2) as layout_v2::LayoutVTableRef,
        );
        v2_layouts.register(
            LegacyStatsV2.id(),
            Arc::new(LegacyStatsV2) as layout_v2::LayoutVTableRef,
        );
        v2_layouts.register(DictV2.id(), Arc::new(DictV2) as layout_v2::LayoutVTableRef);

        Self {
            registry: layouts,
            v2_registry: v2_layouts,
        }
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
