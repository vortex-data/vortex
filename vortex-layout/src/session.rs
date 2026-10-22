// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::DynLayout;
use crate::LayoutEncoding;
use crate::LayoutEncodingRef;
use crate::LayoutSerialization;
use crate::layouts::chunked::Chunked;
use crate::layouts::dict::Dict;
use crate::layouts::flat::Flat;
use crate::layouts::indexed::Indexed;
use crate::layouts::list::List;
use crate::layouts::struct_::Struct;
use crate::layouts::zoned::LegacyStats;
use crate::layouts::zoned::Zoned;

/// Registry of layout encodings.
pub type LayoutRegistry = ArcSwapMap<Id, LayoutEncodingRef>;

/// Session state for layout encodings.
#[derive(Clone, Debug)]
pub struct LayoutSession {
    /// Deserializers keyed by the layout ID found on the wire.
    registry: LayoutRegistry,
    /// Serializers keyed by the in-memory layout ID.
    serializers: LayoutRegistry,
}

impl LayoutSession {
    /// Register a layout encoding and all of its serialized IDs in the session.
    ///
    /// This replaces any serializer with the same in-memory ID and any deserializer registered
    /// under one of [`LayoutEncoding::serialized_ids`].
    pub fn register(&self, layout_ref: impl Into<LayoutEncodingRef>) {
        let layout = layout_ref.into();
        self.serializers.insert(layout.id(), layout.clone());
        for serialized_id in layout.serialized_ids() {
            self.registry.insert(serialized_id, layout.clone());
        }
    }

    /// Register layout encodings in the session, replacing any existing encodings with the same IDs.
    pub fn register_many(&self, layouts: impl IntoIterator<Item = LayoutEncodingRef>) {
        for layout in layouts {
            self.register(layout);
        }
    }

    /// Returns the layout encoding registry, keyed by the layout ID found on the wire.
    pub fn registry(&self) -> &LayoutRegistry {
        &self.registry
    }

    /// Serialize `layout` with the encoding registered for its in-memory ID.
    ///
    /// A layout whose ID has no registered encoding, such as a foreign layout, is written under its
    /// own ID.
    pub fn serialize(&self, layout: &dyn DynLayout) -> VortexResult<LayoutSerialization> {
        let Some(encoding) = self.serializers.get(&layout.encoding_id()) else {
            return Ok(LayoutSerialization::from_layout(layout));
        };
        let serialization = encoding.serialize(layout)?;
        vortex_ensure!(
            encoding
                .serialized_ids()
                .contains(&serialization.serialized_id),
            "layout serializer {} produced undeclared serialized ID {}",
            layout.encoding_id(),
            serialization.serialized_id,
        );
        Ok(serialization)
    }
}

impl Default for LayoutSession {
    fn default() -> Self {
        let this = Self {
            registry: LayoutRegistry::default(),
            serializers: LayoutRegistry::default(),
        };

        // Register the built-in layout encodings.
        this.register(&Chunked as &dyn LayoutEncoding);
        this.register(&Flat as &dyn LayoutEncoding);
        this.register(&Struct as &dyn LayoutEncoding);
        this.register(&Zoned as &dyn LayoutEncoding);
        this.register(&LegacyStats as &dyn LayoutEncoding);
        this.register(&Dict as &dyn LayoutEncoding);
        this.register(&List as &dyn LayoutEncoding);
        this.register(&Indexed as &dyn LayoutEncoding);
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
