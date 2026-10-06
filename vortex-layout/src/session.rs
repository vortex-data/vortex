// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::LayoutEncoding;
use crate::LayoutEncodingRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::Chunked;
use crate::layouts::dict::Dict;
use crate::layouts::flat::Flat;
use crate::layouts::list::List;
use crate::layouts::struct_::Struct;
use crate::layouts::zoned::LegacyStats;
use crate::layouts::zoned::Zoned;

/// Registry of layout encodings.
pub type LayoutRegistry = ArcSwapMap<Id, LayoutEncodingRef>;

/// Builds the writer for a Variant column, given the writer to use for its storage children.
///
/// See [`LayoutSession::register_variant_strategy`].
pub type VariantStrategyFactory =
    Arc<dyn Fn(Arc<dyn LayoutStrategy>) -> Arc<dyn LayoutStrategy> + Send + Sync>;

/// Session state for layout encodings.
#[derive(Clone)]
pub struct LayoutSession {
    registry: LayoutRegistry,
    variant_strategy: Arc<RwLock<Option<VariantStrategyFactory>>>,
}

impl fmt::Debug for LayoutSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayoutSession")
            .field("registry", &self.registry)
            .field(
                "variant_strategy",
                &self.variant_strategy.read().as_ref().map(|_| "<factory>"),
            )
            .finish()
    }
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
    pub fn registry(&self) -> &LayoutRegistry {
        &self.registry
    }

    /// Register the writer that table writers use for Variant columns.
    ///
    /// A Variant encoding that can decompose its storage registers a factory here. The factory
    /// receives the table writer for the storage children (so they are split into columns,
    /// zoned, and compressed like any other field) and returns the writer for the Variant column.
    /// Without a registered factory, Variant columns are written by the leaf strategy.
    pub fn register_variant_strategy(&self, factory: VariantStrategyFactory) {
        *self.variant_strategy.write() = Some(factory);
    }

    /// Returns the registered Variant column writer factory, if any.
    pub fn variant_strategy(&self) -> Option<VariantStrategyFactory> {
        self.variant_strategy.read().clone()
    }
}

impl Default for LayoutSession {
    fn default() -> Self {
        let this = Self {
            registry: LayoutRegistry::default(),
            variant_strategy: Default::default(),
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
