// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped registry of [`ArrowExportPlugin`]s.

use std::sync::Arc;

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::arrow::export_plugin::ArrowExportPlugin;
use crate::arrow::export_plugin::ArrowExportPluginRef;
use crate::dtype::extension::ExtId;
use crate::extension::datetime::DateArrowExport;
use crate::extension::datetime::TimeArrowExport;
use crate::extension::datetime::TimestampArrowExport;

/// Registry of Arrow export plugins keyed by extension id.
pub type ArrowExportRegistry = Registry<ArrowExportPluginRef>;

/// Session for managing Arrow export plugins.
#[derive(Debug)]
pub struct ArrowExportSession {
    registry: ArrowExportRegistry,
}

impl Default for ArrowExportSession {
    fn default() -> Self {
        let this = Self {
            registry: Registry::default(),
        };
        this.register(DateArrowExport);
        this.register(TimeArrowExport);
        this.register(TimestampArrowExport);
        this
    }
}

impl ArrowExportSession {
    /// Register an Arrow export plugin, replacing any existing plugin with the same [`ExtId`].
    pub fn register(&self, plugin: impl ArrowExportPlugin) {
        let id = plugin.id();
        self.registry
            .register(id, Arc::new(plugin) as ArrowExportPluginRef);
    }

    /// Find the plugin registered for `id`, if any.
    pub fn find(&self, id: &ExtId) -> Option<ArrowExportPluginRef> {
        self.registry.find(id)
    }

    /// Return the underlying registry.
    pub fn registry(&self) -> &ArrowExportRegistry {
        &self.registry
    }
}

/// Extension trait for accessing the [`ArrowExportSession`] from a Vortex session.
pub trait ArrowExportSessionExt: SessionExt {
    /// Get the Arrow export session.
    fn arrow_exports(&self) -> Ref<'_, ArrowExportSession>;
}

impl<S: SessionExt> ArrowExportSessionExt for S {
    fn arrow_exports(&self) -> Ref<'_, ArrowExportSession> {
        self.get::<ArrowExportSession>()
    }
}
