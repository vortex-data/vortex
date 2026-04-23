// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Module for managing extension dtypes in a Vortex session.

use std::sync::Arc;

use parking_lot::RwLock;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::dtype::extension::ExtDTypePluginRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::datetime::Date;
use crate::extension::datetime::Time;
use crate::extension::datetime::Timestamp;

/// Registry for extension dtypes.
pub type ExtDTypeRegistry = Registry<ExtDTypePluginRef>;

/// Session for managing extension dtypes.
#[derive(Debug)]
pub struct DTypeSession {
    registry: ExtDTypeRegistry,
    arrow_canonical: RwLock<ArrowCanonicalAliases>,
}

#[derive(Debug, Default)]
struct ArrowCanonicalAliases {
    entries: Vec<(ExtId, &'static str)>,
}

impl Default for DTypeSession {
    fn default() -> Self {
        let this = Self {
            registry: Registry::default(),
            arrow_canonical: RwLock::default(),
        };

        this.register(Date);
        this.register(Time);
        this.register(Timestamp);

        this
    }
}

impl DTypeSession {
    /// Register an extension DType with the Vortex session.
    pub fn register<V: ExtVTable>(&self, vtable: V) {
        self.registry
            .register(vtable.id(), Arc::new(vtable) as ExtDTypePluginRef);
    }

    /// Return the registry of extension dtypes.
    pub fn registry(&self) -> &ExtDTypeRegistry {
        &self.registry
    }

    /// Register an Arrow canonical extension name as an alias for a Vortex extension id.
    /// Aliased extensions emit the canonical name on `ARROW:extension:name` and serialize
    /// metadata as raw UTF-8 instead of base64-wrapped bytes.
    pub fn register_arrow_canonical(&self, vortex_id: ExtId, arrow_name: &'static str) {
        let mut aliases = self.arrow_canonical.write();
        aliases.entries.retain(|(v, _)| *v != vortex_id);
        aliases.entries.push((vortex_id, arrow_name));
    }

    /// Returns the Arrow canonical extension name aliased to the given Vortex id, if any.
    pub fn arrow_canonical_for(&self, vortex_id: &ExtId) -> Option<&'static str> {
        self.arrow_canonical
            .read()
            .entries
            .iter()
            .find(|(v, _)| v == vortex_id)
            .map(|(_, a)| *a)
    }

    /// Returns the Vortex extension id aliased to the given Arrow canonical name, if any.
    pub fn vortex_id_for_arrow_canonical(&self, arrow_name: &str) -> Option<ExtId> {
        self.arrow_canonical
            .read()
            .entries
            .iter()
            .find(|(_, a)| *a == arrow_name)
            .map(|(v, _)| *v)
    }
}

/// Extension trait for accessing the DType session.
pub trait DTypeSessionExt: SessionExt {
    /// Get the DType session.
    fn dtypes(&self) -> Ref<'_, DTypeSession>;
}

impl<S: SessionExt> DTypeSessionExt for S {
    fn dtypes(&self) -> Ref<'_, DTypeSession> {
        self.get::<DTypeSession>()
    }
}
