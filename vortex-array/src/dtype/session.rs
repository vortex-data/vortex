// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Module for managing extension dtypes in a Vortex session.

use std::any::Any;
use std::sync::Arc;

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::registry::Registry;

use crate::dtype::extension::ArrowCanonicalAlias;
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
}

impl Default for DTypeSession {
    fn default() -> Self {
        let this = Self {
            registry: Registry::default(),
        };

        this.register(Date);
        this.register(Time);
        this.register(Timestamp);

        this
    }
}

impl SessionVar for DTypeSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl DTypeSession {
    /// Register an extension DType with the Vortex session.
    ///
    /// The vtable's [`ExtVTable::arrow_canonical`] is consulted lazily on lookup, so the alias
    /// has a single source of truth (the vtable itself) and no per-session bookkeeping.
    pub fn register<V: ExtVTable>(&self, vtable: V) {
        self.registry
            .register(vtable.id(), Arc::new(vtable) as ExtDTypePluginRef);
    }

    /// Return the registry of extension dtypes.
    pub fn registry(&self) -> &ExtDTypeRegistry {
        &self.registry
    }

    /// Returns the canonical Arrow alias declared by `vortex_id`'s vtable, if any.
    pub fn arrow_alias_for(&self, vortex_id: &ExtId) -> Option<ArrowCanonicalAlias> {
        self.registry.find(vortex_id)?.arrow_canonical()
    }

    /// Returns the Vortex id and alias for canonical Arrow extension `arrow_id`, if a vtable
    /// declaring that alias is registered. Linear scan over registered vtables — cheap given
    /// the small number of extensions in practice.
    pub fn vortex_alias_for(&self, arrow_id: &ExtId) -> Option<(ExtId, ArrowCanonicalAlias)> {
        self.registry.items().find_map(|plugin| {
            let alias = plugin.arrow_canonical()?;
            (alias.arrow_id == *arrow_id).then(|| (plugin.id(), alias))
        })
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
