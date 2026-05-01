// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Module for managing extension dtypes in a Vortex session.

use std::any::Any;
use std::sync::Arc;

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::registry::Registry;

use crate::dtype::extension::ExtDTypePluginRef;
use crate::dtype::extension::ExtVTable;
use crate::extension::Lossy;
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

        // Register built-in temporal extension dtypes
        this.register(Date);
        this.register(Time);
        this.register(Timestamp);

        // Register the `Lossy` extension dtype so columns marked as Lossy can be deserialized
        // out of files written by vortex-file (and any other path that depends on the default
        // session registry).
        this.register(Lossy);

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
    pub fn register<V: ExtVTable>(&self, vtable: V) {
        self.registry
            .register(vtable.id(), Arc::new(vtable) as ExtDTypePluginRef);
    }

    /// Return the registry of extension dtypes.
    pub fn registry(&self) -> &ExtDTypeRegistry {
        &self.registry
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
