// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Module for managing extension dtypes in a Vortex session.

use std::sync::Arc;

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::dtype::datetime::Date;
use crate::dtype::datetime::Time;
use crate::dtype::datetime::Timestamp;
use crate::dtype::extension::DynExtDTypeVTable;
use crate::dtype::extension::ExtDTypeVTable;

/// Registry for extension dtypes.
pub type ExtDTypeRegistry = Registry<Arc<dyn DynExtDTypeVTable>>;

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

        this
    }
}

impl DTypeSession {
    /// Register an extension DType with the Vortex session.
    pub fn register<V: ExtDTypeVTable>(&self, vtable: V) {
        self.registry
            .register(vtable.id(), Arc::new(vtable) as Arc<dyn DynExtDTypeVTable>);
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
