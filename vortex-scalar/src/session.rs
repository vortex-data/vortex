// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Module for managing extension dtypes in a Vortex session.

use std::sync::Arc;

use vortex_dtype::datetime::Date;
use vortex_dtype::datetime::Time;
use vortex_dtype::datetime::Timestamp;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::extension::DynExtScalarVTable;
use crate::extension::ExtScalarVTable;

/// Registry for extension dtypes.
pub type ExtScalarRegistry = Registry<Arc<dyn DynExtScalarVTable>>;

/// Session for managing extension dtypes.
#[derive(Debug)]
pub struct ScalarSession {
    registry: ExtScalarRegistry,
}

impl Default for ScalarSession {
    fn default() -> Self {
        let this = Self {
            registry: Registry::default(),
        };

        // Register built-in temporal extension scalars
        this.register(Date);
        this.register(Time);
        this.register(Timestamp);

        this
    }
}

impl ScalarSession {
    /// Register an extension Scalar with the Vortex session.
    pub fn register<V: ExtScalarVTable>(&self, vtable: V) {
        self.registry
            .register(vtable.id(), Arc::new(vtable) as Arc<dyn DynExtScalarVTable>);
    }

    /// Return the registry of extension scalars.
    pub fn registry(&self) -> &ExtScalarRegistry {
        &self.registry
    }
}

/// Extension trait for accessing the Scalar session.
pub trait ScalarSessionExt: SessionExt {
    /// Get the Scalar session.
    fn scalars(&self) -> Ref<'_, ScalarSession>;
}

impl<S: SessionExt> ScalarSessionExt for S {
    fn scalars(&self) -> Ref<'_, ScalarSession> {
        self.get::<ScalarSession>()
    }
}
