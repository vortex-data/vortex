// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Module for managing extension dtypes in a Vortex session.

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::Date;
use crate::DynVTable;
use crate::ExtId;
use crate::Time;
use crate::Timestamp;
use crate::VTable;

/// Session for managing extension dtypes.
#[derive(Debug)]
pub struct DTypeSession {
    registry: Registry<&'static dyn DynVTable>,
}

impl Default for DTypeSession {
    fn default() -> Self {
        let registry = Registry::default();

        // Register built-in temporal extension dtypes
        registry.register(Date::ID, Date);
        registry.register(Time::ID, Time);
        registry.register(Timestamp::ID, Timestamp);

        Self { registry }
    }
}

impl DTypeSession {
    /// Register an extension DType with the Vortex session.
    pub fn register<V: VTable>(
        &self,
        id: impl Into<ExtId>,
        vtable: impl Into<&'static dyn DynVTable>,
    ) {
        self.registry.register(id, vtable);
    }
}

/// Extension trait for accessing the DType session.
pub trait DTypeSessionExt: SessionExt {
    /// Get the DType session.
    fn dtypes(&self) -> Ref<DTypeSession>;
}

impl<S: SessionExt> DTypeSessionExt for S {
    fn dtypes(&self) -> Ref<DTypeSession> {
        self.get::<DTypeSession>()
    }
}
