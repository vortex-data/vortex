// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;

use crate::higher_order_fn::HigherOrderFnId;
use crate::higher_order_fn::HigherOrderFnPluginRef;
use crate::higher_order_fn::HigherOrderFnVTable;

/// Registry of higher-order function vtables.
pub type HigherOrderFnRegistry = ArcSwapMap<HigherOrderFnId, HigherOrderFnPluginRef>;

/// Session state for higher-order function vtables.
#[derive(Clone, Debug, Default)]
pub struct HigherOrderFnSession {
    registry: HigherOrderFnRegistry,
}

impl HigherOrderFnSession {
    pub fn registry(&self) -> &HigherOrderFnRegistry {
        &self.registry
    }

    /// Register a vtable, replacing any existing vtable with the same ID.
    pub fn register<V: HigherOrderFnVTable>(&self, vtable: V) {
        self.registry
            .insert(vtable.id(), Arc::new(vtable) as HigherOrderFnPluginRef);
    }
}

impl SessionVar for HigherOrderFnSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing higher-order-function session state.
pub trait HigherOrderFnSessionExt: SessionExt {
    /// Return the higher-order function vtable registry.
    fn higher_order_fns(&self) -> SessionGuard<'_, HigherOrderFnSession> {
        self.get::<HigherOrderFnSession>()
    }
}

impl<S: SessionExt> HigherOrderFnSessionExt for S {}
