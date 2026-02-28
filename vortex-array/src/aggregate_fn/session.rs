// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::aggregate_fn::AggregateFnPluginRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::mean::Mean;
use crate::aggregate_fn::fns::min_max::Max;
use crate::aggregate_fn::fns::min_max::Min;
use crate::aggregate_fn::fns::sum::Sum;

/// Registry of aggregate function vtables.
pub type AggregateFnRegistry = Registry<AggregateFnPluginRef>;

/// Session state for aggregate function vtables.
#[derive(Debug)]
pub struct AggregateFnSession {
    registry: AggregateFnRegistry,
}

impl Default for AggregateFnSession {
    fn default() -> Self {
        let session = Self {
            registry: AggregateFnRegistry::default(),
        };
        session.register(Mean);
        session.register(Min);
        session.register(Max);
        session.register(Sum);
        session
    }
}

impl AggregateFnSession {
    /// Returns the aggregate function registry.
    pub fn registry(&self) -> &AggregateFnRegistry {
        &self.registry
    }

    /// Register an aggregate function vtable in the session, replacing any existing vtable with
    /// the same ID.
    pub fn register<V: AggregateFnVTable>(&self, vtable: V) {
        self.registry
            .register(vtable.id(), Arc::new(vtable) as AggregateFnPluginRef);
    }
}

/// Extension trait for accessing aggregate function session data.
pub trait AggregateFnSessionExt: SessionExt {
    /// Returns the aggregate function vtable registry.
    fn aggregate_fns(&self) -> Ref<'_, AggregateFnSession> {
        self.get::<AggregateFnSession>()
    }
}
impl<S: SessionExt> AggregateFnSessionExt for S {}
