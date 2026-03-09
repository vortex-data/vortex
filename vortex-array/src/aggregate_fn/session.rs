// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use parking_lot::RwLock;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnPluginRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::arrays::ChunkedVTable;
use crate::arrays::chunked::compute::aggregate::ChunkedArrayAggregate;
use crate::vtable::ArrayId;

/// Registry of aggregate function vtables.
pub type AggregateFnRegistry = Registry<AggregateFnPluginRef>;

/// Session state for aggregate function vtables.
#[derive(Debug)]
pub struct AggregateFnSession {
    registry: AggregateFnRegistry,

    pub(super) kernels: RwLock<HashMap<KernelKey, &'static dyn DynAggregateKernel>>,
    pub(super) grouped_kernels: RwLock<HashMap<KernelKey, &'static dyn DynGroupedAggregateKernel>>,
}

type KernelKey = (ArrayId, Option<AggregateFnId>);

impl Default for AggregateFnSession {
    fn default() -> Self {
        let this = Self {
            registry: AggregateFnRegistry::default(),
            kernels: RwLock::new(HashMap::default()),
            grouped_kernels: RwLock::new(HashMap::default()),
        };

        // Register the built-in aggregate kernels.
        this.register_aggregate_kernel(ChunkedVTable::ID, None, &ChunkedArrayAggregate);

        this
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

    /// Register an aggregate function kernel for a specific aggregate function and array type.
    pub fn register_aggregate_kernel(
        &self,
        array_id: ArrayId,
        agg_fn_id: Option<AggregateFnId>,
        kernel: &'static dyn DynAggregateKernel,
    ) {
        self.kernels.write().insert((array_id, agg_fn_id), kernel);
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
