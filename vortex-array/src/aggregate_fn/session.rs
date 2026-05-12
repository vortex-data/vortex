// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::registry::Registry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnPluginRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
use crate::aggregate_fn::fns::first::First;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::last::Last;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::nan_count::NanCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::array::ArrayId;
use crate::array::VTable;
use crate::arrays::Chunked;
use crate::arrays::Dict;
use crate::arrays::chunked::compute::aggregate::ChunkedArrayAggregate;
use crate::arrays::dict::compute::is_constant::DictIsConstantKernel;
use crate::arrays::dict::compute::is_sorted::DictIsSortedKernel;
use crate::arrays::dict::compute::min_max::DictMinMaxKernel;

/// Registry of aggregate function vtables.
pub type AggregateFnRegistry = Registry<AggregateFnPluginRef>;

/// Session state for aggregate function vtables.
#[derive(Debug)]
pub struct AggregateFnSession {
    registry: AggregateFnRegistry,

    pub(super) kernels: RwLock<HashMap<KernelKey, &'static dyn DynAggregateKernel>>,
    pub(super) grouped_kernels: RwLock<HashMap<KernelKey, &'static dyn DynGroupedAggregateKernel>>,
}

impl SessionVar for AggregateFnSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

type KernelKey = (ArrayId, Option<AggregateFnId>);

impl Default for AggregateFnSession {
    fn default() -> Self {
        let this = Self {
            registry: AggregateFnRegistry::default(),
            kernels: RwLock::new(HashMap::default()),
            grouped_kernels: RwLock::new(HashMap::default()),
        };

        // Register the built-in aggregate functions
        this.register(AllNonDistinct);
        this.register(First);
        this.register(IsConstant);
        this.register(IsSorted);
        this.register(Last);
        this.register(MinMax);
        this.register(NanCount);
        this.register(Sum);
        this.register(UncompressedSizeInBytes);

        // Register the built-in aggregate kernels.
        this.register_aggregate_kernel(Chunked.id(), None::<AggregateFnId>, &ChunkedArrayAggregate);
        this.register_aggregate_kernel(Dict.id(), Some(MinMax.id()), &DictMinMaxKernel);
        this.register_aggregate_kernel(Dict.id(), Some(IsConstant.id()), &DictIsConstantKernel);
        this.register_aggregate_kernel(Dict.id(), Some(IsSorted.id()), &DictIsSortedKernel);

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
        array_id: impl Into<ArrayId>,
        agg_fn_id: Option<impl Into<AggregateFnId>>,
        kernel: &'static dyn DynAggregateKernel,
    ) {
        self.kernels
            .write()
            .insert((array_id.into(), agg_fn_id.map(|id| id.into())), kernel);
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
