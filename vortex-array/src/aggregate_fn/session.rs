// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_utils::aliases::hash_map::HashMap;

use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnPluginRef;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::array::ArrayId;
use crate::array::VTable;
use crate::arrays::Chunked;
use crate::arrays::chunked::compute::aggregate::ChunkedArrayAggregate;
use crate::dtype::DType;

/// Session state for aggregate functions and encoding-specific aggregate kernels.
///
/// The default session registers the built-in aggregate functions and kernels. Additional
/// aggregate functions and kernels may be registered by extensions when they are added to a
/// [`VortexSession`](vortex_session::VortexSession).
#[derive(Clone, Debug)]
pub struct AggregateFnSession {
    registry: AggregateFnRegistry,

    kernels: AggregateKernelRegistry,
    grouped_kernels: GroupedKernelRegistry,
    grouped_encoding_kernels: GroupedEncodingKernelRegistry,
}

impl SessionVar for AggregateFnSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

type AggregateKernelKey = (ArrayId, Option<AggregateFnId>);
type GroupedEncodingKernelKey = (ArrayId, AggregateFnId);

/// Registry of aggregate function plugins, keyed by aggregate function id.
type AggregateFnRegistry = ArcSwapMap<AggregateFnId, AggregateFnPluginRef>;
/// Registry of aggregate kernels, keyed by encoding and optional aggregate function.
type AggregateKernelRegistry = ArcSwapMap<AggregateKernelKey, &'static dyn DynAggregateKernel>;
/// Registry of encoding-agnostic grouped aggregate kernels, keyed by aggregate function id.
type GroupedKernelRegistry = ArcSwapMap<AggregateFnId, &'static dyn DynGroupedAggregateKernel>;
/// Registry of grouped aggregate kernels, keyed by encoding and aggregate function.
type GroupedEncodingKernelRegistry =
    ArcSwapMap<GroupedEncodingKernelKey, &'static dyn DynGroupedAggregateKernel>;

impl Default for AggregateFnSession {
    fn default() -> Self {
        let this = Self {
            registry: AggregateFnRegistry::default(),
            kernels: AggregateKernelRegistry::default(),
            grouped_kernels: GroupedKernelRegistry::default(),
            grouped_encoding_kernels: GroupedEncodingKernelRegistry::default(),
        };

        // Register the built-in aggregate functions.
        this.register(AllNonDistinct);

        // Register the built-in aggregate kernels.
        this.register_aggregate_kernel(Chunked.id(), None::<AggregateFnId>, &ChunkedArrayAggregate);

        this
    }
}

impl AggregateFnSession {
    /// Returns the aggregate function plugin registered for `id`, if any.
    pub fn find_plugin(&self, id: &AggregateFnId) -> Option<AggregateFnPluginRef> {
        self.registry.get(id)
    }

    /// Register an aggregate function vtable in the session, replacing any existing vtable with
    /// the same ID.
    pub fn register<V: AggregateFnVTable>(&self, vtable: V) {
        let id = vtable.id();
        let pluginref = Arc::new(vtable) as AggregateFnPluginRef;
        self.registry.insert(id, pluginref);
    }

    /// The default per-chunk zone statistics for a column of `input_dtype`, collected from every
    /// registered aggregate's `zone_stat_default`.
    ///
    /// Each call scans the whole plugin registry, so this is intended to be called once per
    /// column when a zoned writer is opened, not per chunk or per row.
    pub fn zone_stat_defaults(&self, input_dtype: &DType) -> Vec<AggregateFnRef> {
        self.registry.read(|registry| {
            let mut fns: Vec<AggregateFnRef> = registry
                .values()
                .filter_map(|plugin| plugin.zone_stat_default(input_dtype))
                .collect();
            fns.sort_by_key(|f| f.id());
            fns
        })
    }

    /// Returns the aggregate kernel registered for `array_id` and `agg_fn_id`, if any.
    ///
    /// Lookup first checks for a kernel registered for the exact aggregate function, then falls
    /// back to a kernel registered for all aggregate functions on the same array encoding.
    pub fn find_aggregate_kernel(
        &self,
        array_id: impl Into<ArrayId>,
        agg_fn_id: impl Into<AggregateFnId>,
    ) -> Option<&'static dyn DynAggregateKernel> {
        let id = array_id.into();
        let fn_id = agg_fn_id.into();
        self.kernels
            .read(|kernels| lookup_aggregate_kernel(kernels, id, fn_id))
    }

    /// Registers an aggregate kernel for an array encoding.
    ///
    /// When `agg_fn_id` is `Some`, the kernel is used only for that aggregate function. When
    /// `agg_fn_id` is `None`, the kernel is used as the fallback for aggregate functions on the
    /// array encoding that do not have a more specific kernel.
    pub fn register_aggregate_kernel(
        &self,
        array_id: impl Into<ArrayId>,
        agg_fn_id: Option<impl Into<AggregateFnId>>,
        kernel: &'static dyn DynAggregateKernel,
    ) {
        let id = (array_id.into(), agg_fn_id.map(|id| id.into()));
        self.kernels.insert(id, kernel);
    }

    /// Returns the grouped aggregate kernel registered for `agg_fn_id`, if any.
    ///
    /// These kernels are independent of the element encoding and are checked for each element
    /// representation, after any kernel registered for the current element encoding.
    pub fn find_grouped_kernel(
        &self,
        agg_fn_id: impl Into<AggregateFnId>,
    ) -> Option<&'static dyn DynGroupedAggregateKernel> {
        let fn_id = agg_fn_id.into();
        self.grouped_kernels
            .read(|kernels| kernels.get(&fn_id).copied())
    }

    /// Registers a grouped aggregate kernel for an aggregate function.
    pub fn register_grouped_kernel(
        &self,
        agg_fn_id: impl Into<AggregateFnId>,
        kernel: &'static dyn DynGroupedAggregateKernel,
    ) {
        let fn_id = agg_fn_id.into();
        self.grouped_kernels.insert(fn_id, kernel)
    }

    /// Returns the grouped aggregate kernel registered for `array_id` and `agg_fn_id`, if any.
    ///
    /// These kernels are matched against each intermediate element encoding while the grouped
    /// accumulator executes the element array.
    pub fn find_grouped_encoding_kernel(
        &self,
        array_id: impl Into<ArrayId>,
        agg_fn_id: impl Into<AggregateFnId>,
    ) -> Option<&'static dyn DynGroupedAggregateKernel> {
        let id = array_id.into();
        let fn_id = agg_fn_id.into();
        self.grouped_encoding_kernels
            .read(|kernels| kernels.get(&(id, fn_id)).copied())
    }

    /// Registers a grouped aggregate kernel for a specific aggregate function and array encoding.
    pub fn register_grouped_encoding_kernel(
        &self,
        array_id: impl Into<ArrayId>,
        agg_fn_id: impl Into<AggregateFnId>,
        kernel: &'static dyn DynGroupedAggregateKernel,
    ) {
        let id = array_id.into();
        let fn_id = agg_fn_id.into();
        self.grouped_encoding_kernels.insert((id, fn_id), kernel)
    }

    /// Captures the currently registered aggregate kernels, so repeated lookups skip the session
    /// and registry loads.
    pub(crate) fn kernels_snapshot(&self) -> AggregateKernels {
        AggregateKernels {
            kernels: self.kernels.snapshot(),
            grouped_kernels: self.grouped_kernels.snapshot(),
            grouped_encoding_kernels: self.grouped_encoding_kernels.snapshot(),
        }
    }
}

/// A point-in-time snapshot of the aggregate kernel registries of an [`AggregateFnSession`].
///
/// Lookups follow the same rules as the corresponding `find_*` methods on
/// [`AggregateFnSession`].
#[derive(Clone, Debug)]
pub(crate) struct AggregateKernels {
    kernels: Arc<HashMap<AggregateKernelKey, &'static dyn DynAggregateKernel>>,
    grouped_kernels: Arc<HashMap<AggregateFnId, &'static dyn DynGroupedAggregateKernel>>,
    grouped_encoding_kernels:
        Arc<HashMap<GroupedEncodingKernelKey, &'static dyn DynGroupedAggregateKernel>>,
}

impl AggregateKernels {
    /// See [`AggregateFnSession::find_aggregate_kernel`].
    pub(crate) fn find_aggregate_kernel(
        &self,
        array_id: ArrayId,
        agg_fn_id: AggregateFnId,
    ) -> Option<&'static dyn DynAggregateKernel> {
        lookup_aggregate_kernel(&self.kernels, array_id, agg_fn_id)
    }

    /// See [`AggregateFnSession::find_grouped_kernel`].
    pub(crate) fn find_grouped_kernel(
        &self,
        agg_fn_id: AggregateFnId,
    ) -> Option<&'static dyn DynGroupedAggregateKernel> {
        self.grouped_kernels.get(&agg_fn_id).copied()
    }

    /// See [`AggregateFnSession::find_grouped_encoding_kernel`].
    pub(crate) fn find_grouped_encoding_kernel(
        &self,
        array_id: ArrayId,
        agg_fn_id: AggregateFnId,
    ) -> Option<&'static dyn DynGroupedAggregateKernel> {
        self.grouped_encoding_kernels
            .get(&(array_id, agg_fn_id))
            .copied()
    }
}

fn lookup_aggregate_kernel(
    kernels: &HashMap<AggregateKernelKey, &'static dyn DynAggregateKernel>,
    array_id: ArrayId,
    agg_fn_id: AggregateFnId,
) -> Option<&'static dyn DynAggregateKernel> {
    kernels
        .get(&(array_id, Some(agg_fn_id)))
        .or_else(|| kernels.get(&(array_id, None)))
        .copied()
}

/// Extension trait for accessing aggregate function session data.
pub trait AggregateFnSessionExt: SessionExt {
    /// Returns the aggregate function session data.
    fn aggregate_fns(&self) -> SessionGuard<'_, AggregateFnSession> {
        self.get::<AggregateFnSession>()
    }
}
impl<S: SessionExt> AggregateFnSessionExt for S {}
