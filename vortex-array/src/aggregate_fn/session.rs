// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;

use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnPluginRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::all_nan::AllNan;
use crate::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
use crate::aggregate_fn::fns::all_non_nan::AllNonNan;
use crate::aggregate_fn::fns::all_non_null::AllNonNull;
use crate::aggregate_fn::fns::all_null::AllNull;
use crate::aggregate_fn::fns::bounded_max::BoundedMax;
use crate::aggregate_fn::fns::bounded_min::BoundedMin;
use crate::aggregate_fn::fns::count::Count;
use crate::aggregate_fn::fns::count::CountGroupedKernel;
use crate::aggregate_fn::fns::first::First;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::last::Last;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::nan_count::NanCount;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum::SumGroupedKernel;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::arc_swap_map::ArcSwapMap;
use crate::array::ArrayId;
use crate::array::VTable;
use crate::arrays::Chunked;
use crate::arrays::Dict;
use crate::arrays::chunked::compute::aggregate::ChunkedArrayAggregate;
use crate::arrays::dict::compute::is_constant::DictIsConstantKernel;
use crate::arrays::dict::compute::is_sorted::DictIsSortedKernel;
use crate::arrays::dict::compute::min_max::DictMinMaxKernel;

/// Session state for aggregate functions and encoding-specific aggregate kernels.
///
/// The default session registers the built-in aggregate functions and kernels. Additional
/// aggregate functions and kernels may be registered by extensions when they are added to a
/// [`VortexSession`](vortex_session::VortexSession).
#[derive(Debug)]
pub struct AggregateFnSession {
    registry: ArcSwapMap<AggregateFnId, AggregateFnPluginRef>,

    kernels: ArcSwapMap<AggregateKernelKey, &'static dyn DynAggregateKernel>,
    grouped_kernels: ArcSwapMap<GroupedAggregateKernelKey, &'static dyn DynGroupedAggregateKernel>,
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
type GroupedAggregateKernelKey = (AggregateFnId, Option<ArrayId>, Option<ArrayId>);

impl Default for AggregateFnSession {
    fn default() -> Self {
        let this = Self {
            registry: ArcSwapMap::default(),
            kernels: ArcSwapMap::default(),
            grouped_kernels: ArcSwapMap::default(),
        };

        // Register the built-in aggregate functions
        this.register(AllNonDistinct);
        this.register(AllNonNan);
        this.register(AllNonNull);
        this.register(AllNan);
        this.register(AllNull);
        this.register(BoundedMax);
        this.register(BoundedMin);
        this.register(First);
        this.register(IsConstant);
        this.register(IsSorted);
        this.register(Last);
        this.register(Max);
        this.register(Min);
        this.register(MinMax);
        this.register(NanCount);
        this.register(NullCount);
        this.register(Sum);
        this.register(UncompressedSizeInBytes);

        // Register the built-in aggregate kernels.
        this.register_aggregate_kernel(Chunked.id(), None::<AggregateFnId>, &ChunkedArrayAggregate);
        this.register_aggregate_kernel(Dict.id(), Some(MinMax.id()), &DictMinMaxKernel);
        this.register_aggregate_kernel(Dict.id(), Some(IsConstant.id()), &DictIsConstantKernel);
        this.register_aggregate_kernel(Dict.id(), Some(IsSorted.id()), &DictIsSortedKernel);
        this.register_grouped_kernel(Count.id(), None, None, &CountGroupedKernel);
        this.register_grouped_kernel(Sum.id(), None, None, &SumGroupedKernel);

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
        self.kernels.read(|kernels| {
            kernels
                .get(&(id, Some(fn_id)))
                .or_else(|| kernels.get(&(id, None)))
                .copied()
        })
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

    /// Returns the grouped aggregate kernel registered for this aggregate and pair of encodings.
    ///
    /// Lookup first checks the exact `(aggregate, values encoding, group ids encoding)` key, then
    /// falls back through `(aggregate, values encoding, any group ids)`, `(aggregate, any values,
    /// group ids encoding)`, and finally `(aggregate, any values, any group ids)`.
    pub fn find_grouped_kernel(
        &self,
        agg_fn_id: impl Into<AggregateFnId>,
        values_id: impl Into<ArrayId>,
        group_ids_id: impl Into<ArrayId>,
    ) -> Option<&'static dyn DynGroupedAggregateKernel> {
        let fn_id = agg_fn_id.into();
        let values_id = values_id.into();
        let group_ids_id = group_ids_id.into();
        self.grouped_kernels.read(|kernels| {
            kernels
                .get(&(fn_id, Some(values_id), Some(group_ids_id)))
                .or_else(|| kernels.get(&(fn_id, Some(values_id), None)))
                .or_else(|| kernels.get(&(fn_id, None, Some(group_ids_id))))
                .or_else(|| kernels.get(&(fn_id, None, None)))
                .copied()
        })
    }

    /// Registers a grouped aggregate kernel.
    ///
    /// `values_id` and `group_ids_id` are optional wildcards. Passing `None` for either dimension
    /// makes the kernel a fallback for that encoding dimension.
    pub fn register_grouped_kernel(
        &self,
        agg_fn_id: impl Into<AggregateFnId>,
        values_id: Option<ArrayId>,
        group_ids_id: Option<ArrayId>,
        kernel: &'static dyn DynGroupedAggregateKernel,
    ) {
        let fn_id = agg_fn_id.into();
        self.grouped_kernels
            .insert((fn_id, values_id, group_ids_id), kernel)
    }
}

/// Extension trait for accessing aggregate function session data.
pub trait AggregateFnSessionExt: SessionExt {
    /// Returns the aggregate function session data.
    fn aggregate_fns(&self) -> Ref<'_, AggregateFnSession> {
        self.get::<AggregateFnSession>()
    }
}
impl<S: SessionExt> AggregateFnSessionExt for S {}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::aggregate_fn::AggregateFnRef;
    use crate::aggregate_fn::GroupIds;
    use crate::aggregate_fn::kernels::GroupedAggregateKernelResult;
    use crate::arrays::Constant;
    use crate::arrays::Primitive;

    #[derive(Debug)]
    struct TestGroupedKernel;

    impl DynGroupedAggregateKernel for TestGroupedKernel {
        fn grouped_aggregate(
            &self,
            _aggregate_fn: &AggregateFnRef,
            _batch: &ArrayRef,
            _group_ids: &GroupIds,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<Option<GroupedAggregateKernelResult>> {
            Ok(None)
        }
    }

    static GENERIC_KERNEL: TestGroupedKernel = TestGroupedKernel;
    static GROUP_IDS_KERNEL: TestGroupedKernel = TestGroupedKernel;
    static VALUES_KERNEL: TestGroupedKernel = TestGroupedKernel;
    static EXACT_KERNEL: TestGroupedKernel = TestGroupedKernel;

    fn assert_same_kernel(
        actual: Option<&'static dyn DynGroupedAggregateKernel>,
        expected: &'static dyn DynGroupedAggregateKernel,
    ) {
        assert!(std::ptr::eq(
            actual.expect("expected registered grouped kernel"),
            expected
        ));
    }

    #[test]
    fn grouped_kernel_lookup_prefers_exact_then_value_then_group_ids() {
        let session = AggregateFnSession::default();
        let aggregate_id = AggregateFnId::new("test.grouped_lookup");
        let values_id = Primitive.id();
        let group_ids_id = Constant.id();

        session.register_grouped_kernel(aggregate_id, None, None, &GENERIC_KERNEL);
        assert_same_kernel(
            session.find_grouped_kernel(aggregate_id, values_id, group_ids_id),
            &GENERIC_KERNEL,
        );

        session.register_grouped_kernel(aggregate_id, None, Some(group_ids_id), &GROUP_IDS_KERNEL);
        assert_same_kernel(
            session.find_grouped_kernel(aggregate_id, values_id, group_ids_id),
            &GROUP_IDS_KERNEL,
        );

        session.register_grouped_kernel(aggregate_id, Some(values_id), None, &VALUES_KERNEL);
        assert_same_kernel(
            session.find_grouped_kernel(aggregate_id, values_id, group_ids_id),
            &VALUES_KERNEL,
        );

        session.register_grouped_kernel(
            aggregate_id,
            Some(values_id),
            Some(group_ids_id),
            &EXACT_KERNEL,
        );
        assert_same_kernel(
            session.find_grouped_kernel(aggregate_id, values_id, group_ids_id),
            &EXACT_KERNEL,
        );
    }
}
