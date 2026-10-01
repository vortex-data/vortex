// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Retain typed state without a scalar roundtrip.
//!
//! Finalized results and typed states share one store lock. Typed lookup includes the concrete
//! vtable type, since unrelated vtables can use the same function ID and partial Rust type.

use std::any::Any;
use std::any::TypeId;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_error::VortexResult;

use super::AggregationsRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateDTypes;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::scalar::Scalar;

pub(super) struct PartialEntry {
    aggregate: AggregateFnRef,
    vtable_type: TypeId,
    retained: Arc<dyn DynRetainedPartial>,
}

impl Debug for PartialEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartialEntry")
            .field("aggregate", &self.aggregate)
            .field("vtable_type", &self.vtable_type)
            .finish_non_exhaustive()
    }
}

trait DynRetainedPartial: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn matches_vtable(&self, aggregate: &AggregateFnRef) -> bool;
    fn finalize(&self) -> VortexResult<Scalar>;
}

struct RetainedPartial<V: AggregateFnVTable> {
    vtable: V,
    options: V::Options,
    dtypes: AggregateDTypes,
    partial: Arc<V::Partial>,
}

impl<V: AggregateFnVTable> DynRetainedPartial for RetainedPartial<V>
where
    V::Partial: Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn matches_vtable(&self, aggregate: &AggregateFnRef) -> bool {
        aggregate.is::<V>()
    }

    fn finalize(&self) -> VortexResult<Scalar> {
        self.vtable
            .finalize_scalar(self.dtypes.args(&self.options), &self.partial)
    }
}

impl AggregationsRef<'_> {
    pub(crate) fn finalize_retained_result(
        &self,
        aggregate: &AggregateFnRef,
    ) -> VortexResult<Option<Scalar>> {
        // Finalization can call another aggregate. Clone the state reference and release the lock
        // before calling its vtable, just as for a fresh accumulation.
        let retained = self
            .aggregations
            .entries
            .read()
            .partials
            .iter()
            .find_map(|entry| {
                (&entry.aggregate == aggregate && entry.retained.matches_vtable(aggregate))
                    .then(|| Arc::clone(&entry.retained))
            });
        retained.map(|partial| partial.finalize()).transpose()
    }

    pub(crate) fn compute_partial<V>(
        &self,
        aggregate: &AggregateFn<V>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Arc<V::Partial>>
    where
        V: AggregateFnVTable,
        V::Partial: Sync,
    {
        if let Some(partial) = self.get_partial(aggregate) {
            return Ok(partial);
        }

        let dtypes = AggregateDTypes::try_new(
            aggregate.vtable(),
            aggregate.options(),
            self.array.dtype().clone(),
        )?;
        let mut accumulator = Accumulator::from_dtypes(
            aggregate.vtable().clone(),
            aggregate.options().clone(),
            dtypes.clone(),
        );
        // A finalized result can omit information needed by the typed state. Always accumulate
        // a miss instead of recovering state from the scalar cache.
        accumulator.accumulate_uncached(self.array, ctx)?;
        let partial = Arc::new(accumulator.into_partial()?);
        let retained = Arc::new(RetainedPartial {
            vtable: aggregate.vtable().clone(),
            options: aggregate.options().clone(),
            dtypes,
            partial: Arc::clone(&partial),
        });

        let mut entries = self.aggregations.entries.write();
        // Concurrent requests may race while accumulating. Return the state already retained so
        // every successful lookup shares one allocation.
        if let Some(existing) = find_partial(&entries.partials, aggregate) {
            return Ok(existing);
        }
        entries.partials.push(PartialEntry {
            aggregate: AggregateFn::new(aggregate.vtable().clone(), aggregate.options().clone())
                .erased(),
            vtable_type: TypeId::of::<V>(),
            retained,
        });
        Ok(partial)
    }

    fn get_partial<V: AggregateFnVTable>(
        &self,
        aggregate: &AggregateFn<V>,
    ) -> Option<Arc<V::Partial>>
    where
        V::Partial: Sync,
    {
        find_partial(&self.aggregations.entries.read().partials, aggregate)
    }
}

fn find_partial<V: AggregateFnVTable>(
    entries: &[PartialEntry],
    aggregate: &AggregateFn<V>,
) -> Option<Arc<V::Partial>>
where
    V::Partial: Sync,
{
    let id = aggregate.vtable().id();
    entries
        .iter()
        .find(|entry| {
            entry.vtable_type == TypeId::of::<V>()
                && entry.aggregate.id() == id
                && entry.aggregate.as_opt::<V>() == Some(aggregate.options())
        })
        .and_then(|entry| entry.retained.as_any().downcast_ref::<RetainedPartial<V>>())
        .map(|retained| Arc::clone(&retained.partial))
}
