//! Write-time accumulation and builders for zoned layout stats tables.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::StructArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::builder_with_capacity;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::layouts::zoned::schema::aggregate_state_dtype;

/// Accumulates aggregate-function partials for each logical zone.
pub(crate) struct AggregateStatsAccumulator {
    builders: Vec<AggregateStatsArrayBuilder>,
    length: usize,
}

impl AggregateStatsAccumulator {
    pub(crate) fn new(dtype: &DType, aggregate_fns: &[AggregateFnRef]) -> Self {
        let builders = aggregate_fns
            .iter()
            .filter_map(|aggregate_fn| {
                aggregate_state_dtype(dtype, aggregate_fn).map(|partial_dtype| {
                    AggregateStatsArrayBuilder::new(
                        aggregate_fn.clone(),
                        &partial_dtype.as_nullable(),
                        1024,
                    )
                })
            })
            .collect::<Vec<_>>();

        Self {
            builders,
            length: 0,
        }
    }

    pub(crate) fn aggregate_fns(&self) -> Arc<[AggregateFnRef]> {
        self.builders
            .iter()
            .map(|builder| builder.aggregate_fn.clone())
            .collect::<Vec<_>>()
            .into()
    }

    pub(crate) fn push_partials(&mut self, partials: Vec<Scalar>) -> VortexResult<()> {
        vortex_ensure_eq!(
            partials.len(),
            self.builders.len(),
            "aggregate partial count must match zone stats builder count"
        );

        for (builder, value) in self.builders.iter_mut().zip_eq(partials) {
            builder.append_scalar(value)?;
        }
        self.length += 1;
        Ok(())
    }

    pub(crate) fn as_array(
        &mut self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<(StructArray, Arc<[AggregateFnRef]>)>> {
        let mut names = Vec::new();
        let mut fields = Vec::new();
        let mut aggregate_fns = Vec::new();

        for builder in self
            .builders
            .iter_mut()
            .sorted_unstable_by_key(|builder| builder.aggregate_fn.to_string())
        {
            let values = builder.finish();

            if values.all_invalid(ctx)? {
                continue;
            }

            aggregate_fns.push(builder.aggregate_fn.clone());
            names.extend(values.names);
            fields.extend(values.arrays);
        }

        if names.is_empty() {
            return Ok(None);
        }

        let array = StructArray::try_new(names.into(), fields, self.length, Validity::NonNullable)?;
        Ok(Some((array, aggregate_fns.into())))
    }
}

pub(crate) fn aggregate_partials(
    array: &ArrayRef,
    aggregate_fns: &[AggregateFnRef],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Scalar>> {
    aggregate_fns
        .iter()
        .map(|aggregate_fn| {
            let mut accumulator = aggregate_fn.accumulator(array.dtype())?;
            accumulator.accumulate(array, ctx)?;
            accumulator.partial_scalar()
        })
        .collect()
}

struct AggregateStatsArrayBuilder {
    aggregate_fn: AggregateFnRef,
    dtype: DType,
    builder: Box<dyn ArrayBuilder>,
}

impl AggregateStatsArrayBuilder {
    fn new(aggregate_fn: AggregateFnRef, dtype: &DType, capacity: usize) -> Self {
        Self {
            aggregate_fn,
            dtype: dtype.clone(),
            builder: builder_with_capacity(dtype, capacity),
        }
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        self.builder.append_scalar(&value.cast(&self.dtype)?)
    }

    fn finish(&mut self) -> NamedArrays {
        NamedArrays {
            names: vec![self.aggregate_fn.to_string().into()],
            arrays: vec![self.builder.finish()],
        }
    }
}

/// Arrays with their associated names, reduced version of a `StructArray`.
struct NamedArrays {
    names: Vec<FieldName>,
    arrays: Vec<ArrayRef>,
}

impl NamedArrays {
    fn all_invalid(&self, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        // By convention the first array is the logical validity signal for the stat column.
        self.arrays[0].all_invalid(ctx)
    }
}
