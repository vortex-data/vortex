// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Retain mergeable aggregate states for one file field.
//!
//! Each chunk is finalized for its array hints, then its actual state is merged into the file
//! accumulator. String and binary extrema are truncated only after the entire field is finalized.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AccumulatorRef;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarTruncation;
use vortex_array::scalar::lower_bound;
use vortex_array::scalar::upper_bound;
use vortex_array::stats::AggregateResults;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

/// Mutable states retained until one file field is finalized.
pub(super) struct FieldAccumulator {
    dtype: DType,
    accumulators: Vec<FieldAggregate>,
    min_max: Option<FusedMinMax>,
    max_length: usize,
    seen_input: bool,
    has_rows: bool,
}

impl FieldAccumulator {
    pub(super) fn new(
        dtype: &DType,
        aggregates: &[AggregateFnRef],
        max_length: usize,
    ) -> VortexResult<Self> {
        let min = Min.bind(NumericalAggregateOpts::skip_nans());
        let max = Max.bind(NumericalAggregateOpts::skip_nans());

        // List extrema have a return type for expression lowering, but no comparison kernel.
        // Extension extrema delegate to storage, which must also support comparison.
        let mut storage_dtype = dtype;
        while let DType::Extension(ext) = storage_dtype {
            storage_dtype = ext.storage_dtype();
        }
        let supports_extrema = matches!(
            storage_dtype,
            DType::Bool(_)
                | DType::Primitive(..)
                | DType::Decimal(..)
                | DType::Utf8(_)
                | DType::Binary(_)
        );
        let fuse_min_max =
            supports_extrema && aggregates.contains(&min) && aggregates.contains(&max);
        let min_max = fuse_min_max.then(|| FusedMinMax::new(dtype)).transpose()?;
        let accumulators = aggregates
            .iter()
            .filter(|aggregate| {
                let is_extremum = aggregate.is::<Min>() || aggregate.is::<Max>();
                !matches!(dtype, DType::Variant(_))
                    && aggregate.return_dtype(dtype).is_some()
                    && (!is_extremum || supports_extrema)
                    && !(fuse_min_max && (*aggregate == &min || *aggregate == &max))
            })
            .map(|aggregate| FieldAggregate::new(aggregate.clone(), dtype))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Self {
            dtype: dtype.clone(),
            accumulators,
            min_max,
            max_length,
            seen_input: false,
            has_rows: false,
        })
    }

    pub(super) fn push_chunk(
        &mut self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Empty constants carry a scalar, but contribute no extremum or flag boundary.
        if !array.is_empty()
            && let Some(min_max) = &mut self.min_max
        {
            min_max.push_chunk(array, ctx)?;
        }
        for accumulator in &mut self.accumulators {
            if array.is_empty()
                && (accumulator.aggregate.is::<Min>()
                    || accumulator.aggregate.is::<Max>()
                    || accumulator.aggregate.is::<IsConstant>()
                    || accumulator.aggregate.is::<IsSorted>())
            {
                continue;
            }
            accumulator.push_chunk(array, ctx)?;
        }
        self.seen_input = true;
        self.has_rows |= !array.is_empty();

        Ok(())
    }

    pub(super) fn results(&self) -> VortexResult<AggregateResults> {
        if !self.seen_input {
            return Ok(AggregateResults::default());
        }

        let mut entries = Vec::new();
        for state in &self.accumulators {
            let aggregate = &state.aggregate;
            if !self.has_rows && (aggregate.is::<IsConstant>() || aggregate.is::<IsSorted>()) {
                continue;
            }

            let result = state.accumulator.final_scalar()?;
            // Historical empty and all-null extrema remain absent. A null Sum is known overflow.
            if (aggregate.is::<Min>() || aggregate.is::<Max>()) && result.is_null() {
                continue;
            }
            let value = truncate_result(aggregate, result, self.max_length)?;
            entries.push((aggregate.clone(), value));
        }

        if let Some(min_max) = &self.min_max
            && let Some(result) = MinMaxResult::from_scalar(min_max.accumulator.final_scalar()?)?
        {
            for (aggregate, result) in [
                (Min.bind(NumericalAggregateOpts::skip_nans()), result.min),
                (Max.bind(NumericalAggregateOpts::skip_nans()), result.max),
            ] {
                let result = result.cast(&self.dtype.as_nullable())?;
                let value = truncate_result(&aggregate, result, self.max_length)?;
                entries.push((aggregate, value));
            }
        }

        AggregateResults::try_new(&self.dtype, entries)
    }
}

/// One chunk's state is merged directly, preserving its grouping and overflow marker.
struct FieldAggregate {
    aggregate: AggregateFnRef,
    accumulator: AccumulatorRef,
    chunk_accumulator: AccumulatorRef,
}

impl FieldAggregate {
    fn new(aggregate: AggregateFnRef, dtype: &DType) -> VortexResult<Self> {
        Ok(Self {
            accumulator: aggregate.accumulator(dtype)?,
            chunk_accumulator: aggregate.accumulator(dtype)?,
            aggregate,
        })
    }

    fn push_chunk(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        let result = array.aggregations().compute_into(
            &self.aggregate,
            &mut *self.chunk_accumulator,
            ctx,
        )?;
        let is_nan_sum = self.aggregate.is::<Sum>()
            && result
                .as_primitive_opt()
                .is_some_and(|value| value.is_nan());

        // The file writer's NaN-skipping policy also applies to chunk finals. Publish their hints
        // first, then skip only NaN-valued float states. Null overflow states must still merge.
        if !is_nan_sum {
            self.accumulator.merge_from(&mut *self.chunk_accumulator)?;
        }

        Ok(())
    }
}

struct FusedMinMax {
    aggregate: AggregateFnRef,
    accumulator: Accumulator<MinMax>,
    chunk_accumulator: Accumulator<MinMax>,
}

impl FusedMinMax {
    fn new(dtype: &DType) -> VortexResult<Self> {
        Ok(Self {
            aggregate: MinMax.bind(NumericalAggregateOpts::skip_nans()),
            accumulator: Accumulator::try_new(
                MinMax,
                NumericalAggregateOpts::skip_nans(),
                dtype.clone(),
            )?,
            chunk_accumulator: Accumulator::try_new(
                MinMax,
                NumericalAggregateOpts::skip_nans(),
                dtype.clone(),
            )?,
        })
    }

    fn push_chunk(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        array
            .aggregations()
            .compute_into(&self.aggregate, &mut self.chunk_accumulator, ctx)?;
        self.accumulator.merge_from(&mut self.chunk_accumulator)?;

        Ok(())
    }
}

fn truncate_result(
    aggregate: &AggregateFnRef,
    result: Scalar,
    max_length: usize,
) -> VortexResult<Precision<Scalar>> {
    if !aggregate.is::<Min>() && !aggregate.is::<Max>() {
        return Ok(Precision::Exact(result));
    }

    let is_max = aggregate.is::<Max>();
    match result.dtype() {
        DType::Utf8(_) => truncate_varlen::<BufferString>(result, max_length, is_max),
        DType::Binary(_) => truncate_varlen::<ByteBuffer>(result, max_length, is_max),
        _ => Ok(Precision::Exact(result)),
    }
}

fn truncate_varlen<T: ScalarTruncation>(
    result: Scalar,
    max_length: usize,
    is_max: bool,
) -> VortexResult<Precision<Scalar>> {
    let nullability = result.dtype().nullability();
    let value = T::from_scalar(result)?;
    let bound = if is_max {
        upper_bound(value, max_length, nullability)
    } else {
        lower_bound(value, max_length, nullability)
    };
    Ok(match bound {
        Some((scalar, true)) => Precision::Inexact(scalar),
        Some((scalar, false)) => Precision::Exact(scalar),
        None => Precision::Absent,
    })
}
