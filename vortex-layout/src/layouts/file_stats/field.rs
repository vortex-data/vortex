// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate one file field across chunks and finalize its stored summary.
//!
//! Accumulators retain mergeable states until the stream ends. Only finalized string and binary
//! extrema are truncated, so a discarded chunk bound cannot affect the final result.

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
use vortex_array::aggregate_fn::fns::min_max::supports_min_max;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stats::AggregateResults;
use vortex_array::stats::compat::truncate_results;
use vortex_error::VortexResult;

/// Mutable states owned by one file write, never attached to an array.
pub(super) struct FieldAccumulator {
    dtype: DType,
    accumulators: Vec<(AggregateFnRef, AccumulatorRef)>,
    min_max: Option<Accumulator<MinMax>>,
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

        // The default extension kernel delegates to storage. Unsupported storage must not
        // produce a known null extremum for a field with non-null values.
        let mut storage_dtype = dtype;
        while let DType::Extension(ext) = storage_dtype {
            storage_dtype = ext.storage_dtype();
        }
        let supports_extrema = supports_min_max(storage_dtype);
        let fuse_min_max =
            supports_extrema && aggregates.contains(&min) && aggregates.contains(&max);
        let min_max = fuse_min_max
            .then(|| {
                Accumulator::try_new(MinMax, NumericalAggregateOpts::skip_nans(), dtype.clone())
            })
            .transpose()?;
        let accumulators = aggregates
            .iter()
            .filter(|aggregate| {
                let is_extremum = aggregate.is::<Min>() || aggregate.is::<Max>();
                !matches!(dtype, DType::Variant(_))
                    && aggregate.return_dtype(dtype).is_some()
                    && (!is_extremum || supports_extrema)
                    && !(fuse_min_max && (*aggregate == &min || *aggregate == &max))
            })
            .map(|aggregate| Ok((aggregate.clone(), aggregate.accumulator(dtype)?)))
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
        if let Some(accumulator) = &mut self.min_max {
            accumulator.accumulate(array, ctx)?;
        }
        for (_, accumulator) in &mut self.accumulators {
            accumulator.accumulate(array, ctx)?;
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

        for (aggregate, accumulator) in &self.accumulators {
            // Historical file summaries omit extrema and flags for zero-row input.
            if !self.has_rows
                && (aggregate.is::<Min>()
                    || aggregate.is::<Max>()
                    || aggregate.is::<IsConstant>()
                    || aggregate.is::<IsSorted>())
            {
                continue;
            }

            let value = Precision::Exact(accumulator.final_scalar()?);
            entries.push((aggregate.clone(), value));
        }

        if self.has_rows
            && let Some(accumulator) = &self.min_max
        {
            let result = MinMaxResult::from_scalar(accumulator.final_scalar()?)?;
            let dtype = self.dtype.as_nullable();
            let (min, max) = match result {
                Some(result) => (result.min.cast(&dtype)?, result.max.cast(&dtype)?),
                None => (Scalar::null(dtype.clone()), Scalar::null(dtype)),
            };

            for (aggregate, value) in [
                (Min.bind(NumericalAggregateOpts::skip_nans()), min),
                (Max.bind(NumericalAggregateOpts::skip_nans()), max),
            ] {
                let value = Precision::Exact(value);
                entries.push((aggregate, value));
            }
        }

        let results = AggregateResults::try_new(&self.dtype, entries)?;

        truncate_results(&results, self.max_length)
    }
}
