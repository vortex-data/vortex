//! Write-time accumulation and builders for zoned layout stats tables.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AccumulatorRef;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::StructArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::layouts::file_stats::contains_variant;
use crate::layouts::zoned::schema::aggregate_state_dtype;

/// Accumulates aggregate-function partials for each logical zone.
///
/// The stream arrives as pieces that never cross a zone boundary, each with the partials computed
/// over its rows. The partials of the pieces of one zone are merged, so no zone is ever held as
/// an array.
pub(crate) struct AggregateStatsAccumulator {
    builders: Vec<AggregateStatsArrayBuilder>,
    length: usize,
    zone_len: usize,
    /// Merges the partials of two pieces of a zone, one accumulator for each builder. They are
    /// empty between merges.
    mergers: Vec<AccumulatorRef>,
    /// The merged partials of the zone that is still receiving rows, if `open_len` is not zero.
    open_partials: Vec<Scalar>,
    open_len: usize,
}

impl AggregateStatsAccumulator {
    pub(crate) fn try_new(
        dtype: &DType,
        aggregate_fns: &[AggregateFnRef],
        zone_len: usize,
    ) -> VortexResult<Self> {
        let builders = aggregate_fns
            .iter()
            .filter_map(|aggregate_fn| {
                aggregate_state_dtype(dtype, aggregate_fn)
                    // A variant state has no builder, so a zone map cannot hold it.
                    .filter(|partial_dtype| !contains_variant(partial_dtype))
                    .map(|partial_dtype| {
                        AggregateStatsArrayBuilder::new(
                            aggregate_fn.clone(),
                            &partial_dtype.as_nullable(),
                            1024,
                        )
                    })
            })
            .collect::<Vec<_>>();
        let mergers = builders
            .iter()
            .map(|builder| builder.aggregate_fn.accumulator(dtype))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Self {
            builders,
            length: 0,
            zone_len,
            mergers,
            open_partials: Vec::new(),
            open_len: 0,
        })
    }

    pub(crate) fn aggregate_fns(&self) -> Arc<[AggregateFnRef]> {
        self.builders
            .iter()
            .map(|builder| builder.aggregate_fn.clone())
            .collect::<Vec<_>>()
            .into()
    }

    /// Adds the next `len` rows of the stream, given the partials computed over them, one for
    /// each aggregate in [`Self::aggregate_fns`]. The rows must not cross a zone boundary.
    pub(crate) fn push_piece(&mut self, len: usize, partials: Vec<Scalar>) -> VortexResult<()> {
        vortex_ensure!(
            len > 0 && self.open_len + len <= self.zone_len,
            "a piece of {len} rows does not fit a zone of {} rows holding {} rows",
            self.zone_len,
            self.open_len
        );
        vortex_ensure_eq!(
            partials.len(),
            self.builders.len(),
            "aggregate partial count must match zone stats builder count"
        );

        if self.open_len == 0 {
            // The first piece of a zone has nothing to merge with, so a zone that arrives as one
            // piece records exactly the partials computed over it.
            self.open_partials = partials;
        } else {
            for ((merger, open), partial) in self
                .mergers
                .iter_mut()
                .zip_eq(self.open_partials.iter_mut())
                .zip_eq(partials)
            {
                merger.combine_partials(open)?;
                merger.combine_partials(&partial)?;
                *open = merger.flush()?;
            }
        }

        self.open_len += len;
        if self.open_len == self.zone_len {
            self.close_zone()?;
        }
        Ok(())
    }

    /// Closes the final zone, if the stream did not end on a zone boundary.
    pub(crate) fn finish(&mut self) -> VortexResult<()> {
        if self.open_len > 0 {
            self.close_zone()?;
        }
        Ok(())
    }

    fn close_zone(&mut self) -> VortexResult<()> {
        self.open_len = 0;
        for (builder, value) in self
            .builders
            .iter_mut()
            .zip_eq(mem::take(&mut self.open_partials))
        {
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
            builder: builder_with_capacity_in(
                dtype,
                capacity,
                vortex_buffer::BufferAllocatorRef::static_ref(),
            ),
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
