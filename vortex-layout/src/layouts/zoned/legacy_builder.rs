//! Write-time accumulation of legacy `vortex.stats` zone tables.
//!
//! Readers before `vortex.zoned` (Vortex 0.83 and earlier) only know the `vortex.stats` layout.
//! Its zone table holds one column for each [`Stat`], in [`Stat`] order. A `max` or `min` column
//! has a `max_is_truncated` or `min_is_truncated` column after it. This module writes the same
//! table that those readers wrote.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::BoolBuilder;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarTruncation;
use vortex_array::scalar::lower_bound;
use vortex_array::scalar::upper_bound;
use vortex_array::validity::Validity;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::layouts::zoned::schema::MAX_IS_TRUNCATED;
use crate::layouts::zoned::schema::MIN_IS_TRUNCATED;

/// The stats that a legacy zone table records.
///
/// Sum is not included, because it does not help pruning.
const LEGACY_ZONE_STATS: &[Stat] = &[Stat::Max, Stat::Min, Stat::NullCount, Stat::NaNCount];

/// The maximum length in bytes of a string or binary min or max value in a legacy zone table.
const MAX_VARIABLE_LENGTH_STAT_BYTES: usize = 64;

/// Accumulates the legacy stats of each zone.
pub(crate) struct LegacyStatsAccumulator {
    builders: Vec<Box<dyn LegacyStatsArrayBuilder>>,
    length: usize,
}

impl LegacyStatsAccumulator {
    pub(crate) fn new(dtype: &DType) -> Self {
        if matches!(dtype, DType::Variant(_)) {
            return Self {
                builders: Vec::new(),
                length: 0,
            };
        }

        let builders = LEGACY_ZONE_STATS
            .iter()
            .filter_map(|&stat| {
                stat.dtype(dtype)
                    .map(|stat_dtype| legacy_stats_builder(stat, &stat_dtype.as_nullable()))
            })
            .collect::<Vec<_>>();

        Self {
            builders,
            length: 0,
        }
    }

    /// The stats that this accumulator records, in the order of [`Self::push_values`].
    pub(crate) fn stats(&self) -> Arc<[Stat]> {
        self.builders.iter().map(|builder| builder.stat()).collect()
    }

    /// Adds the stats of one zone. `values` has one entry for each stat in [`Self::stats`].
    pub(crate) fn push_values(&mut self, values: Vec<Option<Scalar>>) -> VortexResult<()> {
        vortex_ensure_eq!(
            values.len(),
            self.builders.len(),
            "legacy stat value count must match legacy zone stats builder count"
        );

        for (builder, value) in self.builders.iter_mut().zip_eq(values) {
            match value {
                Some(value) => builder.append_scalar(value.cast(&value.dtype().as_nullable())?)?,
                None => builder.append_null(),
            }
        }

        self.length += 1;
        Ok(())
    }

    /// Returns the zone table and its stats in [`Stat`] order. Stats that are null in all zones
    /// are not included. Returns `None` if no stat is left.
    pub(crate) fn as_array(
        &mut self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<(StructArray, Arc<[Stat]>)>> {
        let mut names = Vec::new();
        let mut fields = Vec::new();
        let mut stats = Vec::new();

        for builder in self
            .builders
            .iter_mut()
            .sorted_unstable_by_key(|builder| builder.stat())
        {
            let values = builder.finish();

            // The first array is the value column of the stat.
            if values.arrays[0].all_invalid(ctx)? {
                continue;
            }

            stats.push(builder.stat());
            names.extend(values.names);
            fields.extend(values.arrays);
        }

        if names.is_empty() {
            return Ok(None);
        }

        let array = StructArray::try_new(names.into(), fields, self.length, Validity::NonNullable)?;
        Ok(Some((array, stats.into())))
    }
}

/// Computes the stats of `stats` for one zone.
pub(crate) fn legacy_stat_values(
    array: &ArrayRef,
    stats: &[Stat],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Option<Scalar>>> {
    stats
        .iter()
        .map(|&stat| array.statistics().compute_stat(stat, ctx))
        .collect()
}

fn legacy_stats_builder(stat: Stat, dtype: &DType) -> Box<dyn LegacyStatsArrayBuilder> {
    let allocator = BufferAllocatorRef::static_ref();
    let values = builder_with_capacity_in(dtype, 1024, allocator);
    let is_truncated = BoolBuilder::with_capacity_in(Nullability::NonNullable, 1024, allocator);

    match (stat, dtype) {
        (Stat::Max, DType::Utf8(_)) => Box::new(TruncatedStatsBuilder::<BufferString>::new(
            Stat::Max,
            values,
            is_truncated,
        )),
        (Stat::Max, DType::Binary(_)) => Box::new(TruncatedStatsBuilder::<ByteBuffer>::new(
            Stat::Max,
            values,
            is_truncated,
        )),
        (Stat::Min, DType::Utf8(_)) => Box::new(TruncatedStatsBuilder::<BufferString>::new(
            Stat::Min,
            values,
            is_truncated,
        )),
        (Stat::Min, DType::Binary(_)) => Box::new(TruncatedStatsBuilder::<ByteBuffer>::new(
            Stat::Min,
            values,
            is_truncated,
        )),
        _ => Box::new(PlainStatsBuilder { stat, values }),
    }
}

/// Columns of the zone table for one stat.
struct NamedArrays {
    names: Vec<FieldName>,
    arrays: Vec<ArrayRef>,
}

trait LegacyStatsArrayBuilder: Send {
    fn stat(&self) -> Stat;

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()>;

    fn append_null(&mut self);

    fn finish(&mut self) -> NamedArrays;
}

/// Writes the stat value as is. A `max` or `min` column gets a truncation column that is always
/// false.
struct PlainStatsBuilder {
    stat: Stat,
    values: Box<dyn ArrayBuilder>,
}

impl LegacyStatsArrayBuilder for PlainStatsBuilder {
    fn stat(&self) -> Stat {
        self.stat
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        self.values.append_scalar(&value)
    }

    fn append_null(&mut self) {
        self.values.append_null()
    }

    fn finish(&mut self) -> NamedArrays {
        let array = self.values.finish();
        let len = array.len();

        let truncated_name = match self.stat {
            Stat::Max => Some(MAX_IS_TRUNCATED),
            Stat::Min => Some(MIN_IS_TRUNCATED),
            _ => None,
        };

        match truncated_name {
            Some(truncated_name) => NamedArrays {
                names: vec![self.stat.name().into(), truncated_name.into()],
                arrays: vec![array, ConstantArray::new(false, len).into_array()],
            },
            None => NamedArrays {
                names: vec![self.stat.name().into()],
                arrays: vec![array],
            },
        }
    }
}

/// Writes a string or binary `max` or `min` value cut to [`MAX_VARIABLE_LENGTH_STAT_BYTES`].
///
/// A cut `max` is an upper bound and a cut `min` is a lower bound of the zone values. If no bound
/// is possible, the value is null.
struct TruncatedStatsBuilder<T: ScalarTruncation> {
    stat: Stat,
    values: Box<dyn ArrayBuilder>,
    is_truncated: BoolBuilder,
    _marker: PhantomData<T>,
}

impl<T: ScalarTruncation> TruncatedStatsBuilder<T> {
    fn new(stat: Stat, values: Box<dyn ArrayBuilder>, is_truncated: BoolBuilder) -> Self {
        Self {
            stat,
            values,
            is_truncated,
            _marker: PhantomData,
        }
    }
}

impl<T: ScalarTruncation> LegacyStatsArrayBuilder for TruncatedStatsBuilder<T> {
    fn stat(&self) -> Stat {
        self.stat
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        let nullability = value.dtype().nullability();
        let value = T::from_scalar(value)?;

        let bound = if self.stat == Stat::Max {
            upper_bound(value, MAX_VARIABLE_LENGTH_STAT_BYTES, nullability)
        } else {
            lower_bound(value, MAX_VARIABLE_LENGTH_STAT_BYTES, nullability)
        };

        match bound {
            Some((bound, truncated)) => {
                self.values.append_scalar(&bound)?;
                self.is_truncated.append_value(truncated);
            }
            None => self.append_null(),
        }

        Ok(())
    }

    fn append_null(&mut self) {
        self.values.append_null();
        self.is_truncated.append_value(false);
    }

    fn finish(&mut self) -> NamedArrays {
        let truncated_name = if self.stat == Stat::Max {
            MAX_IS_TRUNCATED
        } else {
            MIN_IS_TRUNCATED
        };

        NamedArrays {
            names: vec![self.stat.name().into(), truncated_name.into()],
            arrays: vec![
                self.values.finish(),
                ArrayBuilder::finish(&mut self.is_truncated),
            ],
        }
    }
}
