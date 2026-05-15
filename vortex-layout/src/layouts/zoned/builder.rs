//! Write-time accumulation and builders for zoned layout stats tables.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::BoolBuilder;
use vortex_array::builders::builder_with_capacity;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarTruncation;
use vortex_array::scalar::lower_bound;
use vortex_array::scalar::upper_bound;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::layouts::zoned::schema::MAX_IS_TRUNCATED;
use crate::layouts::zoned::schema::MIN_IS_TRUNCATED;

/// Accumulates write-time statistics for each logical zone.
pub struct StatsAccumulator {
    builders: Vec<Box<dyn StatsArrayBuilder>>,
    length: usize,
}

impl StatsAccumulator {
    pub fn new(dtype: &DType, stats: &[Stat], max_variable_length_statistics_size: usize) -> Self {
        if !supports_file_stats(dtype) {
            return Self {
                builders: Vec::new(),
                length: 0,
            };
        }

        let builders = stats
            .iter()
            .filter_map(|&stat| {
                stat.dtype(dtype).map(|stat_dtype| {
                    stats_builder_with_capacity(
                        stat,
                        &stat_dtype.as_nullable(),
                        1024,
                        max_variable_length_statistics_size,
                    )
                })
            })
            .collect::<Vec<_>>();

        Self {
            builders,
            length: 0,
        }
    }

    pub fn push_chunk_without_compute(&mut self, array: &ArrayRef) -> VortexResult<()> {
        for builder in &mut self.builders {
            if let Some(Precision::Exact(value)) = array.statistics().get(builder.stat()) {
                builder.append_scalar(value.cast(&value.dtype().as_nullable())?)?;
            } else {
                builder.append_null();
            }
        }
        self.length += 1;
        Ok(())
    }

    pub fn push_chunk(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        for builder in &mut self.builders {
            if let Some(value) = array.statistics().compute_stat(builder.stat(), ctx)? {
                builder.append_scalar(value.cast(&value.dtype().as_nullable())?)?;
            } else {
                builder.append_null();
            }
        }
        self.length += 1;
        Ok(())
    }

    pub fn as_array(&mut self) -> VortexResult<Option<(StructArray, Arc<[Stat]>)>> {
        let mut names = Vec::new();
        let mut fields = Vec::new();
        let mut stats = Vec::new();

        for builder in self
            .builders
            .iter_mut()
            // We sort the stats so the DType is deterministic based on which stats are present.
            .sorted_unstable_by_key(|builder| builder.stat())
        {
            let values = builder.finish();

            // We drop any all-null stats columns.
            if values.all_invalid()? {
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

    /// Returns an aggregated stats set for the table.
    pub fn as_stats_set(
        &mut self,
        stats: &[Stat],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();
        let Some((array, _)) = self.as_array()? else {
            return Ok(stats_set);
        };

        for &stat in stats {
            let Some(array) = array.unmasked_field_by_name_opt(stat.name()) else {
                continue;
            };

            // Different stats need different aggregations
            match stat {
                // For stats that are associative, we can just compute them over the stat column
                Stat::Min | Stat::Max | Stat::Sum => {
                    if let Some(s) = array.statistics().compute_stat(stat, ctx)?
                        && let Some(v) = s.into_value()
                    {
                        stats_set.set(stat, Precision::exact(v))
                    }
                }
                // These stats sum up
                Stat::NullCount | Stat::NaNCount | Stat::UncompressedSizeInBytes => {
                    if let Some(sum_value) = sum(array, ctx)?
                        .cast(&DType::Primitive(PType::U64, Nullability::Nullable))?
                        .into_value()
                    {
                        stats_set.set(stat, Precision::exact(sum_value));
                    }
                }
                // We could implement these aggregations in the future, but for now they're unused
                Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => {}
            }
        }
        Ok(stats_set)
    }
}

fn supports_file_stats(dtype: &DType) -> bool {
    !matches!(dtype, DType::Variant(_))
}

fn stats_builder_with_capacity(
    stat: Stat,
    dtype: &DType,
    capacity: usize,
    max_length: usize,
) -> Box<dyn StatsArrayBuilder> {
    let values_builder = builder_with_capacity(dtype, capacity);
    match stat {
        Stat::Max => match dtype {
            DType::Utf8(_) => Box::new(TruncatedMaxBinaryStatsBuilder::<BufferString>::new(
                values_builder,
                BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
                max_length,
            )),
            DType::Binary(_) => Box::new(TruncatedMaxBinaryStatsBuilder::<ByteBuffer>::new(
                values_builder,
                BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
                max_length,
            )),
            _ => Box::new(StatNameArrayBuilder::new(stat, values_builder)),
        },
        Stat::Min => match dtype {
            DType::Utf8(_) => Box::new(TruncatedMinBinaryStatsBuilder::<BufferString>::new(
                values_builder,
                BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
                max_length,
            )),
            DType::Binary(_) => Box::new(TruncatedMinBinaryStatsBuilder::<ByteBuffer>::new(
                values_builder,
                BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
                max_length,
            )),
            _ => Box::new(StatNameArrayBuilder::new(stat, values_builder)),
        },
        _ => Box::new(StatNameArrayBuilder::new(stat, values_builder)),
    }
}

/// Arrays with their associated names, reduced version of a `StructArray`.
struct NamedArrays {
    names: Vec<FieldName>,
    arrays: Vec<ArrayRef>,
}

impl NamedArrays {
    fn all_invalid(&self) -> VortexResult<bool> {
        // By convention the first array is the logical validity signal for the stat column.
        self.arrays[0].all_invalid(&mut LEGACY_SESSION.create_execution_ctx())
    }
}

trait StatsArrayBuilder: Send {
    fn stat(&self) -> Stat;

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()>;

    fn append_null(&mut self);

    fn finish(&mut self) -> NamedArrays;
}

struct StatNameArrayBuilder {
    stat: Stat,
    builder: Box<dyn ArrayBuilder>,
}

impl StatNameArrayBuilder {
    fn new(stat: Stat, builder: Box<dyn ArrayBuilder>) -> Self {
        Self { stat, builder }
    }
}

impl StatsArrayBuilder for StatNameArrayBuilder {
    fn stat(&self) -> Stat {
        self.stat
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        self.builder.append_scalar(&value)
    }

    fn append_null(&mut self) {
        self.builder.append_null()
    }

    fn finish(&mut self) -> NamedArrays {
        let array = self.builder.finish();
        let len = array.len();
        match self.stat {
            Stat::Max => NamedArrays {
                names: vec![self.stat.name().into(), MAX_IS_TRUNCATED.into()],
                arrays: vec![array, ConstantArray::new(false, len).into_array()],
            },
            Stat::Min => NamedArrays {
                names: vec![self.stat.name().into(), MIN_IS_TRUNCATED.into()],
                arrays: vec![array, ConstantArray::new(false, len).into_array()],
            },
            _ => NamedArrays {
                names: vec![self.stat.name().into()],
                arrays: vec![array],
            },
        }
    }
}

struct TruncatedMaxBinaryStatsBuilder<T: ScalarTruncation> {
    values: Box<dyn ArrayBuilder>,
    is_truncated: BoolBuilder,
    max_value_length: usize,
    _marker: PhantomData<T>,
}

impl<T: ScalarTruncation> TruncatedMaxBinaryStatsBuilder<T> {
    fn new(
        values: Box<dyn ArrayBuilder>,
        is_truncated: BoolBuilder,
        max_value_length: usize,
    ) -> Self {
        Self {
            values,
            is_truncated,
            max_value_length,
            _marker: PhantomData,
        }
    }
}

struct TruncatedMinBinaryStatsBuilder<T: ScalarTruncation> {
    values: Box<dyn ArrayBuilder>,
    is_truncated: BoolBuilder,
    max_value_length: usize,
    _marker: PhantomData<T>,
}

impl<T: ScalarTruncation> TruncatedMinBinaryStatsBuilder<T> {
    fn new(
        values: Box<dyn ArrayBuilder>,
        is_truncated: BoolBuilder,
        max_value_length: usize,
    ) -> Self {
        Self {
            values,
            is_truncated,
            max_value_length,
            _marker: PhantomData,
        }
    }
}

impl<T: ScalarTruncation> StatsArrayBuilder for TruncatedMaxBinaryStatsBuilder<T> {
    fn stat(&self) -> Stat {
        Stat::Max
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        let nullability = value.dtype().nullability();
        if let Some((upper_bound, truncated)) =
            upper_bound(T::from_scalar(value)?, self.max_value_length, nullability)
        {
            self.values.append_scalar(&upper_bound)?;
            self.is_truncated.append_value(truncated);
        } else {
            self.append_null()
        }
        Ok(())
    }

    fn append_null(&mut self) {
        ArrayBuilder::append_null(self.values.as_mut());
        self.is_truncated.append_value(false);
    }

    fn finish(&mut self) -> NamedArrays {
        NamedArrays {
            names: vec![Stat::Max.name().into(), MAX_IS_TRUNCATED.into()],
            arrays: vec![
                ArrayBuilder::finish(self.values.as_mut()),
                ArrayBuilder::finish(&mut self.is_truncated),
            ],
        }
    }
}

impl<T: ScalarTruncation> StatsArrayBuilder for TruncatedMinBinaryStatsBuilder<T> {
    fn stat(&self) -> Stat {
        Stat::Min
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        let nullability = value.dtype().nullability();
        if let Some((lower_bound, truncated)) =
            lower_bound(T::from_scalar(value)?, self.max_value_length, nullability)
        {
            self.values.append_scalar(&lower_bound)?;
            self.is_truncated.append_value(truncated);
        } else {
            self.append_null()
        }
        Ok(())
    }

    fn append_null(&mut self) {
        ArrayBuilder::append_null(self.values.as_mut());
        self.is_truncated.append_value(false);
    }

    fn finish(&mut self) -> NamedArrays {
        NamedArrays {
            names: vec![Stat::Min.name().into(), MIN_IS_TRUNCATED.into()],
            arrays: vec![
                ArrayBuilder::finish(self.values.as_mut()),
                ArrayBuilder::finish(&mut self.is_truncated),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::bool::BoolArrayExt;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::stats::Stat;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use super::*;

    #[rstest]
    #[case(DType::Utf8(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    fn truncates_accumulated_stats(#[case] dtype: DType) {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut builder = VarBinViewBuilder::with_capacity(dtype.clone(), 2);
        builder.append_value("Value to be truncated");
        builder.append_value("untruncated");
        let mut builder2 = VarBinViewBuilder::with_capacity(dtype, 2);
        builder2.append_value("Another");
        builder2.append_value("wait a minute");
        let mut acc =
            StatsAccumulator::new(builder.dtype(), &[Stat::Max, Stat::Min, Stat::Sum], 12);
        acc.push_chunk(&builder.finish(), &mut ctx)
            .vortex_expect("push_chunk should succeed for test data");
        acc.push_chunk(&builder2.finish(), &mut ctx)
            .vortex_expect("push_chunk should succeed for test data");
        let (stats_table, _) = acc.as_array().unwrap().expect("Must have stats table");
        assert_eq!(
            stats_table.names().as_ref(),
            &[
                Stat::Max.name(),
                MAX_IS_TRUNCATED,
                Stat::Min.name(),
                MIN_IS_TRUNCATED,
            ]
        );
        let field1_bool = stats_table
            .unmasked_field(1)
            .clone()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_eq!(
            field1_bool.to_bit_buffer(),
            BitBuffer::from(vec![false, true])
        );
        let field3_bool = stats_table
            .unmasked_field(3)
            .clone()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_eq!(
            field3_bool.to_bit_buffer(),
            BitBuffer::from(vec![true, false])
        );
    }

    #[test]
    fn always_adds_is_truncated_column() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = buffer![0, 1, 2].into_array();
        let mut acc = StatsAccumulator::new(array.dtype(), &[Stat::Max, Stat::Min, Stat::Sum], 12);
        acc.push_chunk(&array, &mut ctx)
            .vortex_expect("push_chunk should succeed for test array");
        let (stats_table, _) = acc.as_array().unwrap().expect("Must have stats table");
        assert_eq!(
            stats_table.names().as_ref(),
            &[
                Stat::Max.name(),
                MAX_IS_TRUNCATED,
                Stat::Min.name(),
                MIN_IS_TRUNCATED,
                Stat::Sum.name(),
            ]
        );
        let field1_bool = stats_table
            .unmasked_field(1)
            .clone()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_eq!(field1_bool.to_bit_buffer(), BitBuffer::from(vec![false]));
        let field3_bool = stats_table
            .unmasked_field(3)
            .clone()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_eq!(field3_bool.to_bit_buffer(), BitBuffer::from(vec![false]));
    }
}
