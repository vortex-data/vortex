// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::marker::PhantomData;
use std::sync::Arc;

use futures::StreamExt;
use itertools::Itertools;
use parking_lot::Mutex;
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
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarTruncation;
use vortex_array::scalar::lower_bound;
use vortex_array::scalar::upper_bound;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::layouts::zoned::MAX_IS_TRUNCATED;
use crate::layouts::zoned::MIN_IS_TRUNCATED;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

pub fn accumulate_stats(
    stream: SendableSequentialStream,
    stats: Arc<[Stat]>,
    max_variable_length_statistics_size: usize,
    session: &VortexSession,
) -> (FileStatsAccumulator, SendableSequentialStream) {
    let accumulator = FileStatsAccumulator::new(
        stream.dtype(),
        stats,
        max_variable_length_statistics_size,
        session,
    );
    let stream = SequentialStreamAdapter::new(
        stream.dtype().clone(),
        stream.scan(accumulator.clone(), |acc, item| {
            future::ready(Some(acc.process(item)))
        }),
    )
    .sendable();
    (accumulator, stream)
}

/// Accumulates write-time statistics for a single file column.
struct StatsAccumulator {
    builders: Vec<Box<dyn StatsArrayBuilder>>,
    length: usize,
}

impl StatsAccumulator {
    fn new(dtype: &DType, stats: &[Stat], max_variable_length_statistics_size: usize) -> Self {
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

    fn push_chunk(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
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

    fn as_array(&mut self) -> VortexResult<Option<StructArray>> {
        let mut names = Vec::new();
        let mut fields = Vec::new();

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

            names.extend(values.names);
            fields.extend(values.arrays);
        }

        if names.is_empty() {
            return Ok(None);
        }

        StructArray::try_new(names.into(), fields, self.length, Validity::NonNullable).map(Some)
    }

    /// Returns an aggregated stats set for the table.
    fn as_stats_set(&mut self, stats: &[Stat], ctx: &mut ExecutionCtx) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();
        let Some(array) = self.as_array()? else {
            return Ok(stats_set);
        };

        for &stat in stats {
            let Some(array) = array.unmasked_field_by_name_opt(stat.name()) else {
                continue;
            };

            match stat {
                Stat::Max if is_varlen_dtype(array.dtype()) && !array.all_valid(ctx)? => {
                    // A null truncated varlen max can mean either an empty chunk or no finite
                    // upper bound, so aggregating by skipping nulls would be unsound.
                    continue;
                }
                Stat::Min | Stat::Max | Stat::Sum => {
                    if let Some(s) = array.statistics().compute_stat(stat, ctx)?
                        && let Some(v) = s.into_value()
                    {
                        stats_set.set(stat, Precision::exact(v))
                    }
                }
                Stat::NullCount | Stat::NaNCount | Stat::UncompressedSizeInBytes => {
                    if let Some(sum_value) = sum(array, ctx)?
                        .cast(&DType::Primitive(PType::U64, Nullability::Nullable))?
                        .into_value()
                    {
                        stats_set.set(stat, Precision::exact(sum_value));
                    }
                }
                Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => {}
            }
        }
        Ok(stats_set)
    }
}

fn supports_file_stats(dtype: &DType) -> bool {
    !matches!(dtype, DType::Variant(_))
}

fn is_varlen_dtype(dtype: &DType) -> bool {
    matches!(dtype, DType::Utf8(_) | DType::Binary(_))
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

/// An array stream processor that computes aggregate statistics for all fields.
///
/// Note: for now this only collects top-level struct fields.
#[derive(Clone)]
pub struct FileStatsAccumulator {
    stats: Arc<[Stat]>,
    accumulators: Arc<Mutex<Vec<StatsAccumulator>>>,
    ctx: Arc<Mutex<ExecutionCtx>>,
}

impl FileStatsAccumulator {
    fn new(
        dtype: &DType,
        stats: Arc<[Stat]>,
        max_variable_length_statistics_size: usize,
        session: &VortexSession,
    ) -> Self {
        let accumulators = Arc::new(Mutex::new(match dtype.as_struct_fields_opt() {
            Some(struct_dtype) => {
                if dtype.nullability() == Nullability::Nullable {
                    // top level dtype could be nullable, but we don't support it yet
                    vortex_panic!(
                        "FileStatsAccumulator temporarily does not support nullable top-level structs, got: {}. Use Validity::NonNullable",
                        dtype
                    );
                }

                struct_dtype
                    .fields()
                    .map(|field_dtype| {
                        StatsAccumulator::new(
                            &field_dtype,
                            &stats,
                            max_variable_length_statistics_size,
                        )
                    })
                    .collect()
            }
            None => [StatsAccumulator::new(
                dtype,
                &stats,
                max_variable_length_statistics_size,
            )]
            .into(),
        }));

        Self {
            stats,
            accumulators,
            ctx: Arc::new(Mutex::new(session.create_execution_ctx())),
        }
    }

    fn process(
        &self,
        chunk: VortexResult<(SequenceId, ArrayRef)>,
    ) -> VortexResult<(SequenceId, ArrayRef)> {
        let (sequence_id, chunk) = chunk?;
        let mut ctx = self.ctx.lock();
        if chunk.dtype().is_struct() {
            let struct_chunk = chunk.clone().execute::<StructArray>(&mut ctx)?;
            for (acc, field) in self
                .accumulators
                .lock()
                .iter_mut()
                .zip_eq(struct_chunk.iter_unmasked_fields())
            {
                acc.push_chunk(field, &mut ctx)?;
            }
        } else {
            self.accumulators.lock()[0].push_chunk(&chunk, &mut ctx)?;
        }
        Ok((sequence_id, chunk))
    }

    pub fn stats_sets(&self) -> Vec<StatsSet> {
        let mut ctx = self.ctx.lock();
        self.accumulators
            .lock()
            .iter_mut()
            .map(|acc| {
                acc.as_stats_set(&self.stats, &mut ctx)
                    .vortex_expect("as_stats_table should not fail")
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::bool::BoolArrayExt;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;

    use super::*;

    #[rstest]
    #[case(DType::Utf8(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    fn truncates_accumulated_stats(#[case] dtype: DType) {
        let mut ctx = array_session().create_execution_ctx();
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
        let stats_table = acc.as_array().unwrap().expect("Must have stats table");
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
        let mut ctx = array_session().create_execution_ctx();
        let array = buffer![0, 1, 2].into_array();
        let mut acc = StatsAccumulator::new(array.dtype(), &[Stat::Max, Stat::Min, Stat::Sum], 12);
        acc.push_chunk(&array, &mut ctx)
            .vortex_expect("push_chunk should succeed for test array");
        let stats_table = acc.as_array().unwrap().expect("Must have stats table");
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
