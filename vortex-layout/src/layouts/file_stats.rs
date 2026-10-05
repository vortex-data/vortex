// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Collects chunk statistics and combines them into file statistics.
//!
//! Sums retain their accumulator state so overflow survives across chunks. Other statistics use
//! intermediate columns, including truncation markers for variable-length bounds.

use std::future;
use std::marker::PhantomData;
use std::sync::Arc;

use futures::StreamExt;
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::BoolBuilder;
use vortex_array::builders::builder_with_capacity_in;
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
    /// Retains overflow that a nullable column of finalized chunk sums would discard.
    sum: Option<Accumulator<Sum>>,
    length: usize,
}

impl StatsAccumulator {
    fn new(dtype: &DType, stats: &[Stat], max_variable_length_statistics_size: usize) -> Self {
        if !supports_file_stats(dtype) {
            return Self {
                builders: Vec::new(),
                sum: None,
                length: 0,
            };
        }

        let sum = (stats.contains(&Stat::Sum) && Stat::Sum.dtype(dtype).is_some()).then(|| {
            Accumulator::try_new(Sum, NumericalAggregateOpts::skip_nans(), dtype.clone())
                .vortex_expect("StatsAccumulator::new checked that the input dtype supports Sum")
        });

        let builders = stats
            .iter()
            .filter(|&&stat| stat != Stat::Sum)
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
            sum,
            length: 0,
        }
    }

    fn push_chunk(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        if let Some(sum) = &mut self.sum {
            let mut chunk_sum = Accumulator::try_new(
                Sum,
                NumericalAggregateOpts::skip_nans(),
                array.dtype().clone(),
            )?;
            chunk_sum.accumulate(array, ctx)?;

            let result = chunk_sum.final_scalar()?;
            let is_nan = result
                .as_primitive_opt()
                .is_some_and(|value| value.is_nan());
            if let Some(value) = result.into_value() {
                array.statistics().set(Stat::Sum, Precision::Exact(value));
            }

            // The file writer's NaN-skipping policy also applies to chunk finals.
            if !is_nan {
                sum.merge_from(&mut chunk_sum)?;
            }
        }

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

    fn as_array(&mut self, ctx: &mut ExecutionCtx) -> VortexResult<Option<StructArray>> {
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
            if values.all_invalid(ctx)? {
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

        if self.length != 0
            && stats.contains(&Stat::Sum)
            && let Some(sum) = &self.sum
            && let Some(value) = sum.final_scalar()?.into_value()
        {
            // The legacy stats set cannot store a null, so an overflowed sum remains absent.
            stats_set.set(Stat::Sum, Precision::Exact(value));
        }

        let Some(stats_table) = self.as_array(ctx)? else {
            return Ok(stats_set);
        };

        for &stat in stats {
            let Some(values) = stats_table.unmasked_field_by_name_opt(stat.name()) else {
                continue;
            };

            match stat {
                Stat::Max if is_varlen_dtype(values.dtype()) && !values.all_valid(ctx)? => {
                    // A null truncated varlen max can mean either an empty chunk or no finite
                    // upper bound, so aggregating by skipping nulls would be unsound.
                    continue;
                }
                Stat::Min | Stat::Max => {
                    if let Some(s) = values.statistics().compute_stat(stat, ctx)?
                        && let Some(v) = s.into_value()
                    {
                        let precision = if stat_was_truncated(&stats_table, stat, ctx)? {
                            Precision::inexact(v)
                        } else {
                            Precision::exact(v)
                        };
                        stats_set.set(stat, precision)
                    }
                }
                Stat::NullCount | Stat::NaNCount | Stat::UncompressedSizeInBytes => {
                    if let Some(sum_value) = sum(values, ctx)?
                        .cast(&DType::Primitive(PType::U64, Nullability::Nullable))?
                        .into_value()
                    {
                        stats_set.set(stat, Precision::exact(sum_value));
                    }
                }
                Stat::Sum | Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => {}
            }
        }
        Ok(stats_set)
    }
}

fn stat_was_truncated(
    stats_table: &StructArray,
    stat: Stat,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let field_name = match stat {
        Stat::Min => MIN_IS_TRUNCATED,
        Stat::Max => MAX_IS_TRUNCATED,
        _ => return Ok(false),
    };
    let Some(is_truncated) = stats_table.unmasked_field_by_name_opt(field_name) else {
        return Ok(false);
    };

    Ok(is_truncated
        .statistics()
        .compute_stat(Stat::Max, ctx)?
        .is_some_and(|max| max.as_bool().value() == Some(true)))
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
    let values_builder = builder_with_capacity_in(
        dtype,
        capacity,
        vortex_buffer::BufferAllocatorRef::static_ref(),
    );
    match stat {
        Stat::Max => match dtype {
            DType::Utf8(_) => Box::new(TruncatedMaxBinaryStatsBuilder::<BufferString>::new(
                values_builder,
                BoolBuilder::with_capacity_in(
                    Nullability::NonNullable,
                    capacity,
                    vortex_buffer::BufferAllocatorRef::static_ref(),
                ),
                max_length,
            )),
            DType::Binary(_) => Box::new(TruncatedMaxBinaryStatsBuilder::<ByteBuffer>::new(
                values_builder,
                BoolBuilder::with_capacity_in(
                    Nullability::NonNullable,
                    capacity,
                    vortex_buffer::BufferAllocatorRef::static_ref(),
                ),
                max_length,
            )),
            _ => Box::new(StatNameArrayBuilder::new(stat, values_builder)),
        },
        Stat::Min => match dtype {
            DType::Utf8(_) => Box::new(TruncatedMinBinaryStatsBuilder::<BufferString>::new(
                values_builder,
                BoolBuilder::with_capacity_in(
                    Nullability::NonNullable,
                    capacity,
                    vortex_buffer::BufferAllocatorRef::static_ref(),
                ),
                max_length,
            )),
            DType::Binary(_) => Box::new(TruncatedMinBinaryStatsBuilder::<ByteBuffer>::new(
                values_builder,
                BoolBuilder::with_capacity_in(
                    Nullability::NonNullable,
                    capacity,
                    vortex_buffer::BufferAllocatorRef::static_ref(),
                ),
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
    fn all_invalid(&self, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        self.arrays[0].all_invalid(ctx)
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
        NamedArrays {
            names: vec![self.stat.name().into()],
            arrays: vec![self.builder.finish()],
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
    use vortex_array::IntoArray;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    #[cfg(target_pointer_width = "64")]
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::bool::BoolArrayExt;
    use vortex_array::builders::VarBinViewBuilder;
    #[cfg(target_pointer_width = "64")]
    use vortex_array::dtype::DecimalDType;
    #[cfg(target_pointer_width = "64")]
    use vortex_array::expr::stats::StatsProvider;
    #[cfg(target_pointer_width = "64")]
    use vortex_array::scalar::DecimalValue;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;

    use super::*;

    fn accumulate_sum_stats(dtype: &DType, chunks: &[ArrayRef]) -> VortexResult<StatsSet> {
        let mut ctx = array_session().create_execution_ctx();
        let mut acc = StatsAccumulator::new(dtype, &[Stat::Sum], 64);

        for chunk in chunks {
            acc.push_chunk(chunk, &mut ctx)?;
        }

        acc.as_stats_set(&[Stat::Sum], &mut ctx)
    }

    #[rstest]
    #[case(DType::Utf8(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    fn truncates_accumulated_stats(#[case] dtype: DType) {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            dtype.clone(),
            2,
            vortex_buffer::BufferAllocatorRef::statically_allocated(),
        );
        builder.append_value("Value to be truncated");
        builder.append_value("untruncated");
        let mut builder2 = VarBinViewBuilder::with_capacity_in(
            dtype,
            2,
            vortex_buffer::BufferAllocatorRef::statically_allocated(),
        );
        builder2.append_value("Another");
        builder2.append_value("wait a minute");
        let mut acc =
            StatsAccumulator::new(builder.dtype(), &[Stat::Max, Stat::Min, Stat::Sum], 12);
        acc.push_chunk(&builder.finish(), &mut ctx)
            .vortex_expect("push_chunk should succeed for test data");
        acc.push_chunk(&builder2.finish(), &mut ctx)
            .vortex_expect("push_chunk should succeed for test data");
        let stats_table = acc
            .as_array(&mut ctx)
            .unwrap()
            .expect("Must have stats table");
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

    #[rstest]
    #[case(DType::Utf8(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    fn truncated_accumulated_stats_are_inexact(#[case] dtype: DType) {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            dtype,
            2,
            vortex_buffer::BufferAllocatorRef::statically_allocated(),
        );
        builder.append_value("Value to be truncated");
        builder.append_value("Another truncated value");
        let mut acc = StatsAccumulator::new(builder.dtype(), &[Stat::Max, Stat::Min], 12);
        acc.push_chunk(&builder.finish(), &mut ctx)
            .vortex_expect("push_chunk should succeed for test data");

        let stats = acc
            .as_stats_set(&[Stat::Max, Stat::Min], &mut ctx)
            .vortex_expect("as_stats_set should succeed for test data");

        assert!(matches!(stats.get(Stat::Min), Precision::Inexact(_)));
        assert!(matches!(stats.get(Stat::Max), Precision::Inexact(_)));
    }

    #[test]
    fn fixed_width_stats_omit_is_truncated_columns() {
        let mut ctx = array_session().create_execution_ctx();
        let array = buffer![0, 1, 2].into_array();
        let mut acc = StatsAccumulator::new(array.dtype(), &[Stat::Max, Stat::Min, Stat::Sum], 12);
        acc.push_chunk(&array, &mut ctx)
            .vortex_expect("push_chunk should succeed for test array");
        let stats_table = acc
            .as_array(&mut ctx)
            .unwrap()
            .expect("Must have stats table");
        assert_eq!(
            stats_table.names().as_ref(),
            &[Stat::Max.name(), Stat::Min.name()]
        );
    }

    #[rstest]
    #[case::overflow_across_chunks(vec![vec![Some(i64::MAX)], vec![Some(1)]], None)]
    #[case::nullable_chunks(vec![vec![None, Some(3)], vec![Some(4), None]], Some(7))]
    #[case::no_chunks(vec![], None)]
    #[case::empty_chunk(vec![vec![]], Some(0))]
    fn sum_combines_chunk_states(
        #[case] chunks: Vec<Vec<Option<i64>>>,
        #[case] expected: Option<i64>,
    ) -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
        let chunks = chunks
            .into_iter()
            .map(|values| PrimitiveArray::from_option_iter(values).into_array())
            .collect::<Vec<_>>();

        let stats = accumulate_sum_stats(&dtype, &chunks)?;
        assert_eq!(
            stats.get(Stat::Sum),
            expected.map_or(Precision::Absent, Precision::exact)
        );

        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn sum_keeps_decimal_return_precision() -> VortexResult<()> {
        let value = Scalar::decimal(
            DecimalValue::I128(9),
            DecimalDType::new(1, 0),
            Nullability::NonNullable,
        );
        let chunks = [
            ConstantArray::new(value.clone(), 10_000_000_000).into_array(),
            ConstantArray::new(value, 10_000_000_000).into_array(),
        ];

        let stats = accumulate_sum_stats(chunks[0].dtype(), &chunks)?;
        assert_eq!(stats.get(Stat::Sum), Precision::Absent);

        for chunk in &chunks {
            assert_eq!(
                chunk.statistics().get(Stat::Sum),
                Precision::Exact(Scalar::decimal(
                    DecimalValue::I128(90_000_000_000),
                    DecimalDType::new(11, 0),
                    Nullability::Nullable,
                ))
            );
        }

        Ok(())
    }

    #[rstest]
    #[case::nan_chunk_then_finite(vec![f64::INFINITY, f64::NEG_INFINITY], vec![1.0], 1.0)]
    #[case::all_nan_chunks(
        vec![f64::INFINITY, f64::NEG_INFINITY],
        vec![f64::INFINITY, f64::NEG_INFINITY],
        0.0
    )]
    #[case::nan_from_merge(vec![f64::INFINITY], vec![f64::NEG_INFINITY], f64::NAN)]
    #[case::chunk_grouping(vec![1e16], vec![-1e16, 1.0], 0.0)]
    fn sum_preserves_float_chunk_semantics(
        #[case] first: Vec<f64>,
        #[case] second: Vec<f64>,
        #[case] expected: f64,
    ) -> VortexResult<()> {
        let chunks = [
            PrimitiveArray::from_iter(first).into_array(),
            PrimitiveArray::from_iter(second).into_array(),
        ];

        let stats = accumulate_sum_stats(chunks[0].dtype(), &chunks)?;
        if expected.is_nan() {
            assert!(
                stats
                    .get(Stat::Sum)
                    .as_exact()
                    .is_some_and(|value| value.as_primitive().is_nan())
            );
        } else {
            assert_eq!(stats.get(Stat::Sum), Precision::exact(expected));
        }

        Ok(())
    }
}
