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
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::BoolBuilder;
use vortex_array::builders::builder_with_capacity;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldPath;
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
                Stat::Min | Stat::Max | Stat::Sum => {
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
                Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => {}
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

/// Computes the post-order sequence of `(FieldPath, DType)` entries that file-level statistics
/// are stored against.
///
/// Each leaf field (including opaque `List`/`FixedSizeList` columns, which are not recursed into)
/// gets one entry. Each **nullable** struct additionally gets a trailing entry, keyed by its own
/// path and dtype, inserted immediately after its children's entries — this carries the struct's
/// own null count. Dtypes that don't support file stats (see [`supports_file_stats`]), such as
/// [`DType::Variant`], are skipped entirely: no entry is emitted for them or anything beneath
/// them.
///
/// This function is the single source of truth for the number and order of stats entries, used
/// both when accumulating stats at write time and when reconstructing them from the footer at
/// read time, so no paths or dtypes need to be persisted in the flatbuffer itself.
pub fn postorder_stats_layout(dtype: &DType) -> Vec<(FieldPath, DType)> {
    let mut out = Vec::new();
    postorder_stats_layout_into(dtype, FieldPath::root(), &mut out);
    out
}

fn postorder_stats_layout_into(dtype: &DType, path: FieldPath, out: &mut Vec<(FieldPath, DType)>) {
    match dtype.as_struct_fields_opt() {
        Some(struct_fields) => {
            for (name, field_dtype) in struct_fields.names().iter().zip(struct_fields.fields()) {
                postorder_stats_layout_into(&field_dtype, path.clone().push(name.clone()), out);
            }
            if dtype.nullability() == Nullability::Nullable {
                out.push((path, dtype.clone()));
            }
        }
        None if !supports_file_stats(dtype) => {}
        None => out.push((path, dtype.clone())),
    }
}

/// A node in the tree of accumulators mirroring [`postorder_stats_layout`]'s walk of a `DType`.
enum StatsNode {
    /// An opaque leaf: a non-struct dtype, including `List`/`FixedSizeList` (not recursed into).
    Leaf(StatsAccumulator),
    /// A dtype that does not support file stats (e.g. [`DType::Variant`]); contributes no entries.
    Skipped,
    Struct {
        /// One child per struct field, in declaration order.
        children: Vec<(FieldName, StatsNode)>,
        /// Accumulates the struct's own null count. `Some` iff the struct itself is nullable.
        null_count: Option<StatsAccumulator>,
    },
}

impl StatsNode {
    fn build(dtype: &DType, stats: &[Stat], max_variable_length_statistics_size: usize) -> Self {
        match dtype.as_struct_fields_opt() {
            Some(struct_fields) => {
                let children = struct_fields
                    .names()
                    .iter()
                    .zip(struct_fields.fields())
                    .map(|(name, field_dtype)| {
                        (
                            name.clone(),
                            Self::build(&field_dtype, stats, max_variable_length_statistics_size),
                        )
                    })
                    .collect();
                let null_count = (dtype.nullability() == Nullability::Nullable).then(|| {
                    StatsAccumulator::new(dtype, stats, max_variable_length_statistics_size)
                });
                Self::Struct {
                    children,
                    null_count,
                }
            }
            None if !supports_file_stats(dtype) => Self::Skipped,
            None => Self::Leaf(StatsAccumulator::new(
                dtype,
                stats,
                max_variable_length_statistics_size,
            )),
        }
    }

    fn push_chunk(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        match self {
            Self::Skipped => Ok(()),
            Self::Leaf(acc) => acc.push_chunk(array, ctx),
            Self::Struct {
                children,
                null_count,
            } => {
                // The struct's own `ArrayRef` already carries the validity needed to compute its
                // null count, so we push it directly rather than building a synthetic array.
                if let Some(null_count) = null_count {
                    null_count.push_chunk(array, ctx)?;
                }
                let struct_array = array.clone().execute::<StructArray>(ctx)?;
                for ((_, child), field) in children
                    .iter_mut()
                    .zip_eq(struct_array.iter_unmasked_fields())
                {
                    child.push_chunk(field, ctx)?;
                }
                Ok(())
            }
        }
    }

    /// Appends this node's `StatsSet`s, in the same post-order as [`postorder_stats_layout`].
    fn collect_stats_sets(
        &mut self,
        stats: &[Stat],
        ctx: &mut ExecutionCtx,
        out: &mut Vec<StatsSet>,
    ) -> VortexResult<()> {
        match self {
            Self::Skipped => Ok(()),
            Self::Leaf(acc) => {
                out.push(acc.as_stats_set(stats, ctx)?);
                Ok(())
            }
            Self::Struct {
                children,
                null_count,
            } => {
                for (_, child) in children.iter_mut() {
                    child.collect_stats_sets(stats, ctx, out)?;
                }
                if let Some(null_count) = null_count {
                    out.push(null_count.as_stats_set(stats, ctx)?);
                }
                Ok(())
            }
        }
    }
}

/// An array stream processor that computes aggregate statistics for every field, recursing into
/// nested (possibly nullable) structs. See [`postorder_stats_layout`] for the entry ordering.
#[derive(Clone)]
pub struct FileStatsAccumulator {
    stats: Arc<[Stat]>,
    root: Arc<Mutex<StatsNode>>,
    ctx: Arc<Mutex<ExecutionCtx>>,
}

impl FileStatsAccumulator {
    fn new(
        dtype: &DType,
        stats: Arc<[Stat]>,
        max_variable_length_statistics_size: usize,
        session: &VortexSession,
    ) -> Self {
        let root = Arc::new(Mutex::new(StatsNode::build(
            dtype,
            &stats,
            max_variable_length_statistics_size,
        )));

        Self {
            stats,
            root,
            ctx: Arc::new(Mutex::new(session.create_execution_ctx())),
        }
    }

    fn process(
        &self,
        chunk: VortexResult<(SequenceId, ArrayRef)>,
    ) -> VortexResult<(SequenceId, ArrayRef)> {
        let (sequence_id, chunk) = chunk?;
        let mut ctx = self.ctx.lock();
        self.root.lock().push_chunk(&chunk, &mut ctx)?;
        Ok((sequence_id, chunk))
    }

    pub fn stats_sets(&self) -> Vec<StatsSet> {
        let mut ctx = self.ctx.lock();
        let mut out = Vec::new();
        self.root
            .lock()
            .collect_stats_sets(&self.stats, &mut ctx, &mut out)
            .vortex_expect("collect_stats_sets should not fail");
        out
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::bool::BoolArrayExt;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::dtype::FieldNames;
    use vortex_array::scalar::PValue;
    use vortex_array::scalar::ScalarValue;
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
        let mut builder = VarBinViewBuilder::with_capacity(dtype, 2);
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
            &[Stat::Max.name(), Stat::Min.name(), Stat::Sum.name()]
        );
    }

    fn i32_dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    #[test]
    fn postorder_layout_flat_struct() {
        let dtype = DType::struct_(
            [
                ("a", i32_dtype()),
                ("b", DType::Bool(Nullability::Nullable)),
            ],
            Nullability::NonNullable,
        );
        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![
                (FieldPath::from_name("a"), i32_dtype()),
                (
                    FieldPath::from_name("b"),
                    DType::Bool(Nullability::Nullable)
                ),
            ]
        );
    }

    #[test]
    fn postorder_layout_nested_nullable_struct_trails_children() {
        let inner = DType::struct_([("b", i32_dtype())], Nullability::Nullable);
        let dtype = DType::struct_([("a", inner.clone())], Nullability::NonNullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![
                (FieldPath::from_name("a").push("b"), i32_dtype()),
                (FieldPath::from_name("a"), inner),
            ]
        );
    }

    #[test]
    fn postorder_layout_non_nullable_nested_struct_has_no_own_entry() {
        let inner = DType::struct_([("b", i32_dtype())], Nullability::NonNullable);
        let dtype = DType::struct_([("a", inner)], Nullability::NonNullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![(FieldPath::from_name("a").push("b"), i32_dtype())]
        );
    }

    #[test]
    fn postorder_layout_nullable_root_struct_gets_trailing_root_entry() {
        let dtype = DType::struct_([("a", i32_dtype())], Nullability::Nullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![
                (FieldPath::from_name("a"), i32_dtype()),
                (FieldPath::root(), dtype),
            ]
        );
    }

    #[test]
    fn postorder_layout_list_field_is_opaque_leaf() {
        let list_dtype = DType::list(i32_dtype(), Nullability::NonNullable);
        let dtype = DType::struct_([("a", list_dtype.clone())], Nullability::NonNullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(layout, vec![(FieldPath::from_name("a"), list_dtype)]);
    }

    #[test]
    fn postorder_layout_variant_field_is_skipped() {
        let dtype = DType::struct_(
            [
                ("a", i32_dtype()),
                ("v", DType::Variant(Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(layout, vec![(FieldPath::from_name("a"), i32_dtype())]);
    }

    #[test]
    fn nested_nullable_struct_accumulates_its_own_null_count() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let b = buffer![1i32, 2, 3].into_array();
        let inner_validity =
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let inner = StructArray::new(FieldNames::from(["b"]), [b], 3, inner_validity).into_array();
        let outer = StructArray::new(FieldNames::from(["a"]), [inner], 3, Validity::NonNullable)
            .into_array();

        let requested = [Stat::NullCount, Stat::Min, Stat::Max];
        let mut node = StatsNode::build(outer.dtype(), &requested, 1024);
        node.push_chunk(&outer, &mut ctx)?;

        let mut stats_sets = Vec::new();
        node.collect_stats_sets(&requested, &mut ctx, &mut stats_sets)?;

        // `a.b`'s stats come first (post-order), then `a`'s own null-count entry.
        assert_eq!(stats_sets.len(), 2);
        assert_eq!(
            stats_sets[1].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::Primitive(PValue::U64(1)))
        );
        Ok(())
    }
}
