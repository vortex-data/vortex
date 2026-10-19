// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`LayoutReader`] decorator that performs file-level stats pruning.
//!
//! If file-level statistics prove that a filter expression cannot match any rows in the file,
//! [`FileStatsLayoutReader`] short-circuits [`pruning_evaluation`](LayoutReader::pruning_evaluation)
//! by returning an all-false mask — avoiding all downstream I/O.

use std::iter;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::MaskFuture;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_layout::ArrayFuture;
use vortex_layout::LayoutReader;
use vortex_layout::LayoutReaderRef;
use vortex_layout::RowSplits;
use vortex_layout::SplitRange;
use vortex_layout::layouts::file_stats::AggregateStat;
use vortex_layout::layouts::file_stats::AggregateStats;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;

use crate::FileStatistics;
use crate::pruning::can_prune_file_stats;
use crate::pruning::is_satisfied_by_file_stats;

/// A [`LayoutReader`] decorator that prunes entire files based on file-level statistics.
///
/// This reader wraps an inner `LayoutReader` and intercepts `pruning_evaluation` calls.
/// When file-level stats prove that a filter expression is false for the entire file,
/// it returns an all-false mask immediately — avoiding all downstream I/O.
///
/// Pruning results are cached per-expression since file-level stats are global
/// (the result is the same regardless of which row range is requested).
pub struct FileStatsLayoutReader {
    child: LayoutReaderRef,
    file_stats: FileStatistics,
    session: VortexSession,
    prune_cache: DashMap<ExactBoundExpr, bool>,
}

impl FileStatsLayoutReader {
    /// Creates a new `FileStatsLayoutReader` wrapping the given child reader.
    pub fn new(child: LayoutReaderRef, file_stats: FileStatistics, session: VortexSession) -> Self {
        Self {
            child,
            file_stats,
            session,
            prune_cache: DashMap::with_hasher_and_shard_amount(Default::default(), 8),
        }
    }

    /// Evaluates whether file-level statistics prove `expr` cannot match.
    ///
    /// Row-count placeholders are resolved against the full file row count,
    /// independent of the requested row range.
    fn evaluate_file_stats(&self, expr: &BoundExpression) -> VortexResult<bool> {
        can_prune_file_stats(
            expr,
            self.child.row_count(),
            &self.file_stats,
            &self.session,
        )
    }

    /// Returns `true` if file-level statistics prove that `expr` matches no row of the file.
    ///
    /// Unlike [`LayoutReader::pruning_evaluation`], this never consults the statistics of the
    /// file's zones, so it costs the same however many rows the file holds.
    pub fn prunes_file(&self, expr: &BoundExpression) -> VortexResult<bool> {
        let key = ExactBoundExpr(expr.clone());

        // Check cache first with read-only lock.
        if let Some(pruned) = self.prune_cache.get(&key).map(|pruned| *pruned) {
            return Ok(pruned);
        }
        // Evaluate under the entry lock, so that splits which start together evaluate the
        // expression once rather than each falsifying it again.
        match self.prune_cache.entry(key) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => Ok(*entry.insert(self.evaluate_file_stats(expr)?)),
        }
    }

    /// Returns the file-level statistics used by this reader.
    pub fn file_stats(&self) -> &FileStatistics {
        &self.file_stats
    }

    /// Returns `true` if file-level statistics prove `expr` true for every row of the file, so
    /// that a filter on it keeps every row and need not be evaluated.
    pub fn is_satisfied(&self, expr: &BoundExpression) -> VortexResult<bool> {
        is_satisfied_by_file_stats(
            expr,
            self.child.row_count(),
            &self.file_stats,
            &self.session,
        )
    }

    /// Returns `true` if file-level statistics prove `expr` true for every row whose columns are
    /// not null, which suffices for a filter that may keep rows it does not match.
    pub fn is_satisfied_by_non_null(&self, expr: &BoundExpression) -> VortexResult<bool> {
        let aggregates: Arc<[AggregateStats]> = self
            .file_stats
            .aggregates()
            .iter()
            .map(|stats| {
                let no_nulls = AggregateStat::from_value(
                    NullCount.bind(EmptyOptions),
                    Precision::exact(Scalar::from(0u64)),
                );
                AggregateStats::new(
                    stats
                        .iter()
                        .filter(|stat| !stat.aggregate_fn().is::<NullCount>())
                        .cloned()
                        .chain(iter::once(no_nulls))
                        .collect(),
                )
            })
            .collect();
        let file_stats = FileStatistics::new(
            aggregates,
            Arc::clone(self.file_stats.dtypes()),
            Arc::clone(self.file_stats.paths()),
        );
        is_satisfied_by_file_stats(expr, self.child.row_count(), &file_stats, &self.session)
    }
}

impl LayoutReader for FileStatsLayoutReader {
    fn name(&self) -> &Arc<str> {
        self.child.name()
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn row_count(&self) -> u64 {
        self.child.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        self.child.register_splits(field_mask, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        if self.prunes_file(expr)? {
            Ok(MaskFuture::ready(Mask::new_false(mask.len())))
        } else {
            self.child.pruning_evaluation(row_range, expr, mask)
        }
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.child.filter_evaluation(row_range, expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        self.child.projection_evaluation(row_range, expr, mask)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray as _;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::null_count::NullCount;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::datetime::TemporalData;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldPath;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::checked_add;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Precision;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_layout::LayoutReader;
    use vortex_layout::LayoutStrategy;
    use vortex_layout::layouts::file_stats::AggregateStat;
    use vortex_layout::layouts::file_stats::AggregateStats;
    use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
    use vortex_layout::layouts::table::TableStrategy;
    use vortex_layout::segments::SegmentSink;
    use vortex_layout::segments::TestSegments;
    use vortex_layout::sequence::SequenceId;
    use vortex_layout::sequence::SequentialArrayStreamExt;
    use vortex_layout::session::LayoutSession;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
    });

    fn test_file_stats(min: i32, max: i32) -> FileStatistics {
        let stats = AggregateStats::new(vec![
            AggregateStat::from_value(
                Min.bind(NumericalAggregateOpts::skip_nans()),
                Precision::exact(Scalar::from(min)),
            ),
            AggregateStat::from_value(
                Max.bind(NumericalAggregateOpts::skip_nans()),
                Precision::exact(Scalar::from(max)),
            ),
        ]);
        FileStatistics::new(
            Arc::from([stats]),
            Arc::from([DType::Primitive(PType::I32, Nullability::NonNullable)]),
            Arc::from([FieldPath::from_name("col")]),
        )
    }

    fn test_file_null_count_stats(null_count: u64) -> FileStatistics {
        let stats = AggregateStats::new(vec![AggregateStat::from_value(
            NullCount.bind(EmptyOptions),
            Precision::exact(Scalar::from(null_count)),
        )]);
        FileStatistics::new(
            Arc::from([stats]),
            Arc::from([DType::Primitive(PType::I32, Nullability::Nullable)]),
            Arc::from([FieldPath::from_name("col")]),
        )
    }

    #[test]
    fn satisfied_by_non_null_ignores_nulls() -> VortexResult<()> {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let values = PrimitiveArray::from_option_iter([Some(1i32), None, Some(5)]);
            let struct_array = StructArray::from_fields([("col", values.into_array())].as_slice())?;
            let strategy = TableStrategy::new(
                Arc::new(FlatLayoutStrategy::default()),
                Arc::new(FlatLayoutStrategy::default()),
            );
            let layout = strategy
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    struct_array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;
            let child = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;

            let stats = AggregateStats::new(vec![
                AggregateStat::from_value(
                    Min.bind(NumericalAggregateOpts::skip_nans()),
                    Precision::exact(Scalar::from(1i32)),
                ),
                AggregateStat::from_value(
                    Max.bind(NumericalAggregateOpts::skip_nans()),
                    Precision::exact(Scalar::from(5i32)),
                ),
                AggregateStat::from_value(
                    NullCount.bind(EmptyOptions),
                    Precision::exact(Scalar::from(1u64)),
                ),
            ]);
            let file_stats = FileStatistics::new(
                Arc::from([stats]),
                Arc::from([DType::Primitive(PType::I32, Nullability::Nullable)]),
                Arc::from([FieldPath::from_name("col")]),
            );
            let reader = FileStatsLayoutReader::new(child, file_stats, SESSION.clone());

            // The null row does not match, but every other row does.
            let expr = gt(get_item("col", root()), lit(0i32)).bind(reader.dtype())?;
            assert!(!reader.is_satisfied(&expr)?);
            assert!(reader.is_satisfied_by_non_null(&expr)?);

            let expr = gt(get_item("col", root()), lit(1i32)).bind(reader.dtype())?;
            assert!(!reader.is_satisfied_by_non_null(&expr)?);
            Ok(())
        })
    }

    #[test]
    fn pruning_when_filter_out_of_range() -> VortexResult<()> {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let struct_array = StructArray::from_fields(
                [("col", buffer![1i32, 2, 3, 4, 5].into_array())].as_slice(),
            )?;
            let strategy = TableStrategy::new(
                Arc::new(FlatLayoutStrategy::default()),
                Arc::new(FlatLayoutStrategy::default()),
            );
            let layout = strategy
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    struct_array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let child = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;

            let reader =
                FileStatsLayoutReader::new(child, test_file_stats(0, 100), SESSION.clone());

            // col > 200 should be prunable since max is 100.
            let expr = gt(get_item("col", root()), lit(200i32)).bind(reader.dtype())?;
            let mask = Mask::new_true(5);
            let result = reader.pruning_evaluation(&(0..5), &expr, mask)?.await?;
            assert_eq!(result, Mask::new_false(5));

            Ok(())
        })
    }

    #[test]
    fn no_pruning_when_filter_in_range() -> VortexResult<()> {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let struct_array = StructArray::from_fields(
                [("col", buffer![1i32, 2, 3, 4, 5].into_array())].as_slice(),
            )?;
            let strategy = TableStrategy::new(
                Arc::new(FlatLayoutStrategy::default()),
                Arc::new(FlatLayoutStrategy::default()),
            );
            let layout = strategy
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    struct_array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let child = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;

            let reader =
                FileStatsLayoutReader::new(child, test_file_stats(0, 100), SESSION.clone());

            // col > 50 should NOT be prunable since max is 100 (some rows could match).
            let expr = gt(get_item("col", root()), lit(50i32)).bind(reader.dtype())?;
            let mask = Mask::new_true(5);
            let result = reader.pruning_evaluation(&(0..5), &expr, mask)?.await?;
            // Should delegate to child, which returns the mask unchanged (struct reader doesn't prune).
            assert_eq!(result, Mask::new_true(5));

            Ok(())
        })
    }

    #[test]
    fn no_pruning_for_computed_expression_stats() -> VortexResult<()> {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let struct_array =
                StructArray::from_fields([("col", buffer![0i32, 100].into_array())].as_slice())?;
            let strategy = TableStrategy::new(
                Arc::new(FlatLayoutStrategy::default()),
                Arc::new(FlatLayoutStrategy::default()),
            );
            let layout = strategy
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    struct_array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let child = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
            let reader =
                FileStatsLayoutReader::new(child, test_file_stats(0, 100), SESSION.clone());

            let expr = gt(checked_add(get_item("col", root()), lit(5i32)), lit(102i32))
                .bind(reader.dtype())?;
            let mask = Mask::new_true(2);
            let result = reader.pruning_evaluation(&(0..2), &expr, mask)?.await?;

            assert_eq!(result, Mask::new_true(2));

            Ok(())
        })
    }

    /// Regression test: `IS NULL` on a nullable timestamp column must not fail with a
    /// dtype mismatch. The bug was that `stats_ref` used the *field* dtype (timestamp)
    /// for the `NullCount` stat scalar instead of the stat's own dtype (u64).
    #[test]
    fn is_null_pruning_on_nullable_timestamp_column() -> VortexResult<()> {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();

            // Build a struct with a nullable timestamp column containing some nulls.
            let prim_array =
                PrimitiveArray::from_option_iter([Some(1_000_000i64), None, Some(3_000_000)])
                    .into_array();
            let ts_data = TemporalData::new_timestamp(prim_array, TimeUnit::Microseconds, None);
            let ts_dtype = ts_data.dtype().clone();
            let ts_array = ts_data.into_array();

            let struct_array = StructArray::from_fields([("deleted_at", ts_array)].as_slice())?;

            let strategy = TableStrategy::new(
                Arc::new(FlatLayoutStrategy::default()),
                Arc::new(FlatLayoutStrategy::default()),
            );
            let layout = strategy
                .write_stream(
                    ctx.into(),
                    Arc::clone(&segments) as Arc<dyn SegmentSink>,
                    struct_array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let child = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;

            // File-level stats: 1 null in deleted_at.
            let stats = AggregateStats::new(vec![AggregateStat::from_value(
                NullCount.bind(EmptyOptions),
                Precision::exact(Scalar::from(1u64)),
            )]);
            let file_stats = FileStatistics::new(
                Arc::from([stats]),
                Arc::from([ts_dtype]),
                Arc::from([FieldPath::from_name("deleted_at")]),
            );

            let reader = FileStatsLayoutReader::new(child, file_stats, SESSION.clone());

            // `is_null(deleted_at)` — should NOT panic or error due to dtype mismatch.
            let expr = is_null(get_item("deleted_at", root())).bind(reader.dtype())?;
            let mask = Mask::new_true(3);
            let result = reader.pruning_evaluation(&(0..3), &expr, mask)?.await?;
            // null_count is 1 (non-zero), so is_null is not falsified => not pruned.
            assert_eq!(result, Mask::new_true(3));

            Ok(())
        })
    }

    #[test]
    fn pruning_is_not_null_when_file_is_all_null() -> VortexResult<()> {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let struct_array = StructArray::from_fields(
                [(
                    "col",
                    PrimitiveArray::from_option_iter([None::<i32>, None, None, None, None])
                        .into_array(),
                )]
                .as_slice(),
            )?;
            let strategy = TableStrategy::new(
                Arc::new(FlatLayoutStrategy::default()),
                Arc::new(FlatLayoutStrategy::default()),
            );
            let layout = strategy
                .write_stream(
                    ctx.into(),
                    Arc::clone(&segments) as Arc<dyn SegmentSink>,
                    struct_array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let child = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;

            let reader =
                FileStatsLayoutReader::new(child, test_file_null_count_stats(5), SESSION.clone());

            let expr = is_not_null(get_item("col", root())).bind(reader.dtype())?;
            let mask = Mask::new_true(5);
            let result = reader.pruning_evaluation(&(0..5), &expr, mask)?.await?;
            assert_eq!(result, Mask::new_false(5));

            Ok(())
        })
    }
}
