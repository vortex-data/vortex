// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`LayoutReader`] decorator that performs file-level stats pruning.
//!
//! If file-level statistics prove that a filter expression cannot match any rows in the file,
//! [`FileStatsLayoutReader`] short-circuits [`pruning_evaluation`](LayoutReader::pruning_evaluation)
//! by returning an all-false mask — avoiding all downstream I/O.

use std::ops::Range;
use std::sync::Arc;

use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_error::VortexResult;
use vortex_layout::ArrayFuture;
use vortex_layout::LayoutReader;
use vortex_layout::LayoutReaderRef;
use vortex_layout::RowSplits;
use vortex_layout::SplitRange;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use crate::FileStatistics;
use crate::pruning::is_falsified_by_file_stats;
use crate::pruning::is_satisfied_by_file_stats;

/// A [`LayoutReader`] decorator that prunes entire files based on file-level statistics.
///
/// This reader wraps an inner `LayoutReader` and intercepts `pruning_evaluation` calls.
/// When file-level stats prove that a filter expression is false for the entire file,
/// it returns an all-false mask immediately — avoiding all downstream I/O.
///
/// File-level proofs are cached per expression until the dynamic producer changes. Falsification
/// is checked when pruning is polled; satisfaction is checked after the input mask resolves and
/// bypasses downstream filter evaluation.
pub struct FileStatsLayoutReader {
    child: LayoutReaderRef,
    file_stats: FileStatistics,
    session: VortexSession,
    prune_cache: Arc<DashMap<ExactBoundExpr, CachedProof>>,
    satisfaction_cache: Arc<DashMap<ExactBoundExpr, CachedProof>>,
}

impl FileStatsLayoutReader {
    /// Creates a new `FileStatsLayoutReader` wrapping the given child reader.
    pub fn new(child: LayoutReaderRef, file_stats: FileStatistics, session: VortexSession) -> Self {
        Self {
            child,
            file_stats,
            session,
            prune_cache: Arc::new(DashMap::with_hasher_and_shard_amount(Default::default(), 8)),
            satisfaction_cache: Arc::new(DashMap::with_hasher_and_shard_amount(
                Default::default(),
                8,
            )),
        }
    }

    /// Returns the file-level statistics used by this reader.
    pub fn file_stats(&self) -> &FileStatistics {
        &self.file_stats
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
        let child = Arc::clone(&self.child);
        let file_stats = self.file_stats.clone();
        let session = self.session.clone();
        let cache = Arc::clone(&self.prune_cache);
        let expr = expr.clone();
        let row_range = row_range.clone();
        Ok(MaskFuture::new(mask.len(), async move {
            let pruned = {
                let mut entry = cache
                    .entry(ExactBoundExpr(expr.clone()))
                    .or_insert_with(|| CachedProof::new(&expr));
                entry.evaluate(|| {
                    is_falsified_by_file_stats(&expr, child.row_count(), &file_stats, &session)
                })?
            };
            if pruned {
                Ok(Mask::new_false(mask.len()))
            } else {
                child.pruning_evaluation(&row_range, &expr, mask)?.await
            }
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let child = Arc::clone(&self.child);
        let file_stats = self.file_stats.clone();
        let session = self.session.clone();
        let cache = Arc::clone(&self.satisfaction_cache);
        let expr = expr.clone();
        let row_range = row_range.clone();
        Ok(MaskFuture::new(mask.len(), async move {
            let mask = mask.await?;
            if mask.all_false() {
                return Ok(mask);
            }
            let satisfied = {
                let mut entry = cache
                    .entry(ExactBoundExpr(expr.clone()))
                    .or_insert_with(|| CachedProof::new(&expr));
                entry
                    .evaluate(|| {
                        is_satisfied_by_file_stats(&expr, child.row_count(), &file_stats, &session)
                    })
                    .unwrap_or(false)
            };
            if satisfied {
                Ok(mask)
            } else {
                child
                    .filter_evaluation(&row_range, &expr, MaskFuture::ready(mask))?
                    .await
            }
        }))
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

struct CachedProof {
    updates: Option<DynamicExprUpdates>,
    latest: Option<(u64, bool)>,
}

impl CachedProof {
    fn new(expr: &BoundExpression) -> Self {
        Self {
            updates: DynamicExprUpdates::new(expr),
            latest: None,
        }
    }

    fn evaluate(&mut self, proof: impl FnOnce() -> VortexResult<bool>) -> VortexResult<bool> {
        let version = self.updates.as_ref().map_or(0, DynamicExprUpdates::version);
        if let Some((previous, proven)) = self.latest
            && previous == version
        {
            return Ok(proven);
        }
        let proven = proof()?;
        // A proof evaluated across a producer update must not suppress the current filter.
        if self.updates.as_ref().map_or(0, DynamicExprUpdates::version) != version {
            return Ok(false);
        }
        self.latest = Some((version, proven));
        Ok(proven)
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Display;
    use std::fmt::Formatter;
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

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
    use vortex_array::expr::dynamic_expression;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Precision;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
    use vortex_array::scalar_fn::fns::dynamic_expression::input_field_name;
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
    use vortex_layout::segments::SegmentFuture;
    use vortex_layout::segments::SegmentId;
    use vortex_layout::segments::SegmentSink;
    use vortex_layout::segments::SegmentSource;
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

    struct CountingSegments(Arc<TestSegments>, AtomicUsize);

    impl SegmentSource for CountingSegments {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.1.fetch_add(1, Ordering::Relaxed);
            self.0.request(id)
        }
    }

    #[derive(Debug, Default)]
    struct LivePredicate(AtomicU64);

    impl Display for LivePredicate {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.write_str("live predicate")
        }
    }

    impl DynamicExpressionSource for LivePredicate {
        fn generation(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }

        fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
            match self.generation() {
                0 | 2 | 5 | 8 => lit(true),
                3 | 7 => gt(get_item(input_field_name(0), root()), lit(200i32)),
                4 => lit(Scalar::null(DType::Bool(Nullability::Nullable))),
                _ => gt(get_item(input_field_name(0), root()), lit(2i32)),
            }
            .bind(scope)
        }
    }

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
    fn dynamic_proofs_refresh_before_reads() -> VortexResult<()> {
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

            let counted = Arc::new(CountingSegments(segments, AtomicUsize::new(0)));
            let child = layout.new_reader(
                "".into(),
                Arc::clone(&counted) as Arc<dyn SegmentSource>,
                &SESSION,
                &Default::default(),
            )?;

            let reader =
                FileStatsLayoutReader::new(child, test_file_stats(0, 100), SESSION.clone());

            // col > 50 should NOT be prunable since max is 100 (some rows could match).
            let expr = gt(get_item("col", root()), lit(50i32)).bind(reader.dtype())?;
            let mask = Mask::new_true(5);
            let result = reader.pruning_evaluation(&(0..5), &expr, mask)?.await?;
            // Should delegate to child, which returns the mask unchanged (struct reader doesn't prune).
            assert_eq!(result, Mask::new_true(5));

            let producer = Arc::new(LivePredicate::default());
            let dynamic = dynamic_expression(
                Arc::clone(&producer) as Arc<dyn DynamicExpressionSource>,
                [get_item("col", root())],
            )
            .bind(reader.dtype())?;
            for (generation, expected) in [
                (0, [true; 5]),
                (1, [false, false, true, true, true]),
                (2, [true; 5]),
                (3, [false; 5]),
                (4, [false; 5]),
            ] {
                producer.0.store(generation, Ordering::SeqCst);
                assert_eq!(
                    reader
                        .pruning_evaluation(&(0..5), &dynamic, Mask::new_true(5))?
                        .await?,
                    if generation == 3 {
                        Mask::new_false(5)
                    } else {
                        Mask::new_true(5)
                    }
                );
                let result = reader
                    .filter_evaluation(&(0..5), &dynamic, MaskFuture::new_true(5))?
                    .await?;
                assert_eq!(result, Mask::from_iter(expected));
                if generation == 0 {
                    assert_eq!(counted.1.load(Ordering::Relaxed), 0);
                }
            }
            producer.0.store(5, Ordering::SeqCst);
            let selected = Mask::from_iter([true, false, true, false, true]);
            assert_eq!(
                reader
                    .filter_evaluation(&(0..5), &dynamic, MaskFuture::ready(selected.clone()))?
                    .await?,
                selected
            );
            let pending = reader.filter_evaluation(&(0..5), &dynamic, MaskFuture::new_true(5))?;
            producer.0.store(6, Ordering::SeqCst);
            assert_eq!(
                pending.await?,
                Mask::from_iter([false, false, true, true, true])
            );
            assert!(counted.1.load(Ordering::Relaxed) > 0);

            producer.0.store(7, Ordering::SeqCst);
            assert_eq!(
                reader
                    .pruning_evaluation(&(0..5), &dynamic, Mask::new_true(5))?
                    .await?,
                Mask::new_false(5)
            );
            let pending = reader.pruning_evaluation(&(0..5), &dynamic, Mask::new_true(5))?;
            producer.0.store(8, Ordering::SeqCst);
            assert_eq!(pending.await?, Mask::new_true(5));

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
