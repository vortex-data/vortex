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
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
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
use crate::pruning::can_prune_file_stats;

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
    struct_fields: StructFields,
    session: VortexSession,
    prune_cache: DashMap<Expression, bool>,
}

impl FileStatsLayoutReader {
    /// Creates a new `FileStatsLayoutReader` wrapping the given child reader.
    ///
    /// The `struct_fields` are derived from the child reader's dtype. If the dtype is not a
    /// struct, the available stats will be empty and no pruning will occur.
    ///
    /// Pre-computes the set of available stat field paths from the struct fields and file stats.
    pub fn new(child: LayoutReaderRef, file_stats: FileStatistics, session: VortexSession) -> Self {
        let struct_fields = child
            .dtype()
            .as_struct_fields_opt()
            .cloned()
            .unwrap_or_default();

        Self {
            child,
            file_stats,
            struct_fields,
            session,
            prune_cache: Default::default(),
        }
    }

    /// Evaluates whether file-level statistics prove `expr` cannot match.
    ///
    /// Row-count placeholders are resolved against the full file row count,
    /// independent of the requested row range.
    fn evaluate_file_stats(&self, expr: &Expression) -> VortexResult<bool> {
        can_prune_file_stats(
            expr,
            self.child.dtype(),
            self.child.row_count(),
            &self.file_stats,
            &self.struct_fields,
            &self.session,
        )
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
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // Check cache first with read-only lock.
        if let Some(pruned) = self.prune_cache.get(expr) {
            if *pruned {
                return Ok(MaskFuture::ready(Mask::new_false(mask.len())));
            }
            return self.child.pruning_evaluation(row_range, expr, mask);
        }

        // Evaluate and cache.
        let pruned = self.evaluate_file_stats(expr)?;
        self.prune_cache.insert(expr.clone(), pruned);

        if pruned {
            Ok(MaskFuture::ready(Mask::new_false(mask.len())))
        } else {
            self.child.pruning_evaluation(row_range, expr, mask)
        }
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.child.filter_evaluation(row_range, expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
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
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::datetime::TemporalData;
    use vortex_array::dtype::DType;
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
    use vortex_array::expr::stats::Stat;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::stats::StatsSet;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_layout::LayoutReader;
    use vortex_layout::LayoutStrategy;
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
        vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
    });

    fn test_file_stats(min: i32, max: i32) -> FileStatistics {
        let mut stats = StatsSet::default();
        stats.set(Stat::Min, Precision::exact(ScalarValue::from(min)));
        stats.set(Stat::Max, Precision::exact(ScalarValue::from(max)));
        FileStatistics::new(
            Arc::from([stats]),
            Arc::from([DType::Primitive(PType::I32, Nullability::NonNullable)]),
        )
    }

    fn test_file_null_count_stats(null_count: u64) -> FileStatistics {
        let mut stats = StatsSet::default();
        stats.set(
            Stat::NullCount,
            Precision::exact(ScalarValue::from(null_count)),
        );
        FileStatistics::new(
            Arc::from([stats]),
            Arc::from([DType::Primitive(PType::I32, Nullability::Nullable)]),
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
                    ctx,
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
            let expr = gt(get_item("col", root()), lit(200i32));
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
                    ctx,
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
            let expr = gt(get_item("col", root()), lit(50i32));
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
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    struct_array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let child = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
            let reader =
                FileStatsLayoutReader::new(child, test_file_stats(0, 100), SESSION.clone());

            let expr = gt(checked_add(get_item("col", root()), lit(5i32)), lit(102i32));
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
                    ctx,
                    Arc::clone(&segments) as Arc<dyn SegmentSink>,
                    struct_array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let child = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;

            // File-level stats: 1 null in deleted_at.
            let mut stats = StatsSet::default();
            stats.set(Stat::NullCount, Precision::exact(ScalarValue::from(1u64)));
            let file_stats = FileStatistics::new(Arc::from([stats]), Arc::from([ts_dtype]));

            let reader = FileStatsLayoutReader::new(child, file_stats, SESSION.clone());

            // `is_null(deleted_at)` — should NOT panic or error due to dtype mismatch.
            let expr = is_null(get_item("deleted_at", root()));
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
                    ctx,
                    Arc::clone(&segments) as Arc<dyn SegmentSink>,
                    struct_array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let child = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;

            let reader =
                FileStatsLayoutReader::new(child, test_file_null_count_stats(5), SESSION.clone());

            let expr = is_not_null(get_item("col", root()));
            let mask = Mask::new_true(5);
            let result = reader.pruning_evaluation(&(0..5), &expr, mask)?.await?;
            assert_eq!(result, Mask::new_false(5));

            Ok(())
        })
    }
}
