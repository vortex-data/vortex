// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`LayoutReader`] decorator that performs file-level stats pruning.
//!
//! If file-level statistics prove that a filter expression cannot match any rows in the file,
//! [`FileStatsLayoutReader`] short-circuits [`pruning_evaluation`](LayoutReader::pruning_evaluation)
//! by returning an all-false mask — avoiding all downstream I/O.

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::NullArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::StatsCatalog;
use vortex_array::expr::lit;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_error::VortexResult;
use vortex_layout::ArrayFuture;
use vortex_layout::LayoutReader;
use vortex_layout::LayoutReaderRef;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use crate::FileStatistics;

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

    /// Evaluates whether the file can be fully pruned for the given expression.
    ///
    /// Returns `true` if file-level stats prove no rows can match, `false` otherwise.
    fn evaluate_file_stats(&self, expr: &Expression) -> VortexResult<bool> {
        let Some(pruning_expr) = expr.stat_falsification(self) else {
            // If there is no pruning expression, we can't prune.
            return Ok(false);
        };

        // Given how we implemented the StatsCatalog, we know the expression must be all literals.
        // We can therefore optimize with a null scope since there are no field references that
        // need to be resolved.
        let simplified = pruning_expr.optimize_recursive(&DType::Null)?;
        if let Some(result) = simplified.as_opt::<Literal>() {
            // Can prune if the result is non-nullable and true
            return Ok(result.as_bool().value() == Some(true));
        }

        // Sometimes expressions don't implement constant folding to literals... In this case,
        // we just execute the expression over a null array.
        let pruning = NullArray::new(1).into_array().apply(&pruning_expr)?;

        let mut ctx = self.session.create_execution_ctx();
        let result = pruning
            .execute::<Canonical>(&mut ctx)?
            .into_bool()
            .into_array()
            .scalar_at(0)?;

        Ok(result.as_bool().value() == Some(true))
    }
}

/// Implements [`StatsCatalog`] to provide file-level stats to expressions during pruning evaluation.
impl StatsCatalog for FileStatsLayoutReader {
    fn stats_ref(&self, field_path: &FieldPath, stat: Stat) -> Option<Expression> {
        // FileStats currently only holds top-level field statistics.
        if field_path.parts().len() != 1 {
            return None;
        }

        let field_name = field_path.parts()[0].as_name()?;
        let field_idx = self.struct_fields.find(field_name)?;
        let field_stats = self.file_stats.stats_sets().get(field_idx)?;

        let stat_value = field_stats.get(stat)?.as_exact()?;
        let field_dtype = self.struct_fields.field_by_index(field_idx)?;
        let stat_scalar = Scalar::try_new(field_dtype, Some(stat_value)).ok()?;

        Some(lit(stat_scalar))
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
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.child.register_splits(field_mask, row_range, splits)
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray as _;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Precision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::scalar_fn::session::ScalarFnSession;
    use vortex_array::session::ArraySession;
    use vortex_array::stats::StatsSet;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::LayoutReader;
    use vortex_layout::LayoutStrategy;
    use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
    use vortex_layout::layouts::table::TableStrategy;
    use vortex_layout::segments::TestSegments;
    use vortex_layout::sequence::SequenceId;
    use vortex_layout::sequence::SequentialArrayStreamExt;
    use vortex_layout::session::LayoutSession;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        VortexSession::empty()
            .with::<ArraySession>()
            .with::<LayoutSession>()
            .with::<ScalarFnSession>()
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

    #[test]
    fn pruning_when_filter_out_of_range() -> VortexResult<()> {
        block_on(|handle| async {
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
                    handle,
                )
                .await?;

            let child = layout.new_reader("".into(), segments, &SESSION)?;

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
                    handle,
                )
                .await?;

            let child = layout.new_reader("".into(), segments, &SESSION)?;

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
}
