// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_layout::ArrayFuture;
use vortex_layout::LayoutReader;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutReaderRef;
use vortex_layout::RowSplits;
use vortex_layout::SplitRange;
use vortex_layout::segments::SegmentSource;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use super::ParquetVariantLayout;
use super::expr::rewrite_variant_expr;

/// Reads a [`ParquetVariantLayout`] by rewriting expressions over the Variant column into
/// expressions over its storage struct, and evaluating those with the storage reader.
pub(super) struct ParquetVariantReader {
    layout: ParquetVariantLayout,
    name: Arc<str>,
    storage: LayoutReaderRef,
    /// Rewritten expressions, keyed by the identity of the original expression. Scans evaluate
    /// the same bound expression for every split.
    rewritten: DashMap<ExactBoundExpr, BoundExpression>,
}

impl ParquetVariantReader {
    pub(super) fn try_new(
        layout: ParquetVariantLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: LayoutReaderContext,
    ) -> VortexResult<Self> {
        let storage = layout
            .slot(0)?
            .vortex_expect("ParquetVariantLayout always has a storage child")
            .new_reader(Arc::clone(&name), segment_source, &session, &ctx)?;
        Ok(Self {
            layout,
            name,
            storage,
            rewritten: Default::default(),
        })
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<BoundExpression> {
        let key = ExactBoundExpr(expr.clone());
        if let Some(rewritten) = self.rewritten.get(&key) {
            return Ok(rewritten.clone());
        }
        let rewritten =
            rewrite_variant_expr(expr, self.layout.storage_dtype(), self.layout.typed_paths())?;
        self.rewritten.insert(key, rewritten.clone());
        Ok(rewritten)
    }
}

impl LayoutReader for ParquetVariantReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        // Field masks address the Variant column as a whole, not its storage children.
        self.storage
            .register_splits(&[FieldMask::All], split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        self.storage
            .pruning_evaluation(row_range, &self.rewrite(expr)?, mask)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.storage
            .filter_evaluation(row_range, &self.rewrite(expr)?, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        self.storage
            .projection_evaluation(row_range, &self.rewrite(expr)?, mask)
    }
}
