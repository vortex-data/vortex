// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads a [`VariantLayout`](super::VariantLayout) by reading its stored struct.

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::RowSplits;
use crate::SplitRange;

/// Reads a Variant as an expression over its stored struct.
///
/// Every expression over the Variant is evaluated by the reader of the stored struct, with the
/// Variant replaced by the pack of the stored columns. Optimizing that turns an extraction of a
/// shredded path into a read of the column that holds it, which the struct reader then reads
/// alone.
pub(super) struct VariantReader {
    name: Arc<str>,
    dtype: DType,
    /// Reads the stored struct.
    storage: LayoutReaderRef,
    /// The Variant, as an expression over the stored struct.
    variant: BoundExpression,
}

impl VariantReader {
    pub(super) fn new(
        name: Arc<str>,
        dtype: DType,
        storage: LayoutReaderRef,
        variant: BoundExpression,
    ) -> Self {
        Self {
            name,
            dtype,
            storage,
            variant,
        }
    }

    /// Rewrites `expr`, over the Variant, to an expression over the stored struct.
    fn over_storage(&self, expr: &BoundExpression) -> VortexResult<BoundExpression> {
        let expr = expr.clone().transform_down(|expr| {
            if !expr.is_root() {
                return Ok(Transformed::no(expr));
            }
            Ok(Transformed {
                value: self.variant.clone(),
                order: TraversalOrder::Skip,
                changed: true,
            })
        })?;
        expr.into_inner().optimize_recursive()
    }
}

impl LayoutReader for VariantReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.storage.row_count()
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        // A field of the Variant can be in any of the stored columns.
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
            .pruning_evaluation(row_range, &self.over_storage(expr)?, mask)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.storage
            .filter_evaluation(row_range, &self.over_storage(expr)?, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        self.storage
            .projection_evaluation(row_range, &self.over_storage(expr)?, mask)
    }
}
