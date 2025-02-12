use std::ops::{BitAnd, Sub};

use arrow_buffer::BooleanBufferBuilder;
use async_trait::async_trait;
use vortex_array::Array;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::layouts::stats::reader::StatsReader;
use crate::{ExprEvaluator, RowMask};

#[async_trait]
impl ExprEvaluator for StatsReader {
    async fn evaluate_expr(self: &Self, row_mask: RowMask, expr: ExprRef) -> VortexResult<Array> {
        self.child().evaluate_expr(row_mask, expr).await
    }

    async fn prune_mask(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<RowMask> {
        // Compute the pruning mask
        let Some(pruning_mask) = self.pruning_mask(&expr).await? else {
            // If there is no pruning mask, then we can't prune anything!
            log::debug!(
                "Cannot prune {} in chunked reader, returning mask {}",
                expr,
                row_mask.filter_mask().density()
            );
            return Ok(row_mask);
        };

        log::debug!(
            "Pruning mask for {} {}..{}: {:?}",
            expr,
            row_mask.begin(),
            row_mask.end(),
            pruning_mask
        );

        let mut builder = BooleanBufferBuilder::new(row_mask.len());

        for block_idx in self.block_range(&row_mask) {
            // Figure out the range in the mask that corresponds to the block
            let start = usize::try_from(
                self.block_offset(block_idx)
                    .saturating_sub(row_mask.begin()),
            )?;
            let end = usize::try_from(
                self.block_offset(block_idx + 1)
                    .sub(row_mask.begin())
                    .min(row_mask.len() as u64),
            )?;
            builder.append_n(end - start, pruning_mask.value(block_idx));
        }

        let mask = Mask::from(builder.finish());
        assert_eq!(mask.len(), row_mask.len(), "Mask length mismatch");

        // Apply the mask to the row mask
        let mask = row_mask.filter_mask().bitand(&mask);

        Ok(RowMask::new(mask, row_mask.begin()))
    }
}
