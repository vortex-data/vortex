use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;
use vortex_array::array::ChunkedArray;
use vortex_array::{ArrayDType, ArrayData, Canonical, IntoArrayData, IntoArrayVariant};
use vortex_error::VortexResult;
use vortex_expr::pruning::PruningPredicate;
use vortex_expr::ExprRef;

use crate::layouts::chunked::reader::ChunkedReader;
use crate::reader::LayoutScanExt;
use crate::{Evaluator, RowMask};

#[async_trait]
impl Evaluator for ChunkedReader {
    async fn evaluate(
        self: Arc<Self>,
        row_mask: RowMask,
        expr: ExprRef,
    ) -> VortexResult<ArrayData> {
        // First we need to compute the pruning mask
        let pruning_predicate = PruningPredicate::try_new(&expr);
        let pruning_mask = if let Some(predicate) = pruning_predicate {
            // If the expression is prune-able, then fetch the stats table
            if let Some(stats_table) = self.stats_table_fut().await {
                predicate
                    .evaluate(stats_table.array())?
                    .map(|mask| mask.into_bool())
                    .transpose()?
                    .map(|mask| mask.boolean_buffer())
            } else {
                None
            }
        } else {
            None
        };

        // Now we set up futures to evaluate each chunk at the same time
        let mut chunks = Vec::with_capacity(self.nchunks());
        let mut row_offset = 0;
        for chunk_idx in 0..self.nchunks() {
            let chunk_reader = self.child(chunk_idx)?;

            // Figure out the row range of the chunk
            let chunk_len = chunk_reader.layout().row_count();
            let chunk_range = row_offset..row_offset + chunk_len;
            row_offset += chunk_len;

            // Try to skip the chunk based on the row-mask
            if row_mask.is_disjoint(chunk_range.clone()) {
                continue;
            }

            // Try to skip the chunk based on the pruning predicate
            if let Some(pruning_mask) = &pruning_mask {
                if pruning_mask.value(chunk_idx) {
                    continue;
                }
            }

            // Otherwise, we need to read it. So we set up a mask for the chunk range.
            let chunk_mask = row_mask
                .slice(chunk_range.start, chunk_range.end)?
                .shift(chunk_range.start)?;
            chunks.push(chunk_reader.evaluate(chunk_mask, expr.clone()));
        }

        // Wait for all chunks to be evaluated
        let chunks = try_join_all(chunks).await?;

        let dtype = if let Some(chunk) = chunks.first() {
            chunk.dtype().clone()
        } else {
            expr.evaluate(&Canonical::empty(self.dtype())?.into_array())?
                .dtype()
                .clone()
        };

        Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
    }
}
//
// #[cfg(test)]
// mod test {
//     use std::sync::Arc;
//
//     use vortex_array::{ArrayLen, IntoArrayData, IntoArrayVariant};
//     use vortex_buffer::buffer;
//     use vortex_dtype::Nullability::NonNullable;
//     use vortex_dtype::{DType, PType};
//     use vortex_error::vortex_panic;
//     use vortex_expr::{gt, lit, Identity};
//
//     use crate::layouts::chunked::evaluator::{ChunkState, ChunkedEvaluator};
//     use crate::layouts::chunked::reader::ChunkedReader;
//     use crate::layouts::chunked::writer::ChunkedLayoutWriter;
//     use crate::operations::{Operation, Poll};
//     use crate::segments::test::TestSegments;
//     use crate::strategies::LayoutWriterExt;
//     use crate::{LayoutData, RowMask};
//
//     /// Create a chunked layout with three chunks of primitive arrays.
//     fn chunked_layout() -> (TestSegments, LayoutData) {
//         let mut segments = TestSegments::default();
//         let layout = ChunkedLayoutWriter::new(
//             &DType::Primitive(PType::I32, NonNullable),
//             Default::default(),
//         )
//         .push_all(
//             &mut segments,
//             [
//                 Ok(buffer![1, 2, 3].into_array()),
//                 Ok(buffer![4, 5, 6].into_array()),
//                 Ok(buffer![7, 8, 9].into_array()),
//             ],
//         )
//         .unwrap();
//         (segments, layout)
//     }
//
//     #[test]
//     fn test_chunked_scan() {
//         let (segments, layout) = chunked_layout();
//
//         let scan = layout.reader(Default::default()).unwrap();
//         let result = segments
//             .evaluate(scan, Identity::new_expr())
//             .into_primitive()
//             .unwrap();
//
//         assert_eq!(result.len(), 9);
//         assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
//     }
//
//     #[test]
//     // FIXME(ngates): when we make LayoutReader Send we will fix this
//     #[allow(clippy::arc_with_non_send_sync)]
//     fn test_chunked_pruning_mask() {
//         let (segments, layout) = chunked_layout();
//         let row_count = layout.row_count();
//         let reader = ChunkedReader::try_new(layout, Default::default()).unwrap();
//
//         // Populate the stats table so that we can verify the pruning mask works.
//         _ = reader.stats_table_op().unwrap().poll(&segments).unwrap();
//
//         let expr = gt(Identity::new_expr(), lit(6));
//         let mut evaluator = ChunkedEvaluator::new(
//             Arc::new(reader),
//             RowMask::new_valid_between(0, row_count),
//             expr,
//         );
//
//         // Then we poll the chunked scanner without any segments so _only_ the stats were
//         // available.
//         let Poll::NeedMore(_segments) = evaluator.poll(&TestSegments::default()).unwrap() else {
//             unreachable!()
//         };
//
//         // Now we validate that based on the pruning mask, we have excluded the first two chunks
//         let chunk_states = evaluator.chunk_states.as_ref().unwrap().as_slice();
//         if !matches!(chunk_states[0], ChunkState::Resolved(None)) {
//             vortex_panic!("Expected first chunk to be pruned");
//         }
//         if !matches!(chunk_states[1], ChunkState::Resolved(None)) {
//             vortex_panic!("Expected second chunk to be pruned");
//         }
//         if !matches!(chunk_states[2], ChunkState::Pending(_)) {
//             vortex_panic!("Expected third chunk to be read");
//         }
//     }
// }
