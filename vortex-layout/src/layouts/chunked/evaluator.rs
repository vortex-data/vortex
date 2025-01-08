use std::sync::Arc;

use vortex_array::array::ChunkedArray;
use vortex_array::{ArrayDType, ArrayData, Canonical, IntoArrayData, IntoArrayVariant};
use vortex_error::{vortex_panic, VortexExpect, VortexResult};
use vortex_expr::pruning::PruningPredicate;
use vortex_expr::ExprRef;

use crate::layouts::chunked::reader::ChunkedReader;
use crate::operations::{Operation, Poll};
use crate::reader::{EvalOp, LayoutScanExt};
use crate::segments::SegmentReader;
use crate::{ready, RowMask};

/// Evaluation operation for an expression over a chunked layout.
pub(crate) struct ChunkedEvaluator {
    reader: Arc<ChunkedReader>,
    row_mask: RowMask,
    expr: ExprRef,
    pruning_predicate: Option<PruningPredicate>,
    // State for each chunk in the layout
    chunk_states: Option<Vec<ChunkState>>,
}

impl ChunkedEvaluator {
    pub fn new(chunked_scan: Arc<ChunkedReader>, row_mask: RowMask, expr: ExprRef) -> Self {
        let pruning_predicate = PruningPredicate::try_new(&expr);
        Self {
            reader: chunked_scan,
            row_mask,
            expr,
            pruning_predicate,
            chunk_states: None,
        }
    }
}

enum ChunkState {
    Pending(EvalOp),
    Resolved(Option<ArrayData>),
}

impl Operation for ChunkedEvaluator {
    type Output = ArrayData;

    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        // If we haven't set up our chunk state yet, then we need to do that first.
        if self.chunk_states.is_none() {
            // First we need to compute the pruning mask
            let pruning_mask = if let Some(predicate) = &self.pruning_predicate {
                // If the expression is prune-able, then fetch the stats table
                if let Some(stats_table) = ready!(self.reader.stats_table_op()?.poll(segments)) {
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

            // Now we can set up the chunk state.
            let mut chunks = Vec::with_capacity(self.reader.nchunks());
            let mut row_offset = 0;
            for chunk_idx in 0..self.reader.nchunks() {
                let chunk_reader = self.reader.child(chunk_idx)?;

                // Figure out the row range of the chunk
                let chunk_len = chunk_reader.layout().row_count();
                let chunk_range = row_offset..row_offset + chunk_len;
                row_offset += chunk_len;

                // Try to skip the chunk based on the row-mask
                if self.row_mask.is_disjoint(chunk_range.clone()) {
                    chunks.push(ChunkState::Resolved(None));
                    continue;
                }

                // Try to skip the chunk based on the pruning predicate
                if let Some(pruning_mask) = &pruning_mask {
                    if pruning_mask.value(chunk_idx) {
                        chunks.push(ChunkState::Resolved(None));
                        continue;
                    }
                }

                // Otherwise, we need to read it. So we set up a mask for the chunk range.
                let chunk_mask = self
                    .row_mask
                    .slice(chunk_range.start, chunk_range.end)?
                    .shift(chunk_range.start)?;
                let chunk_evaluator =
                    chunk_reader.create_evaluator(chunk_mask, self.expr.clone())?;
                chunks.push(ChunkState::Pending(chunk_evaluator));
            }

            self.chunk_states = Some(chunks);
        }

        let chunk_states = self
            .chunk_states
            .as_mut()
            .vortex_expect("chunk state not set");

        // Now we try to read the chunks.
        let mut needed = vec![];
        for chunk_state in chunk_states.iter_mut() {
            match chunk_state {
                ChunkState::Pending(scanner) => match scanner.poll(segments)? {
                    Poll::Some(array) => {
                        // Resolve the chunk
                        *chunk_state = ChunkState::Resolved(Some(array));
                    }
                    Poll::NeedMore(segment_ids) => {
                        // Request more segments
                        needed.extend(segment_ids);
                    }
                },
                ChunkState::Resolved(_) => {
                    // Already resolved
                }
            }
        }

        // If we need more segments, then request them.
        if !needed.is_empty() {
            return Ok(Poll::NeedMore(needed));
        }

        // Otherwise, we've read all the chunks, so we're done.
        let chunks = chunk_states
            .iter_mut()
            .filter_map(|state| match state {
                ChunkState::Resolved(array) => array.take(),
                _ => vortex_panic!(
                    "This is a bug. Missing a chunk array with no more segments to read"
                ),
            })
            .collect::<Vec<_>>();

        let dtype = if let Some(chunk) = chunks.first() {
            chunk.dtype().clone()
        } else {
            self.expr
                .evaluate(&Canonical::empty(self.reader.dtype())?.into_array())?
                .dtype()
                .clone()
        };

        Ok(Poll::Some(
            ChunkedArray::try_new(chunks, dtype)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_array::{ArrayLen, IntoArrayData, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};
    use vortex_error::vortex_panic;
    use vortex_expr::{gt, lit, Identity};

    use crate::layouts::chunked::evaluator::{ChunkState, ChunkedEvaluator};
    use crate::layouts::chunked::reader::ChunkedReader;
    use crate::layouts::chunked::writer::ChunkedLayoutWriter;
    use crate::operations::{Operation, Poll};
    use crate::segments::test::TestSegments;
    use crate::strategies::LayoutWriterExt;
    use crate::{LayoutData, RowMask};

    /// Create a chunked layout with three chunks of primitive arrays.
    fn chunked_layout() -> (TestSegments, LayoutData) {
        let mut segments = TestSegments::default();
        let layout = ChunkedLayoutWriter::new(
            &DType::Primitive(PType::I32, NonNullable),
            Default::default(),
        )
        .push_all(
            &mut segments,
            [
                Ok(buffer![1, 2, 3].into_array()),
                Ok(buffer![4, 5, 6].into_array()),
                Ok(buffer![7, 8, 9].into_array()),
            ],
        )
        .unwrap();
        (segments, layout)
    }

    #[test]
    fn test_chunked_scan() {
        let (segments, layout) = chunked_layout();

        let scan = layout.reader(Default::default()).unwrap();
        let result = segments
            .evaluate(scan, Identity::new_expr())
            .into_primitive()
            .unwrap();

        assert_eq!(result.len(), 9);
        assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    // FIXME(ngates): when we make LayoutReader Send we will fix this
    #[allow(clippy::arc_with_non_send_sync)]
    fn test_chunked_pruning_mask() {
        let (segments, layout) = chunked_layout();
        let row_count = layout.row_count();
        let reader = ChunkedReader::try_new(layout, Default::default()).unwrap();

        // Populate the stats table so that we can verify the pruning mask works.
        _ = reader.stats_table_op().unwrap().poll(&segments).unwrap();

        let expr = gt(Identity::new_expr(), lit(6));
        let mut evaluator = ChunkedEvaluator::new(
            Arc::new(reader),
            RowMask::new_valid_between(0, row_count),
            expr,
        );

        // Then we poll the chunked scanner without any segments so _only_ the stats were
        // available.
        let Poll::NeedMore(_segments) = evaluator.poll(&TestSegments::default()).unwrap() else {
            unreachable!()
        };

        // Now we validate that based on the pruning mask, we have excluded the first two chunks
        let chunk_states = evaluator.chunk_states.as_ref().unwrap().as_slice();
        if !matches!(chunk_states[0], ChunkState::Resolved(None)) {
            vortex_panic!("Expected first chunk to be pruned");
        }
        if !matches!(chunk_states[1], ChunkState::Resolved(None)) {
            vortex_panic!("Expected second chunk to be pruned");
        }
        if !matches!(chunk_states[2], ChunkState::Pending(_)) {
            vortex_panic!("Expected third chunk to be read");
        }
    }
}
