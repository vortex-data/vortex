use std::ops::{Deref, Range};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::ready;
use futures::stream::FuturesOrdered;
use futures::{FutureExt, TryStreamExt};
use itertools::Itertools;
use vortex_array::arrays::ChunkedArray;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::layouts::chunked::ChunkedLayout;
use crate::reader::LayoutReader;
use crate::segments::SegmentSource;
use crate::{
    ArrayEvaluation, Layout, LayoutData, LayoutReaderRef, LazyReaderChildren, MaskEvaluation,
    PruningEvaluation,
};

/// A [`LayoutReader`] for chunked layouts.
pub struct ChunkedReader {
    pub(super) layout: Arc<ChunkedLayout>,
    pub(super) lazy_children: LazyReaderChildren,
    /// Row offset for each chunk
    pub(super) chunk_offsets: Vec<u64>,
}

impl ChunkedReader {
    pub fn new(
        layout: Arc<ChunkedLayout>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> Self {
        let nchildren = layout.nchildren();

        let mut chunk_offsets = vec![0; nchildren];
        for i in 1..nchildren {
            chunk_offsets[i] = chunk_offsets[i - 1] + layout.children.child_row_count(i - 1);
        }

        let lazy_children =
            LazyReaderChildren::new(layout.children.clone(), segment_source.clone(), ctx.clone());

        Self {
            layout,
            lazy_children,
            chunk_offsets,
        }
    }

    /// Return the [`LayoutReader`] for the given chunk.
    pub(super) fn chunk_reader(&self, idx: usize) -> VortexResult<LayoutReaderRef> {
        self.lazy_children
            .get(idx, &self.layout.dtype, &format!("[{}]", idx))
    }
}

impl Deref for ChunkedReader {
    type Target = dyn Layout;

    fn deref(&self) -> &Self::Target {
        self.layout.as_ref()
    }
}

impl LayoutReader for ChunkedReader {
    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        let mut chunk_evals = vec![];
        let mut mask_ranges = vec![];

        for (chunk_idx, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.child(chunk_idx)?;
            let chunk_eval = chunk_reader.pruning_evaluation(&chunk_range, expr)?;
            chunk_evals.push(chunk_eval);
            mask_ranges.push(mask_range);
        }

        Ok(Box::new(ChunkedPruningEvaluation {
            layout: self.layout().clone(),
            chunk_evals,
            mask_ranges,
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        let mut chunk_evals = vec![];
        let mut mask_ranges = vec![];

        for (chunk_idx, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.child(chunk_idx)?;
            let chunk_eval = chunk_reader.filter_evaluation(&chunk_range, expr)?;
            chunk_evals.push(chunk_eval);
            mask_ranges.push(mask_range);
        }

        Ok(Box::new(ChunkedMaskEvaluation {
            layout: self.layout().clone(),
            chunk_evals,
            mask_ranges,
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        let dtype = expr.return_dtype(self.dtype())?;
        let mut chunk_evals = vec![];
        let mut mask_ranges = vec![];

        for (chunk_idx, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.child(chunk_idx)?;
            let chunk_eval = chunk_reader.projection_evaluation(&chunk_range, expr)?;
            chunk_evals.push(chunk_eval);
            mask_ranges.push(mask_range);
        }

        Ok(Box::new(ChunkedArrayEvaluation {
            dtype,
            chunk_evals,
            mask_ranges,
        }))
    }
}

struct ChunkedPruningEvaluation {
    layout: LayoutData,
    chunk_evals: Vec<Box<dyn PruningEvaluation>>,
    mask_ranges: Vec<Range<usize>>,
}

#[async_trait]
impl PruningEvaluation for ChunkedPruningEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        log::debug!(
            "Chunked pruning evaluation {} (mask = {})",
            self.layout.name(),
            mask.density()
        );

        // Split the mask over each chunk.
        let masks: Vec<_> = FuturesOrdered::from_iter(
            self.mask_ranges
                .iter()
                .map(|range| mask.slice(range.start, range.end - range.start))
                .zip_eq(&self.chunk_evals)
                .map(|(mask, chunk_eval)| {
                    if mask.all_false() {
                        // If the mask is all false, we can skip the evaluation.
                        ready(Ok(mask)).boxed()
                    } else {
                        chunk_eval.invoke(mask).boxed()
                    }
                }),
        )
        .try_collect()
        .await?;

        // If there is only one mask, we can return it directly.
        if masks.len() == 1 {
            return Ok(masks.into_iter().next().vortex_expect("one mask"));
        }

        // Combine the masks.
        Ok(Mask::from_iter(masks))
    }
}

struct ChunkedMaskEvaluation {
    layout: LayoutData,
    chunk_evals: Vec<Box<dyn MaskEvaluation>>,
    mask_ranges: Vec<Range<usize>>,
}

#[async_trait]
impl MaskEvaluation for ChunkedMaskEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        log::debug!(
            "Chunked mask evaluation {} (mask = {})",
            self.layout.name(),
            mask.density()
        );

        // Split the mask over each chunk.
        let masks: Vec<_> = FuturesOrdered::from_iter(
            self.mask_ranges
                .iter()
                .map(|range| mask.slice(range.start, range.end - range.start))
                .zip_eq(&self.chunk_evals)
                .map(|(mask, chunk_eval)| {
                    if mask.all_false() {
                        // If the mask is all false, we can skip the evaluation.
                        ready(Ok(mask)).boxed()
                    } else {
                        chunk_eval.invoke(mask).boxed()
                    }
                }),
        )
        .try_collect()
        .await?;

        // If there is only one mask, we can return it directly.
        if masks.len() == 1 {
            return Ok(masks.into_iter().next().vortex_expect("one mask"));
        }

        // Combine the masks.
        Ok(Mask::from_iter(masks))
    }
}

struct ChunkedArrayEvaluation {
    dtype: DType,
    chunk_evals: Vec<Box<dyn ArrayEvaluation>>,
    mask_ranges: Vec<Range<usize>>,
}

#[async_trait]
impl ArrayEvaluation for ChunkedArrayEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        // Split the mask over each chunk.
        let chunks: Vec<_> = FuturesOrdered::from_iter(
            self.mask_ranges
                .iter()
                .map(|range| mask.slice(range.start, range.end - range.start))
                .zip_eq(&self.chunk_evals)
                .filter(|(mask, _chunk_eval)| mask.true_count() > 0)
                .map(|(mask, chunk_eval)| chunk_eval.invoke(mask)),
        )
        .try_collect()
        .await?;

        // If there is only one chunk, we can return it directly.
        if chunks.len() == 1 {
            return Ok(chunks.into_iter().next().vortex_expect("one chunk"));
        }

        // Combine the arrays.
        Ok(ChunkedArray::try_new(chunks, self.dtype.clone())?.to_array())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use futures::executor::block_on;
    use rstest::{fixture, rstest};
    use vortex_array::{ArrayContext, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};
    use vortex_expr::Identity;
    use vortex_mask::Mask;

    use crate::LayoutData;
    use crate::layouts::chunked::writer::ChunkedLayoutWriter;
    use crate::segments::{SegmentSource, TestSegments};
    use crate::writer::LayoutWriterExt;

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn chunked_layout() -> (ArrayContext, Arc<dyn SegmentSource>, LayoutData) {
        let ctx = ArrayContext::empty();
        let mut segments = TestSegments::default();
        let layout = ChunkedLayoutWriter::new(
            ctx.clone(),
            DType::Primitive(PType::I32, NonNullable),
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
        (ctx, Arc::new(segments), layout)
    }

    #[rstest]
    fn test_chunked_evaluator(
        #[from(chunked_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn SegmentSource>,
            LayoutData,
        ),
    ) {
        block_on(async {
            let result = layout
                .reader(&segments, &ctx)
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &Identity::new_expr())
                .unwrap()
                .invoke(Mask::new_true(usize::try_from(layout.row_count()).unwrap()))
                .await
                .unwrap()
                .to_primitive()
                .unwrap();

            assert_eq!(result.len(), 9);
            assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        })
    }
}
