use vortex_array::serde::SerializeOptions;
use vortex_array::stats::{STATS_TO_WRITE, Stat};
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentWriter;
use crate::writer::LayoutWriter;
use crate::{Layout, LayoutStrategy, LayoutVTableRef, LayoutWriterExt};

#[derive(Clone)]
pub struct FlatLayoutStrategy {
    /// Stats to preserve when writing arrays
    pub array_stats: Vec<Stat>,
    /// Whether to include padding for memory-mapped reads.
    pub include_padding: bool,
}

impl Default for FlatLayoutStrategy {
    fn default() -> Self {
        Self {
            array_stats: STATS_TO_WRITE.to_vec(),
            include_padding: true,
        }
    }
}

impl LayoutStrategy for FlatLayoutStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(FlatLayoutWriter::new(ctx.clone(), dtype.clone(), self.clone()).boxed())
    }
}

/// Writer for a [`FlatLayout`].
pub struct FlatLayoutWriter {
    ctx: ArrayContext,
    dtype: DType,
    options: FlatLayoutStrategy,
    layout: Option<Layout>,
}

impl FlatLayoutWriter {
    pub fn new(ctx: ArrayContext, dtype: DType, options: FlatLayoutStrategy) -> Self {
        Self {
            ctx,
            dtype,
            options,
            layout: None,
        }
    }
}

fn update_stats(array: &dyn Array, stats: &[Stat]) -> VortexResult<()> {
    // TODO(ngates): consider whether we want to do this
    // array.statistics().compute_all(stats)?;
    array.statistics().retain(stats);
    for child in array.children() {
        update_stats(&child, stats)?
    }
    Ok(())
}

impl LayoutWriter for FlatLayoutWriter {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        if self.layout.is_some() {
            vortex_bail!("FlatLayoutStrategy::push_batch called after finish");
        }
        let row_count = chunk.len() as u64;
        update_stats(&chunk, &self.options.array_stats)?;

        let buffers = chunk.serialize(
            &self.ctx,
            &SerializeOptions {
                offset: 0,
                include_padding: self.options.include_padding,
            },
        );
        let segment_id = segment_writer.put(&buffers);

        self.layout = Some(Layout::new_owned(
            "flat".into(),
            LayoutVTableRef::new_ref(&FlatLayout),
            self.dtype.clone(),
            row_count,
            vec![segment_id],
            vec![],
            None,
        ));
        Ok(())
    }

    fn flush(&mut self, _segment_writer: &mut dyn SegmentWriter) -> VortexResult<()> {
        Ok(())
    }

    fn finish(&mut self, _segment_writer: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        self.layout
            .take()
            .ok_or_else(|| vortex_err!("FlatLayoutStrategy::finish called without push_batch"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::stats::{Precision, Stat};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayContext};
    use vortex_buffer::buffer;
    use vortex_expr::ident;
    use vortex_mask::Mask;

    use crate::ExprEvaluator;
    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::segments::{SegmentSource, TestSegments};
    use crate::writer::LayoutWriterExt;

    // Currently, flat layouts do not force compute stats during write, they only retain
    // pre-computed stats.
    #[should_panic]
    #[test]
    fn flat_stats() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout =
                FlatLayoutWriter::new(ctx.clone(), array.dtype().clone(), Default::default())
                    .push_one(&mut segments, array.into_array())
                    .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let result = layout
                .reader(&segments, &ctx)
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &ident())
                .unwrap()
                .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
                .await
                .unwrap();

            assert_eq!(
                result.statistics().get_as::<bool>(Stat::IsSorted),
                Some(Precision::Exact(true))
            );
        })
    }
}
