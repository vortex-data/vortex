use vortex_array::serde::SerializeOptions;
use vortex_array::stats::{STATS_TO_WRITE, Stat};
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentWriter;
use crate::writer::LayoutWriter;
use crate::{Layout, LayoutStrategy, LayoutVTableRef, LayoutWriterExt};

#[derive(Clone)]
pub struct FlatLayoutOptions {
    /// Stats to preserve when writing arrays
    pub array_stats: Vec<Stat>,
    /// Whether to include padding for memory-mapped reads.
    pub include_padding: bool,
}

impl Default for FlatLayoutOptions {
    fn default() -> Self {
        Self {
            array_stats: STATS_TO_WRITE.to_vec(),
            include_padding: true,
        }
    }
}

impl LayoutStrategy for FlatLayoutOptions {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(FlatLayoutWriter::new(dtype.clone(), self.clone()).boxed())
    }
}

/// Writer for a [`FlatLayout`].
pub struct FlatLayoutWriter {
    options: FlatLayoutOptions,
    dtype: DType,
    layout: Option<Layout>,
}

impl FlatLayoutWriter {
    pub fn new(dtype: DType, options: FlatLayoutOptions) -> Self {
        Self {
            options,
            dtype,
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
        segments: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        if self.layout.is_some() {
            vortex_bail!("FlatLayoutStrategy::push_batch called after finish");
        }
        let row_count = chunk.len() as u64;
        update_stats(&chunk, &self.options.array_stats)?;

        let buffers = chunk.serialize(&SerializeOptions {
            offset: 0,
            include_padding: self.options.include_padding,
        });
        let segment_id = segments.put(&buffers);

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

    fn finish(&mut self, _segments: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        self.layout
            .take()
            .ok_or_else(|| vortex_err!("FlatLayoutStrategy::finish called without push_batch"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use vortex_array::Array;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::stats::{Precision, Stat};
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_expr::ident;

    use crate::RowMask;
    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::segments::test::TestSegments;
    use crate::writer::LayoutWriterExt;

    // Currently, flat layouts do not force compute stats during write, they only retain
    // pre-computed stats.
    #[should_panic]
    #[test]
    fn flat_stats() {
        block_on(async {
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutWriter::new(array.dtype().clone(), Default::default())
                .push_one(&mut segments, array.into_array())
                .unwrap();

            let result = layout
                .reader(Arc::new(segments), Default::default())
                .unwrap()
                .evaluate_expr(RowMask::new_valid_between(0, layout.row_count()), ident())
                .await
                .unwrap();

            assert_eq!(
                result.statistics().get_as::<bool>(Stat::IsSorted),
                Some(Precision::Exact(true))
            );
        })
    }
}
