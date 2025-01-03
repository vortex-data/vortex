use vortex_array::ArrayData;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentWriter;
use crate::strategies::{LayoutStrategy, LayoutWriter};
use crate::LayoutData;

pub struct FlatLayoutStrategy {
    dtype: DType,
    layout: Option<LayoutData>,
}

impl LayoutStrategy for FlatLayout {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(Box::new(FlatLayoutStrategy {
            dtype: dtype.clone(),
            layout: None,
        }) as Box<dyn LayoutWriter>)
    }
}

impl LayoutWriter for FlatLayoutStrategy {
    fn push_chunk(
        &mut self,
        segments: &mut dyn SegmentWriter,
        chunk: ArrayData,
    ) -> VortexResult<()> {
        if self.layout.is_some() {
            vortex_bail!("FlatLayoutStrategy::push_batch called after finish");
        }
        let row_count = chunk.len() as u64;
        let segment_id = segments.put_chunk(chunk);
        self.layout = Some(FlatLayout::new(self.dtype.clone(), row_count, segment_id));
        Ok(())
    }

    fn finish(&mut self, _segments: &mut dyn SegmentWriter) -> VortexResult<LayoutData> {
        self.layout
            .take()
            .ok_or_else(|| vortex_err!("FlatLayoutStrategy::finish called without push_batch"))
    }
}
