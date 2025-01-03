use vortex_array::ArrayData;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentWriter;
use crate::strategies::LayoutWriter;
use crate::LayoutData;

/// Writer for the flat layout.
pub struct FlatLayoutWriter {
    dtype: DType,
    layout: Option<LayoutData>,
}

impl FlatLayoutWriter {
    pub fn new(dtype: DType) -> Self {
        Self {
            dtype,
            layout: None,
        }
    }
}

impl LayoutWriter for FlatLayoutWriter {
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
        self.layout = Some(LayoutData::new_owned(
            &FlatLayout,
            self.dtype.clone(),
            row_count,
            Some(vec![segment_id]),
            None,
            None,
        ));
        Ok(())
    }

    fn finish(&mut self, _segments: &mut dyn SegmentWriter) -> VortexResult<LayoutData> {
        self.layout
            .take()
            .ok_or_else(|| vortex_err!("FlatLayoutStrategy::finish called without push_batch"))
    }
}
