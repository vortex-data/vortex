use vortex_array::parts::ArrayPartsFlatBuffer;
use vortex_array::ArrayData;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_flatbuffers::WriteFlatBufferExt;

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

        // We store each Array buffer in its own segment.
        let mut segment_ids = vec![];
        for child in chunk.depth_first_traversal() {
            for buffer in child.byte_buffers() {
                // TODO(ngates): decide a way of splitting buffers if they exceed u32 size.
                //  We could write empty segments either side of buffers to concatenate?
                //  Or we could use Layout::metadata to store this information.
                segment_ids.push(segments.put(buffer));
            }
        }

        // ...followed by a FlatBuffer describing the array layout.
        let flatbuffer = ArrayPartsFlatBuffer::new(&chunk).write_flatbuffer_bytes();
        segment_ids.push(segments.put(flatbuffer.into_inner()));

        self.layout = Some(LayoutData::new_owned(
            &FlatLayout,
            self.dtype.clone(),
            row_count,
            Some(segment_ids),
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
