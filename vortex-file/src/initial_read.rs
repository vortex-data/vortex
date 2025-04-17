use flatbuffers::root;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_flatbuffers::{FlatBuffer, ReadFlatBuffer, dtype as fbd};

use crate::footer::{FileStatistics, PostscriptSegment};

pub struct InitialRead {
    pub(crate) dtype_segment: Option<PostscriptSegment>,
    pub(crate) stats_segment: Option<PostscriptSegment>,
    pub(crate) layout_segment: PostscriptSegment,
    pub(crate) initial_read: ByteBuffer,
    pub(crate) initial_offset: u64,
}

impl InitialRead {
    pub(crate) fn parse_stats(&self) -> VortexResult<Option<FileStatistics>> {
        self.stats_segment
            .as_ref()
            .map(|segment| {
                let offset = usize::try_from(segment.offset - self.initial_offset)?;
                let sliced_buffer = FlatBuffer::align_from(
                    self.initial_read
                        .slice(offset..offset + (segment.length as usize)),
                );
                FileStatistics::read_flatbuffer_bytes(&sliced_buffer)
            })
            .transpose()
    }

    pub(crate) fn parse_dtype(&self) -> VortexResult<Option<DType>> {
        self.dtype_segment
            .as_ref()
            .map(|segment| {
                let offset = usize::try_from(segment.offset - self.initial_offset)?;
                let sliced_buffer = FlatBuffer::align_from(
                    self.initial_read
                        .slice(offset..offset + (segment.length as usize)),
                );
                let fbd_dtype = root::<fbd::DType>(&sliced_buffer)?;

                DType::try_from_view(fbd_dtype, sliced_buffer.clone())
            })
            .transpose()
    }

    pub fn size(&self) -> usize {
        self.initial_read.len()
    }
}
