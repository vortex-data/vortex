use flatbuffers::{FlatBufferBuilder, WIPOffset};
use vortex_buffer::Buffer;
use vortex_flatbuffers::{footer as fb, FlatBufferRoot, WriteFlatBuffer};

use crate::byte_range::ByteRange;
use crate::{LayoutId, CHUNKED_LAYOUT_ID, COLUMNAR_LAYOUT_ID, FLAT_LAYOUT_ID};

#[derive(Debug, Clone)]
pub struct LayoutSpec {
    id: LayoutId,
    buffers: Option<Vec<ByteRange>>,
    children: Option<Vec<LayoutSpec>>,
    row_count: u64,
    metadata: Option<Buffer>,
}

impl LayoutSpec {
    pub fn flat(buffer: ByteRange, row_count: u64) -> Self {
        Self {
            id: FLAT_LAYOUT_ID,
            buffers: Some(vec![buffer]),
            children: None,
            row_count,
            metadata: None,
        }
    }

    /// Create a chunked layout with children.
    ///
    /// has_metadata indicates whether first child is a layout containing metadata about other children.
    pub fn chunked(children: Vec<LayoutSpec>, row_count: u64, metadata: Option<Buffer>) -> Self {
        Self {
            id: CHUNKED_LAYOUT_ID,
            buffers: None,
            children: Some(children),
            row_count,
            metadata,
        }
    }

    pub fn column(children: Vec<LayoutSpec>, row_count: u64) -> Self {
        Self {
            id: COLUMNAR_LAYOUT_ID,
            buffers: None,
            children: Some(children),
            row_count,
            metadata: None,
        }
    }
}

impl FlatBufferRoot for LayoutSpec {}

impl WriteFlatBuffer for LayoutSpec {
    type Target<'a> = fb::Layout<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let buffer_offsets = self.buffers.as_ref().map(|buf| {
            buf.iter()
                .map(|b| fb::Buffer::new(b.begin, b.end))
                .collect::<Vec<_>>()
        });
        let buffers = buffer_offsets.map(|bufs| fbb.create_vector(&bufs));
        let metadata = self.metadata.as_ref().map(|b| fbb.create_vector(b));
        let child_offsets = self.children.as_ref().map(|children| {
            children
                .iter()
                .map(|layout| layout.write_flatbuffer(fbb))
                .collect::<Vec<_>>()
        });
        let children = child_offsets.map(|c| fbb.create_vector(&c));
        fb::Layout::create(
            fbb,
            &fb::LayoutArgs {
                encoding: self.id.0,
                buffers,
                children,
                row_count: self.row_count,
                metadata,
            },
        )
    }
}
