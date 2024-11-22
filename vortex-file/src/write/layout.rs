use bytes::Bytes;
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use vortex_flatbuffers::{footer as fb, WriteFlatBuffer};
use vortex_ipc::stream_writer::ByteRange;

use crate::{
    LayoutId, CHUNKED_LAYOUT_ID, COLUMNAR_LAYOUT_ID, FLAT_LAYOUT_ID, INLINE_SCHEMA_LAYOUT_ID,
};

#[derive(Debug, Clone)]
pub struct LayoutSpec {
    id: LayoutId,
    buffers: Option<Vec<ByteRange>>,
    children: Option<Vec<LayoutSpec>>,
    row_count: u64,
    metadata: Option<Bytes>,
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
    pub fn chunked(children: Vec<LayoutSpec>, row_count: u64, has_metadata: bool) -> Self {
        Self {
            id: CHUNKED_LAYOUT_ID,
            buffers: None,
            children: Some(children),
            row_count,
            metadata: Some(Bytes::copy_from_slice(&[has_metadata as u8])),
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

    pub fn inlined_schema(
        children: Vec<LayoutSpec>,
        row_count: u64,
        dtype_buffer: ByteRange,
    ) -> Self {
        Self {
            id: INLINE_SCHEMA_LAYOUT_ID,
            buffers: Some(vec![dtype_buffer]),
            children: Some(children),
            row_count,
            metadata: None,
        }
    }
}

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
