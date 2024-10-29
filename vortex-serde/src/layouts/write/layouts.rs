use bytes::Bytes;
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use vortex_flatbuffers::{footer as fb, WriteFlatBuffer};

use crate::layouts::{
    LayoutId, CHUNKED_LAYOUT_ID, COLUMN_LAYOUT_ID, FLAT_LAYOUT_ID, INLINE_SCHEMA_LAYOUT_ID,
};
use crate::stream_writer::ByteRange;

#[derive(Debug, Clone)]
pub struct Layout {
    id: LayoutId,
    buffers: Option<Vec<ByteRange>>,
    children: Option<Vec<Layout>>,
    metadata: Option<Bytes>,
}

impl Layout {
    pub fn flat(buffer: ByteRange) -> Self {
        Self {
            id: FLAT_LAYOUT_ID,
            buffers: Some(vec![buffer]),
            children: None,
            metadata: None,
        }
    }

    /// Create a chunked layout with children.
    ///
    /// has_metadata indicates whether first child is a layout containing metadata about other children.
    pub fn chunked(children: Vec<Layout>, has_metadata: bool) -> Self {
        Self {
            id: CHUNKED_LAYOUT_ID,
            buffers: None,
            children: Some(children),
            metadata: Some(Bytes::copy_from_slice(&[has_metadata as u8])),
        }
    }

    pub fn column(children: Vec<Layout>) -> Self {
        Self {
            id: COLUMN_LAYOUT_ID,
            buffers: None,
            children: Some(children),
            metadata: None,
        }
    }

    pub fn inlined_schema(children: Vec<Layout>, dtype_buffer: ByteRange) -> Self {
        Self {
            id: INLINE_SCHEMA_LAYOUT_ID,
            buffers: Some(vec![dtype_buffer]),
            children: Some(children),
            metadata: None,
        }
    }
}

impl WriteFlatBuffer for Layout {
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
                metadata,
            },
        )
    }
}
