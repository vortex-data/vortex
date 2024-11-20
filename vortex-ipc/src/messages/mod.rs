use flatbuffers::{FlatBufferBuilder, WIPOffset};
use itertools::Itertools;
use vortex_array::stats::ArrayStatistics;
use vortex_array::{flatbuffers as fba, ArrayData};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::VortexExpect as _;
use vortex_flatbuffers::message::Compression;
use vortex_flatbuffers::{message as fb, FlatBufferRoot, WriteFlatBuffer};

use crate::ALIGNMENT;

pub mod reader;
pub mod writer;

pub enum IPCMessage<'a> {
    Schema(IPCSchema<'a>),
    Batch(IPCBatch<'a>),
    Page(IPCPage<'a>),
}

pub struct IPCSchema<'a>(pub &'a DType);
pub struct IPCBatch<'a>(pub &'a ArrayData);
pub struct IPCArray<'a>(pub &'a ArrayData, usize);
pub struct IPCPage<'a>(pub &'a Buffer);

impl FlatBufferRoot for IPCMessage<'_> {}

impl WriteFlatBuffer for IPCMessage<'_> {
    type Target<'a> = fb::Message<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let header = match self {
            Self::Schema(f) => f.write_flatbuffer(fbb).as_union_value(),
            Self::Batch(f) => f.write_flatbuffer(fbb).as_union_value(),
            Self::Page(f) => f.write_flatbuffer(fbb).as_union_value(),
        };

        let mut msg = fb::MessageBuilder::new(fbb);
        msg.add_version(Default::default());
        msg.add_header_type(match self {
            Self::Schema(_) => fb::MessageHeader::Schema,
            Self::Batch(_) => fb::MessageHeader::Batch,
            Self::Page(_) => fb::MessageHeader::Page,
        });
        msg.add_header(header);
        msg.finish()
    }
}

impl WriteFlatBuffer for IPCSchema<'_> {
    type Target<'t> = fb::Schema<'t>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let dtype = Some(self.0.write_flatbuffer(fbb));
        fb::Schema::create(fbb, &fb::SchemaArgs { dtype })
    }
}

impl WriteFlatBuffer for IPCBatch<'_> {
    type Target<'t> = fb::Batch<'t>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let array_data = self.0;
        let array = Some(IPCArray(array_data, 0).write_flatbuffer(fbb));

        let length = array_data.len() as u64;

        // Walk the ColumnData depth-first to compute the buffer offsets.
        let mut buffers = vec![];
        let mut offset = 0;

        for array_data in array_data.depth_first_traversal() {
            if let Some(buffer) = array_data.buffer() {
                let aligned_size = (buffer.len() + (ALIGNMENT - 1)) & !(ALIGNMENT - 1);
                buffers.push(fb::Buffer::new(
                    offset as u64,
                    (aligned_size - buffer.len()) as u16,
                    Compression::None,
                ));
                offset += aligned_size;
            }
        }
        let buffers = Some(fbb.create_vector(&buffers));

        fb::Batch::create(
            fbb,
            &fb::BatchArgs {
                array,
                length,
                buffers,
                buffer_size: offset as u64,
            },
        )
    }
}

impl WriteFlatBuffer for IPCArray<'_> {
    type Target<'t> = fba::Array<'t>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let column_data = self.0;

        let encoding = column_data.encoding().id().code();
        let metadata = column_data
            .metadata_bytes()
            .vortex_expect("IPCArray is missing metadata during serialization");
        let metadata = Some(fbb.create_vector(metadata.as_ref()));

        // Assign buffer indices for all child arrays.
        // The second tuple element holds the buffer_index for this Array subtree. If this array
        // has a buffer, that is its buffer index. If it does not, that buffer index belongs
        // to one of the children.
        let child_buffer_offset = self.1 + if self.0.buffer().is_some() { 1 } else { 0 };

        let children = column_data
            .children()
            .iter()
            .scan(child_buffer_offset, |buffer_offset, child| {
                // Update the number of buffers required.
                let msg = IPCArray(child, *buffer_offset).write_flatbuffer(fbb);
                *buffer_offset += child.cumulative_nbuffers();
                Some(msg)
            })
            .collect_vec();
        let children = Some(fbb.create_vector(&children));

        let stats = Some(column_data.statistics().write_flatbuffer(fbb));

        fba::Array::create(
            fbb,
            &fba::ArrayArgs {
                version: Default::default(),
                buffer_index: self.0.buffer().is_some().then_some(self.1 as u64),
                encoding,
                metadata,
                stats,
                children,
            },
        )
    }
}

impl WriteFlatBuffer for IPCPage<'_> {
    type Target<'t> = fb::Page<'t>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let buffer_size = self.0.len();
        let aligned_size = (buffer_size + (ALIGNMENT - 1)) & !(ALIGNMENT - 1);
        let padding_size = aligned_size - buffer_size;

        fb::Page::create(
            fbb,
            &fb::PageArgs {
                buffer_size: buffer_size as u32,
                padding: padding_size as u16,
            },
        )
    }
}
