use flatbuffers::{FlatBufferBuilder, WIPOffset};
use itertools::Itertools;
use vortex_array::stats::ArrayStatistics;
use vortex_array::{flatbuffers as fba, ArrayData};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_flatbuffers::{message as fb, FlatBufferRoot, WriteFlatBuffer};

use crate::ALIGNMENT;

pub mod reader;
pub mod writer;

pub enum IPCMessage {
    Array(ArrayData),
    Buffer(Buffer),
    DType(DType),
}

impl FlatBufferRoot for IPCMessage {}

impl WriteFlatBuffer for IPCMessage {
    type Target<'a> = fb::Message<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let header = match self {
            Self::Array(array) => ArrayDataWriter { array }
                .write_flatbuffer(fbb)
                .as_union_value(),
            Self::Buffer(buffer) => {
                let aligned_len = buffer.len().next_multiple_of(ALIGNMENT);
                let padding = aligned_len - buffer.len();
                fba::Buffer::create(
                    fbb,
                    &fba::BufferArgs {
                        length: buffer.len() as u64,
                        padding: padding.try_into().vortex_expect("padding must fit in u16"),
                    },
                )
                .as_union_value()
            }
            Self::DType(dtype) => dtype.write_flatbuffer(fbb).as_union_value(),
        };

        let mut msg = fb::MessageBuilder::new(fbb);
        msg.add_version(Default::default());
        msg.add_header_type(match self {
            Self::Array(_) => fb::MessageHeader::ArrayData,
            Self::Buffer(_) => fb::MessageHeader::Buffer,
            Self::DType(_) => fb::MessageHeader::DType,
        });
        msg.add_header(header);
        msg.finish()
    }
}

struct ArrayDataWriter<'a> {
    array: &'a ArrayData,
}

impl WriteFlatBuffer for ArrayDataWriter<'_> {
    type Target<'t> = fba::ArrayData<'t>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let array = Some(
            ArrayWriter {
                array: self.array,
                buffer_idx: 0,
            }
            .write_flatbuffer(fbb),
        );

        // Walk the ColumnData depth-first to compute the buffer lengths.
        let mut buffers = vec![];
        for array_data in self.array.depth_first_traversal() {
            if let Some(buffer) = array_data.buffer() {
                let aligned_size = buffer.len().next_multiple_of(ALIGNMENT);
                let padding = aligned_size - buffer.len();
                buffers.push(fba::Buffer::create(
                    fbb,
                    &fba::BufferArgs {
                        length: buffer.len() as u64,
                        padding: padding.try_into().vortex_expect("padding must fit in u16"),
                    },
                ));
            }
        }
        let buffers = Some(fbb.create_vector(&buffers));

        fba::ArrayData::create(
            fbb,
            &fba::ArrayDataArgs {
                array,
                row_count: self.array.len() as u64,
                buffers,
            },
        )
    }
}

struct ArrayWriter<'a> {
    array: &'a ArrayData,
    buffer_idx: u16,
}

impl WriteFlatBuffer for ArrayWriter<'_> {
    type Target<'t> = fba::Array<'t>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let encoding = self.array.encoding().id().code();
        let metadata = self
            .array
            .metadata_bytes()
            .vortex_expect("IPCArray is missing metadata during serialization");
        let metadata = Some(fbb.create_vector(metadata.as_ref()));

        // Assign buffer indices for all child arrays.
        // The second tuple element holds the buffer_index for this Array subtree. If this array
        // has a buffer, that is its buffer index. If it does not, that buffer index belongs
        // to one of the children.
        let child_buffer_idx = self.buffer_idx + if self.array.buffer().is_some() { 1 } else { 0 };

        let children = self
            .array
            .children()
            .iter()
            .scan(child_buffer_idx, |buffer_idx, child| {
                // Update the number of buffers required.
                let msg = ArrayWriter {
                    array: child,
                    buffer_idx: *buffer_idx,
                }
                .write_flatbuffer(fbb);
                *buffer_idx = u16::try_from(child.cumulative_nbuffers())
                    .ok()
                    .and_then(|nbuffers| nbuffers.checked_add(*buffer_idx))
                    .vortex_expect("Too many buffers (u16) for ArrayData");
                Some(msg)
            })
            .collect_vec();
        let children = Some(fbb.create_vector(&children));

        let buffers = self
            .array
            .buffer()
            .is_some()
            .then_some(self.buffer_idx)
            .map(|buffer_idx| fbb.create_vector_from_iter(std::iter::once(buffer_idx)));

        let stats = Some(self.array.statistics().write_flatbuffer(fbb));

        fba::Array::create(
            fbb,
            &fba::ArrayArgs {
                encoding,
                metadata,
                children,
                buffers,
                stats,
            },
        )
    }
}
