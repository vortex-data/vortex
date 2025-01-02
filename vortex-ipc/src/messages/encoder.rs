use bytes::Bytes;
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use itertools::Itertools;
use vortex_array::stats::ArrayStatistics;
use vortex_array::{flatbuffers as fba, ArrayData};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexExpect};
use vortex_flatbuffers::{message as fb, FlatBufferRoot, WriteFlatBuffer};

use crate::ALIGNMENT;

/// An IPC message ready to be passed to the encoder.
pub enum EncoderMessage<'a> {
    Array(&'a ArrayData),
    Buffer(&'a ByteBuffer),
    DType(&'a DType),
}

pub struct MessageEncoder {
    /// The alignment used for each message and buffer.
    /// TODO(ngates): I'm not sure we need to include this much padding in the stream itself.
    alignment: usize,
    /// The current position in the stream. Used to calculate leading padding.
    pos: usize,
    /// A reusable buffer of zeros used for padding.
    zeros: Bytes,
}

impl Default for MessageEncoder {
    fn default() -> Self {
        Self::new(ALIGNMENT)
    }
}

impl MessageEncoder {
    /// Create a new message encoder that pads each message and buffer with the given alignment.
    ///
    /// ## Panics
    ///
    /// Panics if `alignment` is greater than `u16::MAX` or is not a power of 2.
    pub fn new(alignment: usize) -> Self {
        // We guarantee that alignment fits inside u16.
        u16::try_from(alignment).vortex_expect("Alignment must fit into u16");
        if !alignment.is_power_of_two() {
            vortex_panic!("Alignment must be a power of 2");
        }

        Self {
            alignment,
            pos: 0,
            zeros: Bytes::from(vec![0; alignment]),
        }
    }

    /// Encode an IPC message for writing to a byte stream.
    ///
    /// The returned buffers should be written contiguously to the stream.
    pub fn encode(&mut self, message: EncoderMessage) -> Vec<Bytes> {
        let mut buffers = vec![];
        assert_eq!(
            self.pos.next_multiple_of(self.alignment),
            self.pos,
            "pos must be aligned at start of a message"
        );

        // We'll push one buffer as a placeholder for the flatbuffer message length, and one
        // for the flatbuffer itself.
        buffers.push(self.zeros.clone());
        buffers.push(self.zeros.clone());

        // We initialize the flatbuffer builder with a 4-byte vector that we will use to store
        // the flatbuffer length into. By passing this vector into the FlatBufferBuilder, the
        // flatbuffers internal alignment mechanisms will handle everything else for us.
        // TODO(ngates): again, this a ton of padding...
        let mut fbb = FlatBufferBuilder::from_vec(vec![0u8; 4]);

        let header = match message {
            EncoderMessage::Array(array) => {
                let row_count = array.len();
                let array_def = ArrayWriter {
                    array,
                    buffer_idx: 0,
                }
                .write_flatbuffer(&mut fbb);

                let mut fb_buffers = vec![];
                for child in array.depth_first_traversal() {
                    if let Some(buffer) = child.byte_buffer() {
                        let end_excl_padding = self.pos + buffer.len();
                        let end_incl_padding = end_excl_padding.next_multiple_of(self.alignment);
                        let padding = u16::try_from(end_incl_padding - end_excl_padding)
                            .vortex_expect("We know padding fits into u16");
                        fb_buffers.push(fba::Buffer::create(
                            &mut fbb,
                            &fba::BufferArgs {
                                length: buffer.len() as u64,
                                padding,
                                alignment_: buffer.alignment().into(),
                            },
                        ));
                        buffers.push(buffer.clone().into_inner());
                        if padding > 0 {
                            buffers.push(self.zeros.slice(0..usize::from(padding)));
                        }
                    }
                }
                let fb_buffers = fbb.create_vector(&fb_buffers);

                fba::ArrayData::create(
                    &mut fbb,
                    &fba::ArrayDataArgs {
                        array: Some(array_def),
                        row_count: row_count as u64,
                        buffers: Some(fb_buffers),
                    },
                )
                .as_union_value()
            }
            EncoderMessage::Buffer(buffer) => {
                let end_incl_padding = buffer.len().next_multiple_of(self.alignment);
                let padding = u16::try_from(end_incl_padding - buffer.len())
                    .vortex_expect("We know padding fits into u16");
                buffers.push(buffer.clone().into_inner());
                if padding > 0 {
                    buffers.push(self.zeros.slice(0..usize::from(padding)));
                }
                fba::Buffer::create(
                    &mut fbb,
                    &fba::BufferArgs {
                        length: buffer.len() as u64,
                        padding,
                        // Buffer messages have no minimum alignment, the reader decides.
                        alignment_: 0,
                    },
                )
                .as_union_value()
            }
            EncoderMessage::DType(dtype) => dtype.write_flatbuffer(&mut fbb).as_union_value(),
        };

        let mut msg = fb::MessageBuilder::new(&mut fbb);
        msg.add_version(Default::default());
        msg.add_header_type(match message {
            EncoderMessage::Array(_) => fb::MessageHeader::ArrayData,
            EncoderMessage::Buffer(_) => fb::MessageHeader::Buffer,
            EncoderMessage::DType(_) => fb::MessageHeader::DType,
        });
        msg.add_header(header);
        let msg = msg.finish();

        // Finish the flatbuffer and swap it out for the placeholder buffer.
        fbb.finish_minimal(msg);
        let (mut fbv, pos) = fbb.collapse();

        // Add some padding to the flatbuffer vector to ensure it is aligned.
        // Note that we have to include the 4-byte length prefix in the alignment calculation.
        let unaligned_len = fbv.len() - pos + 4;
        let padding = unaligned_len.next_multiple_of(self.alignment) - unaligned_len;
        fbv.extend_from_slice(&self.zeros[0..padding]);
        let fbv_len = fbv.len();
        let fb_buffer = Bytes::from(fbv).slice(pos..fbv_len);

        let fb_buffer_len = u32::try_from(fb_buffer.len())
            .vortex_expect("IPC flatbuffer headers must fit into u32 bytes");
        buffers[0] = Bytes::from(fb_buffer_len.to_le_bytes().to_vec());
        buffers[1] = fb_buffer;

        // Update the write cursor.
        self.pos += buffers.iter().map(|b| b.len()).sum::<usize>();

        buffers
    }
}

struct ArrayWriter<'a> {
    array: &'a ArrayData,
    buffer_idx: u16,
}

impl FlatBufferRoot for ArrayWriter<'_> {}

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
        let child_buffer_idx = self.buffer_idx
            + if self.array.byte_buffer().is_some() {
                1
            } else {
                0
            };

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
            .byte_buffer()
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
