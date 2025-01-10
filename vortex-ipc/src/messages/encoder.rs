use bytes::Bytes;
use flatbuffers::FlatBufferBuilder;
use vortex_array::parts::ArrayPartsFlatBuffer;
use vortex_array::ArrayData;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexExpect};
use vortex_flatbuffers::{message as fb, WriteFlatBuffer};

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
                let fb_array = ArrayPartsFlatBuffer::new(array).write_flatbuffer(&mut fbb);

                let mut fb_buffers = vec![];
                for child in array.depth_first_traversal() {
                    for buffer in child.byte_buffers() {
                        let end_excl_padding = self.pos + buffer.len();
                        let end_incl_padding = end_excl_padding.next_multiple_of(self.alignment);
                        let padding = u16::try_from(end_incl_padding - end_excl_padding)
                            .vortex_expect("We know padding fits into u16");
                        fb_buffers.push(fb::Buffer::create(
                            &mut fbb,
                            &fb::BufferArgs {
                                length: buffer.len() as u64,
                                padding,
                                alignment: buffer.alignment().into(),
                            },
                        ));
                        buffers.push(buffer.clone().into_inner());
                        if padding > 0 {
                            buffers.push(self.zeros.slice(0..usize::from(padding)));
                        }
                    }
                }
                let fb_buffers = fbb.create_vector(&fb_buffers);

                fb::ArrayMessage::create(
                    &mut fbb,
                    &fb::ArrayMessageArgs {
                        array: Some(fb_array),
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
                fb::Buffer::create(
                    &mut fbb,
                    &fb::BufferArgs {
                        length: buffer.len() as u64,
                        padding,
                        // Buffer messages have no minimum alignment, the reader decides.
                        alignment: 0,
                    },
                )
                .as_union_value()
            }
            EncoderMessage::DType(dtype) => dtype.write_flatbuffer(&mut fbb).as_union_value(),
        };

        let mut msg = fb::MessageBuilder::new(&mut fbb);
        msg.add_version(Default::default());
        msg.add_header_type(match message {
            EncoderMessage::Array(_) => fb::MessageHeader::ArrayMessage,
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
