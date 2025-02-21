use bytes::{Bytes, BytesMut};
use flatbuffers::FlatBufferBuilder;
use vortex_array::serde::SerializeOptions;
use vortex_array::Array;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_flatbuffers::{message as fb, FlatBuffer, WriteFlatBufferExt};

/// An IPC message ready to be passed to the encoder.
pub enum EncoderMessage<'a> {
    Array(&'a dyn Array),
    Buffer(&'a ByteBuffer),
    DType(&'a DType),
}

pub struct MessageEncoder {
    /// A reusable buffer of zeros used for padding.
    zeros: Bytes,
}

impl Default for MessageEncoder {
    fn default() -> Self {
        Self {
            zeros: BytesMut::zeroed(u16::MAX as usize).freeze(),
        }
    }
}

impl MessageEncoder {
    /// Encode an IPC message for writing to a byte stream.
    ///
    /// The returned buffers should be written contiguously to the stream.
    pub fn encode(&mut self, message: EncoderMessage) -> Vec<Bytes> {
        let mut buffers = vec![];

        // We'll push one buffer as a placeholder for the flatbuffer message length, and one
        // for the flatbuffer itself.
        buffers.push(self.zeros.clone());
        buffers.push(self.zeros.clone());

        // We initialize the flatbuffer builder with a 4-byte vector that we will use to store
        // the flatbuffer length into. By passing this vector into the FlatBufferBuilder, the
        // flatbuffers internal alignment mechanisms will handle everything else for us.
        // TODO(ngates): again, this a ton of padding...
        let mut fbb = FlatBufferBuilder::from_vec(vec![0u8; 4]);

        let (header, body_len) = match message {
            EncoderMessage::Array(array) => {
                let header = fb::ArrayMessage::create(
                    &mut fbb,
                    &fb::ArrayMessageArgs {
                        row_count: u32::try_from(array.len())
                            .vortex_expect("Array length must fit into u32"),
                    },
                )
                .as_union_value();
                let array_buffers = array.serialize(&SerializeOptions::default());
                let body_len = array_buffers.iter().map(|b| b.len() as u64).sum::<u64>();
                buffers.extend(array_buffers.into_iter().map(|b| b.into_inner()));

                (header, body_len)
            }
            EncoderMessage::Buffer(buffer) => {
                let header = fb::BufferMessage::create(
                    &mut fbb,
                    &fb::BufferMessageArgs {
                        alignment_exponent: buffer.alignment().exponent(),
                    },
                )
                .as_union_value();
                let body_len = buffer.len() as u64;
                buffers.push(buffer.clone().into_inner());

                (header, body_len)
            }
            EncoderMessage::DType(dtype) => {
                let header =
                    fb::DTypeMessage::create(&mut fbb, &fb::DTypeMessageArgs {}).as_union_value();

                let buffer = dtype.write_flatbuffer_bytes().into_inner().into_inner();
                let body_len = buffer.len() as u64;
                buffers.push(buffer);

                (header, body_len)
            }
        };

        let mut msg = fb::MessageBuilder::new(&mut fbb);
        msg.add_version(Default::default());
        msg.add_header_type(match message {
            EncoderMessage::Array(_) => fb::MessageHeader::ArrayMessage,
            EncoderMessage::Buffer(_) => fb::MessageHeader::BufferMessage,
            EncoderMessage::DType(_) => fb::MessageHeader::DTypeMessage,
        });
        msg.add_header(header);
        msg.add_body_size(body_len);
        let msg = msg.finish();

        // Finish the flatbuffer and swap it out for the placeholder buffer.
        fbb.finish_minimal(msg);
        let (fbv, pos) = fbb.collapse();
        let fb_buffer = FlatBuffer::copy_from(&fbv[pos..]);
        let fb_buffer_len = u32::try_from(fb_buffer.len())
            .vortex_expect("IPC flatbuffer headers must fit into u32 bytes");

        buffers[0] = Bytes::from(fb_buffer_len.to_le_bytes().to_vec());
        buffers[1] = fb_buffer.into_inner().into_inner();

        buffers
    }
}
