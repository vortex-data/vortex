// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bytes::Bytes;
use bytes::BytesMut;
use vortex_array::ArrayContext;
use vortex_array::DynArray;
use vortex_array::dtype::DType;
use vortex_array::serde::SerializeOptions;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBufferBuilder;
use vortex_flatbuffers::WriteFlatBufferExt;
use vortex_flatbuffers::message as fb;

/// An IPC message ready to be passed to the encoder.
pub enum EncoderMessage<'a> {
    Array(&'a dyn DynArray),
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
    pub fn encode(&mut self, message: EncoderMessage) -> VortexResult<Vec<Bytes>> {
        let mut buffers = vec![];

        // We'll push one buffer as a placeholder for the flatbuffer message length, and one
        // for the flatbuffer itself.
        buffers.push(self.zeros.clone());
        buffers.push(self.zeros.clone());

        let mut fbb = FlatBufferBuilder::new();

        let (header, body_len) = match message {
            EncoderMessage::Array(array) => {
                // Currently we include a Context in every message. We could convert this to
                // sending deltas later.
                let ctx = ArrayContext::empty();

                let array_buffers = array.serialize(&ctx, &SerializeOptions::default())?;
                let body_len = array_buffers.iter().map(|b| b.len() as u64).sum::<u64>();

                let array_encodings = ctx
                    .to_ids()
                    .iter()
                    .map(|e| fbb.create_string(e.as_ref()))
                    .collect::<Vec<_>>();
                let array_encodings = fbb.create_vector(array_encodings.as_slice());

                let header = fb::MessageHeader::builder()
                    .array_message(fb::ArrayMessage::create(
                        &mut fbb,
                        u32::try_from(array.len())
                            .map_err(|_| vortex_err!("Array length must fit into u32"))?,
                        Some(array_encodings),
                    ))
                    .finish(&mut fbb);

                buffers.extend(array_buffers.into_iter().map(|b| b.into_inner()));

                (header, body_len)
            }
            EncoderMessage::Buffer(buffer) => {
                let header = fb::MessageHeader::builder()
                    .buffer_message(fb::BufferMessage::create(
                        &mut fbb,
                        buffer.alignment().exponent(),
                    ))
                    .finish(&mut fbb);
                let body_len = buffer.len() as u64;
                buffers.push(buffer.clone().into_inner());

                (header, body_len)
            }
            EncoderMessage::DType(dtype) => {
                let header = fb::MessageHeader::builder()
                    .d_type_message(fb::DTypeMessage::create(&mut fbb))
                    .finish(&mut fbb);

                let buffer = dtype.write_flatbuffer_bytes()?.into_inner().into_inner();
                let body_len = buffer.len() as u64;
                buffers.push(buffer);

                (header, body_len)
            }
        };

        let msg = fb::Message::create(&mut fbb, fb::MessageVersion::V0, Some(header), body_len);

        // Finish the flatbuffer and swap it out for the placeholder buffer.
        let fb_buffer = fbb.finish(msg, None);
        let fb_buffer_len = u32::try_from(fb_buffer.len())
            .map_err(|_| vortex_err!("Array flatbuffer length must fit into u32"))?;

        buffers[0] = Bytes::copy_from_slice(&fb_buffer_len.to_le_bytes());
        buffers[1] = Bytes::copy_from_slice(fb_buffer);

        Ok(buffers)
    }
}
