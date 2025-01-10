use std::fmt::Debug;

use bytes::Buf;
use flatbuffers::{root, root_unchecked};
use itertools::Itertools;
use vortex_array::parts::ArrayParts;
use vortex_buffer::{AlignedBuf, Alignment, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_flatbuffers::message::{MessageHeader, MessageVersion};
use vortex_flatbuffers::{message as fb, FlatBuffer};

use crate::ALIGNMENT;

/// A message decoded from an IPC stream.
///
/// Note that the `Array` variant cannot fully decode into an [`vortex_array::ArrayData`] without
/// a [`vortex_array::ContextRef`] and a [`DType`]. As such, we partially decode into an
/// [`ArrayParts`] and allow the caller to finish the decoding.
#[derive(Debug)]
pub enum DecoderMessage {
    Array(ArrayParts),
    Buffer(ByteBuffer),
    DType(DType),
}

#[derive(Default)]
enum State {
    #[default]
    Length,
    Header(usize),
    Array(ReadingArray),
    Buffer(ReadingBuffer),
}

struct ReadingArray {
    header: FlatBuffer,
    buffers_length: usize,
}

struct ReadingBuffer {
    length: usize,
    length_with_padding: usize,
    alignment: Alignment,
}

#[derive(Debug)]
pub enum PollRead {
    Some(DecoderMessage),
    /// Returns the _total_ number of bytes needed to make progress.
    /// Note this is _not_ the incremental number of bytes needed to make progress.
    NeedMore(usize),
}

// NOTE(ngates): we should design some trait that the Decoder can take that doesn't require unique
//  ownership of the underlying bytes. The decoder needs to split out bytes, and advance a cursor,
//  but it doesn't need to mutate any bytes. So in theory, we should be able to do this zero-copy
//  over a shared buffer of bytes, instead of requiring a `BytesMut`.
/// A stateful reader for decoding IPC messages from an arbitrary stream of bytes.
pub struct MessageDecoder {
    /// The minimum alignment to use when reading a data `Buffer`.
    alignment: Alignment,
    /// The current state of the decoder.
    state: State,
}

impl Default for MessageDecoder {
    fn default() -> Self {
        Self {
            alignment: ALIGNMENT.into(),
            state: Default::default(),
        }
    }
}

impl MessageDecoder {
    /// Attempt to read the next message from the bytes object.
    ///
    /// If the message is incomplete, the function will return `NeedMore` with the _total_ number
    /// of bytes needed to make progress. The next call to read_next _should_ provide at least
    /// this number of bytes otherwise it will be given the same `NeedMore` response.
    pub fn read_next<B: AlignedBuf>(&mut self, bytes: &mut B) -> VortexResult<PollRead> {
        loop {
            match &self.state {
                State::Length => {
                    if bytes.remaining() < 4 {
                        return Ok(PollRead::NeedMore(4));
                    }

                    let msg_length = bytes.get_u32_le();
                    self.state = State::Header(msg_length as usize);
                }
                State::Header(msg_length) => {
                    if bytes.remaining() < *msg_length {
                        return Ok(PollRead::NeedMore(*msg_length));
                    }

                    let msg_bytes = bytes.copy_to_const_aligned(*msg_length);
                    let msg = root::<fb::Message>(msg_bytes.as_ref())?;
                    if msg.version() != MessageVersion::V0 {
                        vortex_bail!("Unsupported message version {:?}", msg.version());
                    }

                    match msg.header_type() {
                        MessageHeader::ArrayMessage => {
                            let array_msg = msg
                                .header_as_array_message()
                                .vortex_expect("array message header");
                            let buffers_length: u64 = array_msg
                                .buffers()
                                .unwrap_or_default()
                                .iter()
                                .map(|buffer| buffer.length() + (buffer.padding() as u64))
                                .sum();

                            let buffers_length = usize::try_from(buffers_length).map_err(|_| {
                                vortex_err!("buffers length is too large for usize")
                            })?;

                            self.state = State::Array(ReadingArray {
                                header: msg_bytes,
                                buffers_length,
                            });
                        }
                        MessageHeader::Buffer => {
                            let buffer = msg.header_as_buffer().vortex_expect("buffer header");
                            let length = usize::try_from(buffer.length())
                                .vortex_expect("Buffer length is too large for usize");
                            let length_with_padding = length + buffer.padding() as usize;

                            self.state = State::Buffer(ReadingBuffer {
                                length,
                                length_with_padding,
                                alignment: buffer.alignment().max(1).into(),
                            });
                        }
                        MessageHeader::DType => {
                            let msg_dtype = msg.header_as_dtype().vortex_expect("dtype header");
                            let dtype = DType::try_from_view(msg_dtype, msg_bytes.clone())?;

                            // Nothing else to read, so we reset the state to Length
                            self.state = Default::default();
                            return Ok(PollRead::Some(DecoderMessage::DType(dtype)));
                        }
                        _ => {
                            vortex_bail!("Unsupported message header type {:?}", msg.header_type());
                        }
                    }
                }
                State::Buffer(ReadingBuffer {
                    length,
                    length_with_padding,
                    alignment,
                }) => {
                    // Ensure the buffer is read with maximum of reader and message alignment.
                    let read_alignment = self.alignment.max(*alignment);
                    if bytes.remaining() < *length_with_padding {
                        return Ok(PollRead::NeedMore(*length_with_padding));
                    }
                    let buffer = bytes.copy_to_aligned(*length, read_alignment);

                    // Then use the buffer-requested alignment for the result.
                    let msg = DecoderMessage::Buffer(buffer.aligned(*alignment));
                    bytes.advance(length_with_padding - length);

                    // Nothing else to read, so we reset the state to Length
                    self.state = Default::default();
                    return Ok(PollRead::Some(msg));
                }
                State::Array(ReadingArray {
                    header,
                    buffers_length,
                }) => {
                    if bytes.remaining() < *buffers_length {
                        return Ok(PollRead::NeedMore(*buffers_length));
                    }

                    // SAFETY: we've already validated the header
                    let msg = unsafe { root_unchecked::<fb::Message>(header.as_ref()) };
                    let array_msg = msg
                        .header_as_array_message()
                        .vortex_expect("array message header");
                    let array = array_msg
                        .array()
                        .ok_or_else(|| vortex_err!("array data message missing array"))?;

                    let buffers = array_msg
                        .buffers()
                        .unwrap_or_default()
                        .iter()
                        .map(|buffer_msg| {
                            let buffer_len = usize::try_from(buffer_msg.length())
                                .vortex_expect("buffer length is too large for usize");
                            let buffer_alignment = Alignment::from(buffer_msg.alignment().max(1));

                            // Ensure the buffer is read with maximum of reader and message alignment.
                            let read_alignment = self.alignment.max(buffer_alignment);
                            let buffer = bytes.copy_to_aligned(buffer_len, read_alignment);
                            bytes.advance(buffer_msg.padding() as usize);
                            // But use the buffer-requested alignment for the result.
                            buffer.aligned(buffer_alignment)
                        })
                        .collect_vec();

                    let row_count = usize::try_from(array_msg.row_count())
                        .map_err(|_| vortex_err!("row count is too large for usize"))?;

                    let msg = DecoderMessage::Array(ArrayParts::new(
                        row_count,
                        array,
                        header.clone(),
                        buffers,
                    ));

                    self.state = Default::default();
                    return Ok(PollRead::Some(msg));
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use bytes::BytesMut;
    use vortex_array::array::ConstantArray;
    use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
    use vortex_buffer::buffer;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::messages::{EncoderMessage, MessageEncoder};

    fn write_and_read(expected: ArrayData) {
        let mut ipc_bytes = BytesMut::new();
        let mut encoder = MessageEncoder::default();
        for buf in encoder.encode(EncoderMessage::Array(&expected)) {
            ipc_bytes.extend_from_slice(buf.as_ref());
        }

        let mut decoder = MessageDecoder::default();

        // Since we provide all bytes up-front, we should never hit a NeedMore.
        let mut buffer = BytesMut::from(ipc_bytes.as_ref());
        let array_parts = match decoder.read_next(&mut buffer).unwrap() {
            PollRead::Some(DecoderMessage::Array(array_parts)) => array_parts,
            otherwise => vortex_panic!("Expected an array, got {:?}", otherwise),
        };

        // Decode the array parts with the context
        let actual = array_parts
            .decode(Default::default(), expected.dtype().clone())
            .unwrap();

        assert_eq!(expected.len(), actual.len());
        assert_eq!(expected.encoding(), actual.encoding());
    }

    #[test]
    fn array_ipc() {
        write_and_read(buffer![0i32, 1, 2, 3].into_array());
    }

    #[test]
    fn array_no_buffers() {
        // Constant arrays have no buffers
        let array = ConstantArray::new(10i32, 20).into_array();
        assert!(
            array.byte_buffer(0).is_none(),
            "Array should have no buffers"
        );
        write_and_read(array);
    }
}
