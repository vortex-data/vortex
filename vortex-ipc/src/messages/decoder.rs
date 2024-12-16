use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use bytes::{Buf, BytesMut};
use flatbuffers::{root, root_unchecked, Follow};
use itertools::Itertools;
use vortex_array::{flatbuffers as fba, ArrayData, Context};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_flatbuffers::message as fb;
use vortex_flatbuffers::message::{MessageHeader, MessageVersion};

/// A message decoded from an IPC stream.
///
/// Note that the `Array` variant cannot fully decode into an [`ArrayData`] without a [`Context`]
/// and a [`DType`]. As such, we partially decode into an [`ArrayParts`] and allow the caller to
/// finish the decoding.
#[derive(Debug)]
pub enum DecoderMessage {
    Array(ArrayParts),
    Buffer(Buffer),
    DType(DType),
}

/// ArrayParts represents a partially decoded Vortex array.
/// It can be completely decoded calling `into_array_data` with a context and dtype.
pub struct ArrayParts {
    row_count: usize,
    // Typed as fb::Array
    array_flatbuffer: Buffer,
    array_flatbuffer_loc: usize,
    buffers: Vec<Buffer>,
}

impl Debug for ArrayParts {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayComponents")
            .field("row_count", &self.row_count)
            .field("array_flatbuffer", &self.array_flatbuffer.len())
            .field("buffers", &self.buffers.len())
            .finish()
    }
}

impl ArrayParts {
    pub fn into_array_data(self, ctx: Arc<Context>, dtype: DType) -> VortexResult<ArrayData> {
        ArrayData::try_new_viewed(
            ctx,
            dtype,
            self.row_count,
            self.array_flatbuffer,
            // SAFETY: ArrayComponents guarantees the buffers are valid.
            |buf| unsafe { Ok(fba::Array::follow(buf, self.array_flatbuffer_loc)) },
            self.buffers,
        )
    }
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
    header: Buffer,
    buffers_length: usize,
}

struct ReadingBuffer {
    length: usize,
    length_with_padding: usize,
}

pub enum NextMessage {
    Some(DecoderMessage),
    NeedMore(usize),
}

/// A stateful reader for decoding IPC messages from an arbitrary stream of bytes.
#[derive(Default)]
pub struct MessageDecoder {
    state: State,
}

impl MessageDecoder {
    /// Attempt to read the next message from the bytes object.
    /// If the message is incomplete, the function will return `NeedMore` with the _total_ number
    /// of bytes needed. The next call to read_next _should_ provide at least this number of bytes.
    pub fn read_next(&mut self, bytes: &mut BytesMut) -> VortexResult<NextMessage> {
        match &self.state {
            State::Length => {
                if bytes.len() < 4 {
                    return Ok(NextMessage::NeedMore(4));
                }
                let msg_length = bytes.get_u32_le();
                self.state = State::Header(msg_length as usize);
                Ok(NextMessage::NeedMore(msg_length as usize))
            }
            State::Header(msg_length) => {
                if bytes.len() < *msg_length {
                    return Ok(NextMessage::NeedMore(*msg_length));
                }
                let mut msg_bytes = bytes.split_to(*msg_length);

                let msg = root::<fb::Message>(msg_bytes.as_ref())?;
                if msg.version() != MessageVersion::V0 {
                    vortex_bail!("Unsupported message version {:?}", msg.version());
                }

                match msg.header_type() {
                    MessageHeader::ArrayData => {
                        let array_data = msg
                            .header_as_array_data()
                            .vortex_expect("array data header");
                        let buffers_length: u64 = array_data
                            .buffers()
                            .unwrap_or_default()
                            .iter()
                            .map(|buffer| buffer.length() + (buffer.padding() as u64))
                            .sum();

                        let buffers_length = usize::try_from(buffers_length)
                            .map_err(|_| vortex_err!("buffers length is too large for usize"))?;

                        self.state = State::Array(ReadingArray {
                            header: Buffer::from(msg_bytes.split().freeze()),
                            buffers_length,
                        });
                        Ok(NextMessage::NeedMore(buffers_length))
                    }
                    MessageHeader::Buffer => {
                        let buffer = msg.header_as_buffer().vortex_expect("buffer header");
                        let length = usize::try_from(buffer.length())
                            .vortex_expect("Buffer length is too large for usize");
                        let length_with_padding = length + buffer.padding() as usize;

                        self.state = State::Buffer(ReadingBuffer {
                            length,
                            length_with_padding,
                        });
                        Ok(NextMessage::NeedMore(length_with_padding))
                    }
                    MessageHeader::DType => {
                        let dtype = msg.header_as_dtype().vortex_expect("dtype header");

                        self.state = State::Length;
                        Ok(NextMessage::Some(DecoderMessage::DType(DType::try_from(
                            dtype,
                        )?)))
                    }
                    _ => {
                        vortex_bail!("Unsupported message header type {:?}", msg.header_type());
                    }
                }
            }
            State::Buffer(ReadingBuffer {
                length,
                length_with_padding,
            }) => {
                if bytes.len() < *length_with_padding {
                    return Ok(NextMessage::NeedMore(*length_with_padding));
                }

                let buffer = bytes.split_to(*length);
                let msg = DecoderMessage::Buffer(Buffer::from(buffer.freeze()));
                let _padding = bytes.split_to(length_with_padding - length);
                self.state = State::Length;
                Ok(NextMessage::Some(msg))
            }
            State::Array(ReadingArray {
                header,
                buffers_length,
            }) => {
                if bytes.len() < *buffers_length {
                    return Ok(NextMessage::NeedMore(*buffers_length));
                }

                // SAFETY: we've already validated the header
                let msg = unsafe { root_unchecked::<fb::Message>(header.as_ref()) };
                let array_data_msg = msg
                    .header_as_array_data()
                    .vortex_expect("array data header");
                let array_msg = array_data_msg
                    .array()
                    .ok_or_else(|| vortex_err!("array data message missing array"))?;

                let mut all_buffers = bytes.split_to(*buffers_length);
                let buffers = array_data_msg
                    .buffers()
                    .unwrap_or_default()
                    .iter()
                    .map(|buffer_msg| {
                        let buffer_len = usize::try_from(buffer_msg.length())
                            .vortex_expect("buffer length is too large for usize");
                        let buffer = all_buffers.split_to(buffer_len);
                        let _padding = all_buffers.split_to(buffer_msg.padding() as usize);
                        Buffer::from(buffer.freeze())
                    })
                    .collect_vec();

                let row_count = usize::try_from(array_data_msg.row_count())
                    .map_err(|_| vortex_err!("row count is too large for usize"))?;

                let msg = DecoderMessage::Array(ArrayParts {
                    row_count,
                    array_flatbuffer: header.clone(),
                    array_flatbuffer_loc: array_msg._tab.loc(),
                    buffers,
                });
                self.state = State::Length;
                Ok(NextMessage::Some(msg))
            }
        }
    }
}
