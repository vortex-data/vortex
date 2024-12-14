use bytes::{Buf, Bytes};
use flatbuffers::{root, root_unchecked};
use itertools::Itertools;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_flatbuffers::message as fb;
use vortex_flatbuffers::message::{MessageHeader, MessageVersion};

use crate::messages::IPCMessage;

enum State {
    ReadingLength,
    ReadingHeader(usize),
    ReadingArray(ReadingArray),
    ReadingBuffer(ReadingBuffer),
}

struct ReadingArray {
    header: Bytes,
    buffers_length: usize,
}

struct ReadingBuffer {
    length: usize,
    length_with_padding: usize,
}

pub enum NextMessage {
    Some(IPCMessage),
    NeedMore(usize),
}

pub struct MessageReader {
    state: State,
}

impl MessageReader {
    /// Attempt to read the next message from the bytes object.
    /// If the message is incomplete, the function will return `NeedMore` with the _total_ number
    /// of bytes needed. The next call to read_next _should_ provide at least this number of bytes.
    pub fn read_next(&mut self, mut bytes: Bytes) -> VortexResult<NextMessage> {
        match &self.state {
            State::ReadingLength => {
                if bytes.len() < 4 {
                    return Ok(NextMessage::NeedMore(4));
                }
                let msg_length = bytes.get_u32_le();
                self.state = State::ReadingHeader(msg_length as usize);
                Ok(NextMessage::NeedMore(msg_length as usize))
            }
            State::ReadingHeader(msg_length) => {
                if bytes.len() < *msg_length {
                    return Ok(NextMessage::NeedMore(*msg_length));
                }
                let msg_bytes = bytes.split_to(*msg_length);

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

                        self.state = State::ReadingArray(ReadingArray {
                            header: msg_bytes,
                            buffers_length,
                        });
                        Ok(NextMessage::NeedMore(buffers_length))
                    }
                    MessageHeader::Buffer => {
                        let buffer = msg.header_as_buffer().vortex_expect("buffer header");
                        let length = buffer.length() as usize;
                        let length_with_padding = length + buffer.padding() as usize;

                        self.state = State::ReadingBuffer(ReadingBuffer {
                            length,
                            length_with_padding,
                        });
                        Ok(NextMessage::NeedMore(length_with_padding))
                    }
                    MessageHeader::DType => {
                        let dtype = msg.header_as_dtype().vortex_expect("dtype header");

                        self.state = State::ReadingLength;
                        Ok(NextMessage::Some(IPCMessage::DType(DType::try_from(
                            dtype,
                        )?)))
                    }
                    _ => {
                        vortex_bail!("Unsupported message header type {:?}", msg.header_type());
                    }
                }
            }
            State::ReadingBuffer(ReadingBuffer {
                length,
                length_with_padding,
            }) => {
                if bytes.len() < *length_with_padding {
                    return Ok(NextMessage::NeedMore(*length_with_padding));
                }

                let buffer = bytes.split_to(*length);
                let msg = IPCMessage::Buffer(Buffer::from(buffer));
                let _padding = bytes.split_to(length_with_padding - length);
                self.state = State::ReadingLength;
                Ok(NextMessage::Some(msg))
            }
            State::ReadingArray(ReadingArray {
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

                let mut all_buffers = bytes.split_to(*buffers_length);
                let _buffers = array_data_msg
                    .buffers()
                    .unwrap_or_default()
                    .iter()
                    .map(|buffer_msg| {
                        let buffer = all_buffers.split_to(buffer_msg.length() as usize);
                        let _padding = all_buffers.split_to(buffer_msg.padding() as usize);
                        buffer
                    })
                    .collect_vec();

                todo!()
            }
        }
    }
}
