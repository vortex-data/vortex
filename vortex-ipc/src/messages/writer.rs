#![allow(clippy::assertions_on_constants)]
use std::io;

use bytes::Bytes;
use flatbuffers::FlatBufferBuilder;
use vortex_array::ArrayData;
use vortex_buffer::io_buf::IoBuf;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::VortexUnwrap;
use vortex_flatbuffers::{WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::VortexWrite;

use crate::messages::IPCMessage;
use crate::ALIGNMENT;

static ZEROS: [u8; 512] = [0; 512];

#[derive(Debug)]
pub struct MessageWriter<W> {
    write: W,
    pos: u64,
    alignment: usize,

    scratch: Option<Vec<u8>>,
}

impl<W: VortexWrite> MessageWriter<W> {
    pub fn new(write: W) -> Self {
        assert!(ALIGNMENT <= ZEROS.len(), "ALIGNMENT must be <= 512");
        Self {
            write,
            pos: 0,
            alignment: ALIGNMENT,
            scratch: Some(Vec::new()),
        }
    }

    pub fn into_inner(self) -> W {
        self.write
    }

    /// Returns the current position in the stream.
    pub fn tell(&self) -> u64 {
        self.pos
    }

    pub async fn write_dtype_raw(&mut self, dtype: &DType) -> io::Result<()> {
        let buffer = dtype.write_flatbuffer_bytes();
        let written_len = buffer.len();
        self.write_all(buffer).await?;

        let aligned_size = written_len.next_multiple_of(self.alignment);
        let padding = aligned_size - written_len;

        self.write_all(Bytes::from(&ZEROS[..padding])).await?;

        Ok(())
    }

    pub async fn write_dtype(&mut self, dtype: DType) -> io::Result<()> {
        self.write_message(IPCMessage::DType(dtype)).await
    }

    pub async fn write_array(&mut self, array: ArrayData) -> io::Result<()> {
        self.write_message(IPCMessage::Array(array.clone())).await?;

        for array in array.depth_first_traversal() {
            if let Some(buffer) = array.buffer() {
                let buffer_len = buffer.len();
                let padding = buffer_len.next_multiple_of(self.alignment) - buffer_len;
                self.write_all(buffer.clone()).await?;
                self.write_all(Bytes::from(&ZEROS[..padding])).await?;
            }
        }

        Ok(())
    }

    pub async fn write_page(&mut self, buffer: Buffer) -> io::Result<()> {
        let buffer_len = buffer.len();
        let padding = buffer_len.next_multiple_of(self.alignment) - buffer_len;
        self.write_message(IPCMessage::Buffer(buffer.clone()))
            .await?;
        self.write_all(buffer).await?;
        self.write_all(Bytes::from(&ZEROS[..padding])).await?;

        Ok(())
    }

    pub async fn write_message<F: WriteFlatBuffer>(&mut self, flatbuffer: F) -> io::Result<()> {
        // We reuse the scratch buffer each time and then replace it at the end.
        // The scratch buffer may be missing if a previous write failed. We could use scopeguard
        // or similar here if it becomes a problem in practice.
        let mut scratch = self.scratch.take().unwrap_or_default();
        scratch.clear();

        // In order for FlatBuffers to use the correct alignment, we insert 4 bytes at the start
        // of the flatbuffer vector since we will be writing this to the stream later.
        scratch.extend_from_slice(&[0_u8; 4]);

        let mut fbb = FlatBufferBuilder::from_vec(scratch);
        let root = flatbuffer.write_flatbuffer(&mut fbb);
        fbb.finish_minimal(root);

        let (buffer, buffer_begin) = fbb.collapse();
        let buffer_end = buffer.len();
        let buffer_len = buffer_end - buffer_begin;

        let unaligned_size = 4 + buffer_len;
        let aligned_size = (unaligned_size + (self.alignment - 1)) & !(self.alignment - 1);
        let padding = aligned_size - unaligned_size;

        // Write the size as u32, followed by the buffer, followed by padding.
        self.write_all(
            u32::try_from(aligned_size - 4)
                .vortex_unwrap()
                .to_le_bytes(),
        )
        .await?;
        let buffer = self
            .write_all(buffer.slice_owned(buffer_begin..buffer_end))
            .await?
            .into_inner();
        self.write_all(Bytes::from(&ZEROS[..padding])).await?;

        assert_eq!(self.pos % self.alignment as u64, 0);

        // Replace the scratch buffer
        self.scratch = Some(buffer);

        Ok(())
    }

    async fn write_all<B: IoBuf>(&mut self, buf: B) -> io::Result<B> {
        let buf = self.write.write_all(buf).await?;
        self.pos += buf.bytes_init() as u64;
        Ok(buf)
    }
}
