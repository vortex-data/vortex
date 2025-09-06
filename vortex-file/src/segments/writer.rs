// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{ready, Context, Poll};

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{vortex_err, VortexResult};
use vortex_layout::segments::{SegmentId, SegmentSink};
use vortex_layout::sequence::SequenceId;

use crate::footer::SegmentSpec;

pub struct BufferedSegmentSink {
    buffers: kanal::AsyncSender<VortexResult<ByteBuffer>>,
    byte_offset: AtomicU64,
    segment_specs: Mutex<Vec<SegmentSpec>>,
}

impl BufferedSegmentSink {
    pub fn new(send: kanal::AsyncSender<VortexResult<ByteBuffer>>, byte_offset: u64) -> Self {
        Self {
            buffers: send,
            byte_offset: AtomicU64::new(byte_offset),
            segment_specs: Default::default(),
        }
    }

    pub fn to_specs(&self) -> Vec<SegmentSpec> {
        let specs = self.segment_specs.lock();
        specs.clone()
    }
}

#[async_trait]
impl SegmentSink for BufferedSegmentSink {
    async fn write(
        &self,
        mut sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId> {
        // We wait for all segment IDs before this one to be dropped. Then while we hold a strong
        // reference to this one, we essentially have an exclusive lock on the segment writer.
        sequence_id.collapse().await;

        let (segment_id, padding_bufer) = {
            let mut specs = self.segment_specs.lock();
            let segment_id = SegmentId::from(
                u32::try_from(specs.len())
                    .map_err(|_| vortex_err!("Too mant segments, u32 overflow"))?,
            );

            // The API requires us to write these buffers contiguously. Therefore, we can only
            // respect the alignment of the first one.
            // Don't worry, in most cases the caller knows what they're doing and will align the
            // buffers themselves, inserting padding buffers where necessary.
            let alignment = buffers
                .first()
                .map(|buffer| buffer.alignment())
                .unwrap_or_else(Alignment::none);
            let length = u32::try_from(buffers.iter().map(|buffer| buffer.len()).sum::<usize>())
                .map_err(|_| vortex_err!("segment buffer length exceeds maximum u32"))?;

            // Add any padding required to align the segment.
            let byte_offset = self.byte_offset.load(Ordering::Relaxed);
            let padding = byte_offset.next_multiple_of(*alignment as u64) - byte_offset;
            let offset = byte_offset + padding;
            specs.push(SegmentSpec {
                offset,
                length,
                alignment,
            });

            self.byte_offset
                .store(byte_offset + padding + u64::from(length), Ordering::Relaxed);

            // Send the buffers to the stream.
            if padding > 0 {
                (segment_id, Some(ByteBuffer::zeroed(padding as usize)))
            } else {
                (segment_id, None)
            }
        };

        if let Some(padding) = padding_bufer {
            let _ = self.buffers.send(Ok(padding)).await;
        }
        for buffer in buffers {
            let _ = self.buffers.send(Ok(buffer)).await;
        }

        Ok(segment_id)
    }
}

pub struct BufferStream {
    inner: BoxStream<'static, VortexResult<ByteBuffer>>,
    byte_offset: u64,
}

impl BufferStream {
    pub fn byte_offset(&self) -> u64 {
        self.byte_offset
    }
}

impl Stream for BufferStream {
    type Item = VortexResult<ByteBuffer>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match ready!(this.inner.poll_next_unpin(cx)) {
            Some(Ok(buffer)) => {
                this.byte_offset += buffer.len() as u64;
                Poll::Ready(Some(Ok(buffer)))
            }
            Some(Err(e)) => Poll::Ready(Some(Err(e))),
            None => Poll::Ready(None),
        }
    }
}
