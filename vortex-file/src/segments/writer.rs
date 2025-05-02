#![allow(dead_code)]
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use parking_lot::Mutex;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_io::VortexWrite;
use vortex_layout::segments::{SegmentId, SegmentWriter};

use super::ordered::{OrderedBuffers, Region};
use crate::footer::SegmentSpec;

/// A segment writer that holds buffers in memory until they are flushed by a writer.
#[derive(Default)]
pub struct InOrderSegmentWriter {
    buffers: Arc<Mutex<OrderedBuffers>>,
    region: Region,
    region_offset: usize,
}

#[async_trait]
impl SegmentWriter for InOrderSegmentWriter {
    async fn put(&mut self, data: Vec<ByteBuffer>) -> VortexResult<SegmentId> {
        let buffer_idx = self.region.start + self.region_offset;
        if buffer_idx >= self.region.end {
            vortex_bail!("region space exhausted!");
        }
        self.buffers.lock().insert_buffer(buffer_idx, data);
        self.region_offset += 1;
        self.next_segment_id_once_active().await
    }
}

impl InOrderSegmentWriter {
    pub fn split(self, splits: usize) -> VortexResult<Vec<Self>> {
        Ok(self
            .buffers
            .lock()
            .split_region(&self.region, splits)?
            .map(|region| Self {
                buffers: self.buffers.clone(),
                region,
                region_offset: 0,
            })
            .collect())
    }

    async fn next_segment_id_once_active(&self) -> VortexResult<SegmentId> {
        WaitRegionFuture {
            buffers: self.buffers.clone(),
            region: self.region,
        }
        .await
    }

    pub async fn flush<W: VortexWrite>(
        &mut self,
        writer: &mut futures::io::Cursor<W>,
        segment_specs: &mut Vec<SegmentSpec>,
    ) -> VortexResult<()> {
        let completed = {
            let mut guard = self.buffers.lock();
            guard.completed_buffers()?
        };
        for buffers in completed.into_values() {
            // The API requires us to write these buffers contiguously. Therefore, we can only
            // respect the alignment of the first one.
            // Don't worry, in most cases the caller knows what they're doing and will align the
            // buffers themselves, inserting padding buffers where necessary.
            let alignment = buffers
                .first()
                .map(|buffer| buffer.alignment())
                .unwrap_or_else(Alignment::none);

            // Add any padding required to align the segment.
            let offset = writer.position();
            let padding = offset.next_multiple_of(*alignment as u64) - offset;
            if padding > 0 {
                writer
                    .write_all(ByteBuffer::zeroed(padding as usize))
                    .await?;
            }
            let offset = writer.position();

            for buffer in buffers {
                writer.write_all(buffer).await?;
            }

            segment_specs.push(SegmentSpec {
                offset,
                length: u32::try_from(writer.position() - offset)
                    .map_err(|_| vortex_err!("segment length exceeds maximum u32"))?,
                alignment,
            });
        }
        Ok(())
    }
}

impl Drop for InOrderSegmentWriter {
    fn drop(&mut self) {
        self.buffers.lock().finish_region(&self.region);
    }
}

struct WaitRegionFuture {
    buffers: Arc<Mutex<OrderedBuffers>>,
    region: Region,
}

impl Future for WaitRegionFuture {
    type Output = VortexResult<SegmentId>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut guard = self.buffers.lock();
        let current_first = match guard.first_region() {
            Ok(first) => first,
            Err(e) => return Poll::Ready(Err(e)),
        };
        if self.region == current_first {
            return Poll::Ready(Ok(guard.next_segment_id()));
        }
        guard.register_waker(self.region, cx.waker().clone());
        Poll::Pending
    }
}
