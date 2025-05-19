#![allow(dead_code)]
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::io::Cursor;
use parking_lot::Mutex;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexResult, vortex_err};
use vortex_io::VortexWrite;
use vortex_layout::segments::{ConcurrentSegmentWriter, SegmentId, SegmentWriter};

use super::ordered::{OrderedBuffers, Section};
use crate::footer::SegmentSpec;

/// A segment writer that holds buffers in memory until they are flushed by a writer.
pub struct InOrderSegmentWriter {
    buffers: Arc<Mutex<OrderedBuffers>>,
    section: Section,
    subsection_idx: usize,
    buffers_tx: mpsc::UnboundedSender<Vec<ByteBuffer>>,
}

#[async_trait]
impl SegmentWriter for InOrderSegmentWriter {
    async fn put(&mut self, data: Vec<ByteBuffer>) -> VortexResult<SegmentId> {
        self.buffers
            .lock()
            .insert_buffer(self.section.subsection(self.subsection_idx), data);
        self.subsection_idx += 1;
        self.next_segment_id_once_active().await
    }
}

impl ConcurrentSegmentWriter for InOrderSegmentWriter {
    fn split_off(&mut self, splits: usize) -> VortexResult<Vec<Box<dyn ConcurrentSegmentWriter>>> {
        let mut guard = self.buffers.lock();
        let splits = guard
            .split_section(&self.section, splits, self.subsection_idx)?
            .map(|section| {
                Box::new(Self {
                    buffers: self.buffers.clone(),
                    buffers_tx: self.buffers_tx.clone(),
                    section,
                    subsection_idx: 0,
                }) as Box<dyn ConcurrentSegmentWriter>
            })
            .collect();
        self.section.increment();
        guard.add_section(&self.section);
        Ok(splits)
    }
}

impl InOrderSegmentWriter {
    pub fn create() -> (Self, SegmentFlusher) {
        // TODO(os): make this bounded, slow I/O means we will buffer
        // in memory unbounded. Currently tx is used in an impl Drop so
        // we can't do a bounded async send.
        let (buffers_tx, rx) = mpsc::unbounded();
        (
            InOrderSegmentWriter {
                buffers: Default::default(),
                section: Section::default(),
                subsection_idx: 0,
                buffers_tx,
            },
            SegmentFlusher {
                rx,
                segment_specs: Vec::new(),
            },
        )
    }

    async fn next_segment_id_once_active(&self) -> VortexResult<SegmentId> {
        WaitRegionFuture {
            buffers: self.buffers.clone(),
            section: self.section.clone(),
        }
        .await
    }
}

impl Drop for InOrderSegmentWriter {
    fn drop(&mut self) {
        let Some(completed) = self.buffers.lock().finish_section(&self.section) else {
            return;
        };
        for buffer in completed.into_values() {
            self.buffers_tx
                .unbounded_send(buffer)
                .expect("out of memory");
        }
    }
}

struct WaitRegionFuture {
    buffers: Arc<Mutex<OrderedBuffers>>,
    section: Section,
}

impl Future for WaitRegionFuture {
    type Output = VortexResult<SegmentId>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut guard = self.buffers.lock();
        let current_first = match guard.first_section() {
            Ok(first) => first,
            Err(e) => return Poll::Ready(Err(e)),
        };
        if self.section == current_first {
            return Poll::Ready(Ok(guard.next_segment_id()));
        }
        guard.register_waker(self.section.clone(), cx.waker().clone());
        Poll::Pending
    }
}

pub struct SegmentFlusher {
    rx: mpsc::UnboundedReceiver<Vec<ByteBuffer>>,
    segment_specs: Vec<SegmentSpec>,
}

impl SegmentFlusher {
    pub async fn flush<W: VortexWrite>(
        mut self,
        mut writer: Cursor<W>,
    ) -> VortexResult<(Cursor<W>, Vec<SegmentSpec>)> {
        while let Some(buffers) = self.rx.next().await {
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

            self.segment_specs.push(SegmentSpec {
                offset,
                length: u32::try_from(writer.position() - offset)
                    .map_err(|_| vortex_err!("segment length exceeds maximum u32"))?,
                alignment,
            });
        }
        Ok((writer, self.segment_specs))
    }
}
