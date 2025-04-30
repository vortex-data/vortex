use futures::StreamExt;
use futures::channel::mpsc;
use futures::io::Cursor;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexResult, vortex_err};
use vortex_io::VortexWrite;
use vortex_layout::segments::{SegmentId, SegmentWriter};

use crate::footer::SegmentSpec;

type Segments = Vec<Vec<ByteBuffer>>;

/// A segment writer that holds buffers in memory until they are flushed by a writer.
pub(crate) struct SegmentsBuffer {
    next_id: SegmentId,
    current: Segments,
    to_flush: mpsc::UnboundedSender<Segments>,
}

/// Segment Writer that flushes all segment buffers it receives until the sender [SegmentsBuffer]
/// is dropped.
pub(crate) struct BufferedSegmentWriter {
    rx: mpsc::UnboundedReceiver<Segments>,
}

impl BufferedSegmentWriter {
    pub fn create() -> (SegmentsBuffer, Self) {
        let (tx, rx) = mpsc::unbounded();
        (
            SegmentsBuffer {
                next_id: SegmentId::default(),
                current: Vec::new(),
                to_flush: tx,
            },
            Self { rx },
        )
    }
}

impl SegmentWriter for SegmentsBuffer {
    fn put(&mut self, data: &[ByteBuffer]) -> SegmentId {
        self.current.push(data.to_vec());
        let id = self.next_id;
        self.next_id = SegmentId::from(*self.next_id + 1);
        id
    }
}

impl SegmentsBuffer {
    pub fn flush(&mut self) -> VortexResult<()> {
        self.to_flush
            .unbounded_send(std::mem::take(&mut self.current))
            .map_err(|_| vortex_err!("buffered segment writer dropped"))
    }
}

impl BufferedSegmentWriter {
    /// Flush the segments to the provided async writer.
    pub async fn write<W: VortexWrite>(
        mut self,
        mut writer: Cursor<W>,
    ) -> VortexResult<(Cursor<W>, Vec<SegmentSpec>)> {
        let mut segment_specs = Vec::new();
        while let Some(mut segments) = self.rx.next().await {
            for buffers in segments.drain(..) {
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
        }
        Ok((writer, segment_specs))
    }
}
