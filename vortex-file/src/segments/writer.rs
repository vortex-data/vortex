use futures::StreamExt;
use futures::channel::mpsc;
use futures::io::Cursor;
use parking_lot::Mutex;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexResult, VortexUnwrap, vortex_bail, vortex_err};
use vortex_io::VortexWrite;
use vortex_layout::segments::{SegmentId, SegmentWriter};

use crate::footer::SegmentSpec;

pub struct SerialSegmentWriter {
    state: Mutex<State>,
}

struct State {
    flush_tx: mpsc::UnboundedSender<Vec<ByteBuffer>>,
    next_expected: SegmentId,
}

impl SegmentWriter for SerialSegmentWriter {
    fn put(&self, segment_id: SegmentId, buffer: Vec<ByteBuffer>) -> VortexResult<()> {
        let mut guard = self.state.lock();
        if segment_id != guard.next_expected {
            vortex_bail!(
                "out of order segment id, expected {:?}, got {:?}",
                guard.next_expected,
                segment_id
            );
        }
        guard.next_expected = SegmentId::from(*segment_id + 1);
        guard
            .flush_tx
            .unbounded_send(buffer)
            .map_err(|_| vortex_err!("out of memory"))
            .vortex_unwrap();
        Ok(())
    }
}

impl SerialSegmentWriter {
    pub fn create() -> (Self, SegmentFlusher) {
        // TODO(os): make this bounded, slow I/O means we will buffer
        // in memory unbounded. Currently tx is used in an impl Drop so
        // we can't do a bounded async send.
        let (flush_tx, rx) = mpsc::unbounded();
        (
            SerialSegmentWriter {
                state: Mutex::new(State {
                    flush_tx,
                    next_expected: SegmentId::from(0),
                }),
            },
            SegmentFlusher {
                rx,
                segment_specs: Vec::new(),
            },
        )
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
