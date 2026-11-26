// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_channel::Sender;
use async_trait::async_trait;
use futures::channel::oneshot;
use parking_lot::Mutex;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::BufferedBytesReservation;
use vortex_layout::BufferedBytesTracker;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSink;
use vortex_layout::sequence::SequenceId;

use crate::encryption::SegmentCipher;
use crate::encryption::SegmentRole;
use crate::footer::SegmentSpec;

/// A buffer of a segment on its way to the file, in file order.
///
/// The last buffer of each segment carries the reservation that counts the segment as buffered.
/// The writer drops it after it writes the buffer, so the bytes of a segment are always in the
/// buffered bytes or in the bytes written.
pub(crate) struct SegmentBuffer {
    buffer: PendingBuffer,
    reservation: Option<BufferedBytesReservation>,
}

/// A buffer sent from a [`BufferedSegmentSink`] to the file, in file order.
enum PendingBuffer {
    Ready(ByteBuffer),
    /// A segment that is still being encrypted. It is sent in order before the encryption
    /// starts, so that segments can be encrypted in parallel and still be written in order.
    Encrypting(oneshot::Receiver<VortexResult<ByteBuffer>>),
}

impl SegmentBuffer {
    fn ready(buffer: ByteBuffer, reservation: Option<BufferedBytesReservation>) -> Self {
        Self {
            buffer: PendingBuffer::Ready(buffer),
            reservation,
        }
    }

    fn encrypting(
        receiver: oneshot::Receiver<VortexResult<ByteBuffer>>,
        reservation: Option<BufferedBytesReservation>,
    ) -> Self {
        Self {
            buffer: PendingBuffer::Encrypting(receiver),
            reservation,
        }
    }

    /// Wait for the buffer and return it with the reservation to drop after the buffer is written.
    pub(crate) async fn resolve(
        self,
    ) -> VortexResult<(ByteBuffer, Option<BufferedBytesReservation>)> {
        let buffer = match self.buffer {
            PendingBuffer::Ready(buffer) => buffer,
            PendingBuffer::Encrypting(receiver) => receiver
                .await
                .map_err(|_| vortex_err!("Segment encryption was cancelled"))??,
        };
        Ok((buffer, self.reservation))
    }
}

pub struct BufferedSegmentSink {
    buffers: Sender<SegmentBuffer>,
    byte_offset: AtomicU64,
    segment_specs: Mutex<Vec<SegmentSpec>>,
    /// Counts the segments that the sink has not written yet, so that the writer's buffered bytes
    /// include them.
    buffered_bytes: BufferedBytesTracker,
    /// The cipher that encrypts every segment, and the index of its spec in the footer.
    cipher: Option<(SegmentCipher, u16)>,
}

impl BufferedSegmentSink {
    pub(crate) fn new(
        send: Sender<SegmentBuffer>,
        byte_offset: u64,
        buffered_bytes: BufferedBytesTracker,
    ) -> Self {
        Self {
            buffers: send,
            byte_offset: AtomicU64::new(byte_offset),
            segment_specs: Default::default(),
            buffered_bytes,
            cipher: None,
        }
    }

    /// Encrypt every segment with `cipher`, whose spec is at `encryption` in the footer.
    pub(crate) fn with_cipher(mut self, cipher: SegmentCipher, encryption: u16) -> Self {
        self.cipher = Some((cipher, encryption));
        self
    }

    /// Close the sink, returning the segment specs and the final byte offset.
    pub fn segment_specs(&self) -> Arc<[SegmentSpec]> {
        let specs = self.segment_specs.lock();
        specs.clone().into()
    }
}

#[async_trait]
impl SegmentSink for BufferedSegmentSink {
    async fn write(
        &self,
        mut sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId> {
        // Segments reach the file in sequence order. A layout that holds an early sequence ID,
        // such as a dictionary that writes its values last, can hold back every later segment
        // until the end of the file. Count the segment as buffered until the writer writes it.
        let segment_bytes: usize = buffers.iter().map(|buffer| buffer.len()).sum();
        let reservation = self.buffered_bytes.reserve(segment_bytes as u64);

        let cipher = self.cipher.as_ref();
        // We wait for all segment IDs before this one to be dropped. Then while we hold a strong
        // reference to this one, we essentially have an exclusive lock on the segment writer.
        sequence_id.collapse().await;

        let (segment_id, padding_buffer) = {
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
            let plaintext_length = buffers.iter().map(|buffer| buffer.len()).sum::<usize>();
            let (length, encryption) = match cipher {
                None => (plaintext_length, 0),
                Some((cipher, encryption)) => (plaintext_length + cipher.overhead(), *encryption),
            };
            let length = u32::try_from(length)
                .map_err(|_| vortex_err!("segment buffer length exceeds maximum u32"))?;

            // Add any padding required to align the segment.
            let byte_offset = self.byte_offset.load(Ordering::Relaxed);
            let padding = byte_offset.next_multiple_of(alignment.as_usize() as u64) - byte_offset;
            let offset = byte_offset + padding;
            specs.push(SegmentSpec {
                offset,
                length,
                alignment,
                encryption,
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

        let Some((cipher, _)) = cipher else {
            let buffers: Vec<ByteBuffer> = padding_buffer.into_iter().chain(buffers).collect();
            let last = buffers.len().saturating_sub(1);
            let mut reservation = Some(reservation);

            // A failed send drops the buffer with its reservation.
            for (index, buffer) in buffers.into_iter().enumerate() {
                let reservation = if index == last {
                    reservation.take()
                } else {
                    None
                };
                drop(
                    self.buffers
                        .send(SegmentBuffer::ready(buffer, reservation))
                        .await,
                );
            }
            return Ok(segment_id);
        };

        // The sequence id still orders this write after all earlier ones, so sending to the
        // stream keeps the file order.
        if let Some(padding) = padding_buffer {
            drop(self.buffers.send(SegmentBuffer::ready(padding, None)).await);
        }

        // Reserve the segment's place in the stream, then release the ordering before
        // encrypting, so that later segments, of this column or others, are not held up by it.
        let (encrypted_tx, encrypted_rx) = oneshot::channel();
        drop(
            self.buffers
                .send(SegmentBuffer::encrypting(encrypted_rx, Some(reservation)))
                .await,
        );
        drop(sequence_id);

        let alignment = buffers
            .first()
            .map(|buffer| buffer.alignment())
            .unwrap_or_else(Alignment::none);
        let encrypted = cipher.encrypt(SegmentRole::Data(*segment_id), &buffers, alignment);
        let failed = encrypted.is_err();
        // The file writer reports the error it receives, before it awaits the layout. If it
        // dropped the receiver, it failed already, and reports why itself.
        drop(encrypted_tx.send(encrypted));
        if failed {
            vortex_bail!("Failed to encrypt segment {}", *segment_id);
        }
        Ok(segment_id)
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;

    use async_channel::bounded;
    use futures::poll;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_layout::BufferedBytesTracker;
    use vortex_layout::segments::SegmentSink;
    use vortex_layout::sequence::SequenceId;

    use super::BufferedSegmentSink;

    #[tokio::test]
    async fn segments_count_as_buffered_until_written() -> VortexResult<()> {
        let tracker = BufferedBytesTracker::new();
        let (send, recv) = bounded(4);
        let sink = BufferedSegmentSink::new(send, 0, tracker.clone());

        let mut pointer = SequenceId::root();
        let first = pointer.advance();
        let second = pointer.advance();

        // The second segment waits until the first sequence ID is dropped.
        let mut write = pin!(sink.write(second, vec![ByteBuffer::zeroed(24)]));
        assert!(poll!(write.as_mut()).is_pending());
        assert_eq!(tracker.buffered_bytes(), 24);

        // A sent segment counts until the writer drops its reservation.
        drop(first);
        write.await?;
        assert_eq!(tracker.buffered_bytes(), 24);

        let (buffer, reservation) = recv
            .recv()
            .await
            .map_err(|e| vortex_err!("The sink sent no segment: {e}"))?
            .resolve()
            .await?;
        assert_eq!(buffer.len(), 24);
        assert_eq!(tracker.buffered_bytes(), 24);

        drop(reservation);
        assert_eq!(tracker.buffered_bytes(), 0);

        Ok(())
    }
}
