// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::channel::oneshot;
use parking_lot::Mutex;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::segments::EncryptionKeyName;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSink;
use vortex_layout::segments::SegmentWriteOptions;
use vortex_layout::sequence::SequenceId;
use vortex_utils::aliases::hash_map::HashMap;

use crate::encryption::SegmentCipher;
use crate::encryption::SegmentRole;
use crate::footer::SegmentSpec;

/// A buffer sent from a [`BufferedSegmentSink`] to the file, in file order.
pub(crate) enum PendingBuffer {
    Ready(ByteBuffer),
    /// A segment that is still being encrypted. It is sent in order before the encryption
    /// starts, so that segments can be encrypted in parallel and still be written in order.
    Encrypting(oneshot::Receiver<VortexResult<ByteBuffer>>),
}

impl PendingBuffer {
    /// Wait for the buffer.
    pub(crate) async fn resolve(self) -> VortexResult<ByteBuffer> {
        match self {
            PendingBuffer::Ready(buffer) => Ok(buffer),
            PendingBuffer::Encrypting(receiver) => receiver
                .await
                .map_err(|_| vortex_err!("Segment encryption was cancelled"))?,
        }
    }
}

pub struct BufferedSegmentSink {
    buffers: kanal::AsyncSender<PendingBuffer>,
    byte_offset: AtomicU64,
    segment_specs: Mutex<Vec<SegmentSpec>>,
    /// The cipher for segments that require no particular key, and the index of its spec in the
    /// footer.
    default_cipher: Option<(SegmentCipher, u16)>,
    /// The ciphers for segments that require a named key, and the indices of their specs.
    named_ciphers: HashMap<EncryptionKeyName, (SegmentCipher, u16)>,
}

impl BufferedSegmentSink {
    pub(crate) fn new(send: kanal::AsyncSender<PendingBuffer>, byte_offset: u64) -> Self {
        Self {
            buffers: send,
            byte_offset: AtomicU64::new(byte_offset),
            segment_specs: Default::default(),
            default_cipher: None,
            named_ciphers: HashMap::default(),
        }
    }

    /// Encrypt segments that require no particular key with `cipher`, whose spec is at
    /// `encryption` in the footer.
    pub(crate) fn with_default_cipher(mut self, cipher: SegmentCipher, encryption: u16) -> Self {
        self.default_cipher = Some((cipher, encryption));
        self
    }

    /// Encrypt segments that require the key `name` with `cipher`, whose spec is at `encryption`
    /// in the footer.
    pub(crate) fn with_named_cipher(
        mut self,
        name: EncryptionKeyName,
        cipher: SegmentCipher,
        encryption: u16,
    ) -> Self {
        self.named_ciphers.insert(name, (cipher, encryption));
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
        sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId> {
        self.write_with_options(sequence_id, buffers, SegmentWriteOptions::default())
            .await
    }

    async fn write_with_options(
        &self,
        mut sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
        options: SegmentWriteOptions,
    ) -> VortexResult<SegmentId> {
        let cipher = match &options.encryption {
            None => self.default_cipher.as_ref(),
            Some(name) => Some(self.named_ciphers.get(name).ok_or_else(|| {
                vortex_err!("No encryption key named {name} was configured for the file")
            })?),
        };

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

        // The sequence id still orders this write after all earlier ones, so sending to the
        // stream keeps the file order.
        if let Some(padding) = padding_buffer {
            let _ = self.buffers.send(PendingBuffer::Ready(padding)).await;
        }
        let Some((cipher, _)) = cipher else {
            for buffer in buffers {
                let _ = self.buffers.send(PendingBuffer::Ready(buffer)).await;
            }
            return Ok(segment_id);
        };

        // Reserve the segment's place in the stream, then release the ordering before
        // encrypting, so that later segments, of this column or others, are not held up by it.
        let (encrypted_tx, encrypted_rx) = oneshot::channel();
        let _ = self
            .buffers
            .send(PendingBuffer::Encrypting(encrypted_rx))
            .await;
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
