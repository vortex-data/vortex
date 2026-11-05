// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::segments::SegmentId;
use crate::sequence::SequenceId;

/// Shared writer-side segment sink.
pub type SegmentSinkRef = Arc<dyn SegmentSink>;

/// Names the key a segment must be encrypted with.
///
/// The name is opaque to layouts. The segment sink resolves it to an actual key, e.g. one
/// registered with the file writer.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EncryptionKeyName(Arc<str>);

impl EncryptionKeyName {
    /// Create a key name.
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self(name.into())
    }

    /// The name as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<S: Into<Arc<str>>> From<S> for EncryptionKeyName {
    fn from(name: S) -> Self {
        Self::new(name)
    }
}

impl Display for EncryptionKeyName {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Requirements on how a segment is stored, passed to [`SegmentSink::write_with_options`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SegmentWriteOptions {
    /// The key the segment must be encrypted with, or `None` for the sink's default.
    pub encryption: Option<EncryptionKeyName>,
}

impl SegmentWriteOptions {
    /// Require the segment to be encrypted with the named key.
    pub fn with_encryption(mut self, key: EncryptionKeyName) -> Self {
        self.encryption = Some(key);
        self
    }
}

#[async_trait]
/// Assigns segment ids and writes segment buffers during layout writing.
///
/// Segment sinks are responsible for preserving any ordering guarantees required by the storage
/// backend. The [`SequenceId`] argument lets sinks serialize id assignment while still allowing
/// upstream layout strategies to do work concurrently.
pub trait SegmentSink: Send + Sync {
    /// Write the given data into a segment, ordered based on the provided sequence identifier.
    ///
    /// Implementations of this trait should call [`SequenceId::collapse`] on the provided
    /// `sequence_id` if they need to ensure that the segment IDs are monotonically increasing.
    /// While they hold onto the returned `SequenceId`, they can be sure that no other subsequent
    /// calls to [`SequenceId::collapse`] will complete.
    ///
    /// If they do not require ordered segment IDs, for example if segments are stored in
    /// random-access key/values storage, then the sequence ID can be dropped and the segment
    /// written immediately.
    async fn write(
        &self,
        sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId>;

    /// Write a segment like [`write`](Self::write), honoring `options`.
    ///
    /// Sinks that support encryption must override this. The default implementation fails for a
    /// segment that requires encryption, so that it is never silently written in plaintext.
    async fn write_with_options(
        &self,
        sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
        options: SegmentWriteOptions,
    ) -> VortexResult<SegmentId> {
        if let Some(key) = options.encryption {
            vortex_bail!(
                "Segment sink does not support encryption, but a segment requires key {key}"
            );
        }
        self.write(sequence_id, buffers).await
    }
}

/// A [`SegmentSink`] that requires every segment written through it to be encrypted with a key.
///
/// Segments that already require a key, from a sink nested beneath this one, keep it.
pub struct EncryptingSegmentSink {
    inner: SegmentSinkRef,
    key: EncryptionKeyName,
}

impl EncryptingSegmentSink {
    /// Wrap `inner` to encrypt segments with `key`.
    pub fn new(inner: SegmentSinkRef, key: EncryptionKeyName) -> Self {
        Self { inner, key }
    }
}

#[async_trait]
impl SegmentSink for EncryptingSegmentSink {
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
        sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
        mut options: SegmentWriteOptions,
    ) -> VortexResult<SegmentId> {
        if options.encryption.is_none() {
            options.encryption = Some(self.key.clone());
        }
        self.inner
            .write_with_options(sequence_id, buffers, options)
            .await
    }
}
