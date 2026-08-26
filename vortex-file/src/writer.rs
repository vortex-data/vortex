// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::Fuse;
use futures::future::LocalBoxFuture;
use futures::future::ready;
use futures::pin_mut;
use futures::select;
use itertools::Itertools;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::stats::Stat;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorExt;
use vortex_array::stats::PRUNING_STATS;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::ByteBuffer;
use vortex_edition::ComponentKind;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::IoBuf;
use vortex_io::VortexWrite;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::BufferedBytesTracker;
use vortex_layout::LayoutContext;
use vortex_layout::LayoutStrategy;
use vortex_layout::LayoutWriterContext;
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialStreamAdapter;
use vortex_layout::sequence::SequentialStreamExt;
use vortex_session::SessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::Id;
use vortex_session::registry::ReadContext;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::Footer;
use crate::MAGIC_BYTES;
use crate::WriteStrategyBuilder;
use crate::counting::CountingVortexWrite;
use crate::footer::FileStatistics;
use crate::footer::MAX_METADATA_KEY_BYTES;
use crate::footer::MAX_METADATA_SEGMENTS;
use crate::segments::writer::BufferedSegmentSink;

/// Configure a new writer, which can eventually be used to write an [`ArrayStream`] into a sink
/// that implements [`VortexWrite`].
///
/// All write strategies are restricted to the components in the session's enabled editions: an
/// array, layout, extension dtype, or zone-map aggregate outside them fails the write. An empty
/// component set therefore forbids writing any component of that kind.
///
/// Construct with [`WriteOptionsSessionExt::write_options`] for normal use so the writer inherits
/// the session's runtime, array registry, and memory configuration.
pub struct VortexWriteOptions {
    session: VortexSession,
    strategy: Arc<dyn LayoutStrategy>,
    buffered_bytes: BufferedBytesTracker,
    exclude_dtype: bool,
    max_variable_length_statistics_size: usize,
    file_statistics: Vec<Stat>,
    metadata: HashMap<String, ByteBuffer>,
}

/// Extension trait for constructing [`VortexWriteOptions`] from a session.
pub trait WriteOptionsSessionExt: SessionExt {
    /// Create [`VortexWriteOptions`] for writing to a Vortex file.
    fn write_options(&self) -> VortexWriteOptions {
        VortexWriteOptions::new(self.session())
    }
}
impl<S: SessionExt> WriteOptionsSessionExt for S {}

impl VortexWriteOptions {
    /// Create a new [`VortexWriteOptions`] with the given session.
    pub fn new(session: VortexSession) -> Self {
        let strategy = WriteStrategyBuilder::default()
            .with_allow_encodings(
                session
                    .enabled_component_ids(ComponentKind::Array)
                    .into_iter()
                    .collect(),
            )
            .build();
        VortexWriteOptions {
            strategy,
            buffered_bytes: BufferedBytesTracker::new(),
            session,
            exclude_dtype: false,
            file_statistics: PRUNING_STATS.to_vec(),
            max_variable_length_statistics_size: 64,
            metadata: HashMap::default(),
        }
    }

    /// Replace the default layout strategy with the provided one.
    ///
    /// The strategy controls repartitioning, statistics layout, compression, and leaf segment
    /// emission. Use [`WriteStrategyBuilder`] when only a small part of the default strategy needs
    /// customization. Replacing the strategy does not change the enabled-edition encoding policy.
    pub fn with_strategy(mut self, strategy: Arc<dyn LayoutStrategy>) -> Self {
        self.strategy = strategy;
        self
    }

    /// Returns the tracker accounting for bytes that layout strategies are holding but have not
    /// yet emitted.
    ///
    /// The tracker is shared with the write these options start, so it can be captured before
    /// calling [`Self::write`] and polled while the write runs. [`Writer::buffered_bytes`] exposes
    /// the same counter for the push-based API.
    pub fn buffered_bytes_tracker(&self) -> BufferedBytesTracker {
        self.buffered_bytes.clone()
    }

    /// Exclude the DType from the Vortex file. You must provide the DType to the reader.
    // TODO(ngates): Should we store some sort of DType checksum to make sure the one passed at
    //  read-time is sane? I guess most layouts will have some reasonable validation.
    pub fn exclude_dtype(mut self) -> Self {
        self.exclude_dtype = true;
        self
    }

    /// Configure which statistics to compute at the file level.
    ///
    /// Pass an empty vector to omit file-level statistics.
    pub fn with_file_statistics(mut self, file_statistics: Vec<Stat>) -> Self {
        self.file_statistics = file_statistics;
        self
    }

    /// Add a user-defined metadata segment (keyed, opaque bytes); a repeated key replaces the
    /// previous value. Keys and the segment count are validated against `MAX_METADATA_*`.
    pub fn with_metadata_segment(
        mut self,
        key: impl Into<String>,
        metadata: impl Into<ByteBuffer>,
    ) -> Self {
        let key = key.into();
        let metadata = metadata.into();
        self.metadata.insert(key, metadata);
        self
    }

    /// Add user-defined metadata segments to the file.
    ///
    /// If a key already exists, the previous segment for that key is replaced.
    pub fn with_metadata_segments<I, K, B>(mut self, metadata: I) -> Self
    where
        I: IntoIterator<Item = (K, B)>,
        K: Into<String>,
        B: Into<ByteBuffer>,
    {
        for (key, metadata) in metadata {
            self = self.with_metadata_segment(key, metadata);
        }
        self
    }

    /// Check the configured metadata segments against the `MAX_METADATA_*` limits.
    ///
    /// [`Self::write`] performs the same check, but only once the sink is already being written
    /// to. Callers that accept metadata from elsewhere (FFI bindings, for example) can use this to
    /// reject an invalid set before any bytes are produced.
    pub fn validate_metadata(&self) -> VortexResult<()> {
        validate_metadata_segments(&self.metadata)
    }
}

impl VortexWriteOptions {
    /// Drop into the blocking writer API using the given runtime.
    ///
    /// The returned adapter drives async writer internals on `runtime` while accepting ordinary
    /// [`std::io::Write`] sinks and [`ArrayIterator`] inputs.
    pub fn blocking<B: BlockingRuntime>(self, runtime: &B) -> BlockingWrite<'_, B> {
        BlockingWrite {
            options: self,
            runtime,
        }
    }

    /// Write an [`ArrayStream`] as a Vortex file.
    ///
    /// Note that buffers are flushed as soon as they are available with no buffering, the caller
    /// is responsible for deciding how to configure buffering on the underlying `Write` sink.
    ///
    /// The set of encodings permitted in the file is snapshotted from the session's enabled
    /// editions here, so editions enabled after this call do not affect the write.
    pub async fn write<W: VortexWrite + Unpin, S: ArrayStream + Send + 'static>(
        self,
        write: W,
        stream: S,
    ) -> VortexResult<WriteSummary> {
        self.write_internal(write, ArrayStreamExt::boxed(stream))
            .await
    }

    async fn write_internal<W: VortexWrite + Unpin>(
        self,
        mut write: W,
        stream: SendableArrayStream,
    ) -> VortexResult<WriteSummary> {
        validate_metadata_segments(&self.metadata)?;

        // The array context is built here, rather than when the options were constructed, so that
        // encodings registered on the session in between are still eligible for the file.
        let ctx = LayoutWriterContext::new(new_array_context(&self.session))
            .with_buffered_bytes_tracker(self.buffered_bytes.clone())
            .with_allowed_aggregates(edition_filter(&self.session, ComponentKind::Aggregate));
        let dtype = stream.dtype().clone();
        validate_dtype_editions(&self.session, &dtype)?;

        let (mut ptr, eof) = SequenceId::root().split();

        let stream = SequentialStreamAdapter::new(
            dtype.clone(),
            stream
                .try_filter(|chunk| ready(!chunk.is_empty()))
                .map(move |result| result.map(|chunk| (ptr.advance(), chunk))),
        )
        .sendable();
        let (file_stats, stream) = accumulate_stats(
            stream,
            self.file_statistics.clone().into(),
            self.max_variable_length_statistics_size,
            &self.session,
        );

        // First, write the magic bytes.
        write.write_all(ByteBuffer::copy_from(MAGIC_BYTES)).await?;
        let mut position = MAGIC_BYTES.len() as u64;

        // Create a channel to send buffers from the segment sink to the output stream.
        let (send, recv) = kanal::bounded_async(1);

        let segments = Arc::new(BufferedSegmentSink::new(send, position));

        // We spawn the layout future so it is driven in the background while we write the
        // buffer stream, so we don't need to poll it until all buffers have been drained.
        let ctx2 = ctx.clone();
        let session = self.session.clone();
        let layout_fut = self.session.handle().spawn_nested(move |_| async move {
            let layout = self
                .strategy
                .write_stream(
                    ctx2,
                    Arc::<BufferedSegmentSink>::clone(&segments),
                    stream,
                    eof,
                    &session,
                )
                .await?;
            Ok::<_, VortexError>((layout, segments.segment_specs()))
        });

        // Flush buffers as they arrive
        let recv_stream = recv.into_stream();
        pin_mut!(recv_stream);
        while let Some(buffer) = recv_stream.next().await {
            if buffer.is_empty() {
                continue;
            }
            position += buffer.len() as u64;
            write.write_all(buffer).await?;
        }

        let (layout, segment_specs) = layout_fut.await?;

        // Assemble the Footer object now that we have all the segments.
        let statistics = if self.file_statistics.is_empty() {
            None
        } else {
            Some(FileStatistics::new_with_dtype(
                file_stats.stats_sets().into(),
                &dtype,
            ))
        };
        let mut footer = Footer::new(
            Arc::clone(&layout),
            segment_specs,
            statistics,
            ReadContext::new(ctx.array_ctx().to_ids()),
        );

        // Emit the footer buffers and EOF.
        let (footer_buffers, metadata, approx_byte_size) = footer
            .clone()
            .into_serializer()
            .with_layout_context(new_layout_context(&self.session))
            .with_metadata_segments(self.metadata)
            .with_offset(position)
            .with_exclude_dtype(self.exclude_dtype)
            .serialize_with_metadata()?;
        footer = footer
            .with_metadata_segments(metadata)
            .with_approx_byte_size(approx_byte_size);

        for buffer in footer_buffers {
            position += buffer.len() as u64;
            write.write_all(buffer).await?;
        }

        write.flush().await?;

        Ok(WriteSummary {
            footer,
            size: position,
        })
    }

    /// Create a push-based [`Writer`] that can be used to incrementally write arrays to the file.
    ///
    /// Each pushed chunk must have dtype `dtype`. Call [`Writer::finish`] to close the input stream,
    /// flush remaining buffers, and receive the [`WriteSummary`].
    pub fn writer<'w, W: VortexWrite + Unpin + 'w>(self, write: W, dtype: DType) -> Writer<'w> {
        // Create a channel for sending arrays to the layout task.
        let (arrays_send, arrays_recv) = kanal::bounded_async(1);

        let arrays =
            ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays_recv.into_stream()));

        let write = CountingVortexWrite::new(write);
        let bytes_written = write.counter();
        let buffered_bytes = self.buffered_bytes.clone();
        let future = self.write(write, arrays).boxed_local().fuse();

        Writer {
            arrays: Some(arrays_send),
            future,
            bytes_written,
            buffered_bytes,
        }
    }
}

fn new_array_context(session: &VortexSession) -> ArrayContext {
    // NOTE(os): Set up an array context with all enabled encodings pre-populated.
    // This is preferred for now over having an empty context here, because only the
    // serialised array order is deterministic. The serialisation of arrays are done
    // parallel and with an empty context they can register their encodings to the context
    // in different order, changing the written bytes from run to run.
    let enabled_encoding_ids = session.enabled_component_ids(ComponentKind::Array);
    ArrayContext::new(enabled_encoding_ids.iter().cloned().sorted().collect())
        // Only permit encodings in the enabled editions.
        .with_allowed_ids(enabled_encoding_ids.into_iter().collect())
}

/// The ids of `kind` the enabled editions permit.
fn edition_filter(session: &VortexSession, kind: ComponentKind) -> HashSet<Id> {
    session.enabled_component_ids(kind).into_iter().collect()
}

/// Validate every extension dtype nested in the file schema against the enabled editions.
fn validate_dtype_editions(session: &VortexSession, dtype: &DType) -> VortexResult<()> {
    let allowed = edition_filter(session, ComponentKind::DType);

    fn validate(dtype: &DType, allowed: &HashSet<Id>) -> VortexResult<()> {
        match dtype {
            DType::List(element, _) | DType::FixedSizeList(element, ..) => {
                validate(element, allowed)
            }
            DType::Map(map, _) => {
                validate(&map.key_dtype(), allowed)?;
                validate(&map.value_dtype(), allowed)
            }
            DType::Struct(fields, _) => {
                for field in fields.fields() {
                    validate(&field, allowed)?;
                }
                Ok(())
            }
            DType::Union(variants, _) => {
                for variant in variants.variants() {
                    validate(&variant, allowed)?;
                }
                Ok(())
            }
            DType::Extension(extension) => {
                if !allowed.contains(&extension.id()) {
                    vortex_bail!(
                        "Extension DType {} not permitted by enabled editions",
                        extension.id()
                    );
                }
                validate(extension.storage_dtype(), allowed)
            }
            DType::Null
            | DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(_)
            | DType::Binary(_)
            | DType::Variant(_) => Ok(()),
        }
    }

    validate(dtype, &allowed)
}

/// The context every layout in the file is interned through, restricted to the layouts the
/// enabled editions permit.
fn new_layout_context(session: &VortexSession) -> LayoutContext {
    LayoutContext::default().with_allowed_ids(edition_filter(session, ComponentKind::Layout))
}

fn validate_metadata_segments(metadata: &HashMap<String, ByteBuffer>) -> VortexResult<()> {
    if metadata.len() > MAX_METADATA_SEGMENTS {
        vortex_bail!(
            "Vortex files may contain at most {} metadata segments; got {} metadata segments. Metadata keys must be non-empty and at most {} bytes",
            MAX_METADATA_SEGMENTS,
            metadata.len(),
            MAX_METADATA_KEY_BYTES
        );
    }

    for key in metadata.keys() {
        if key.is_empty() {
            vortex_bail!(
                "Vortex metadata keys must be non-empty and at most {} bytes; files may contain at most {} metadata segments",
                MAX_METADATA_KEY_BYTES,
                MAX_METADATA_SEGMENTS
            );
        }

        let key_bytes = key.len();
        if key_bytes > MAX_METADATA_KEY_BYTES {
            vortex_bail!(
                "Vortex metadata key {key:?} is {key_bytes} bytes, but keys must be at most {} bytes; files may contain at most {} metadata segments",
                MAX_METADATA_KEY_BYTES,
                MAX_METADATA_SEGMENTS
            );
        }
    }

    Ok(())
}

/// An async API for writing Vortex files.
pub struct Writer<'w> {
    // The input channel for sending arrays to the writer.
    arrays: Option<kanal::AsyncSender<VortexResult<ArrayRef>>>,
    // The writer task that ultimately produces the footer.
    future: Fuse<LocalBoxFuture<'w, VortexResult<WriteSummary>>>,
    // The bytes written so far.
    bytes_written: Arc<AtomicU64>,
    // The buffered bytes accounting shared with the layout strategies for this write.
    buffered_bytes: BufferedBytesTracker,
}

impl Writer<'_> {
    /// Push a new chunk into the writer.
    pub async fn push(&mut self, chunk: ArrayRef) -> VortexResult<()> {
        let arrays = self.arrays.clone().vortex_expect("missing arrays sender");
        let send_fut = async move { arrays.send(Ok(chunk)).await }.fuse();
        pin_mut!(send_fut);

        // We poll the writer future to continue writing bytes to the output, while waiting for
        // enough room to push the next chunk into the channel.
        select! {
            result = send_fut => {
                // If the send future failed, the writer has failed or panicked.
                if result.is_err() {
                    return Err(self.handle_failed_task().await);
                }
            },
            result = &mut self.future => {
                // Under normal operation, the writer future should never complete until
                // finish() is called. Therefore, we can assume the writer has failed.
                // The writer future has failed, we need to propagate the error.
                match result {
                    Ok(_) => vortex_bail!("Internal error: writer future completed early"),
                    Err(e) => return Err(e),
                }
            }
        }

        Ok(())
    }

    /// Push an entire [`ArrayStream`] into the writer, consuming it.
    ///
    /// A task is spawned to consume the stream and push it into the writer, with the current
    /// thread being used to write buffers to the output.
    pub async fn push_stream(&mut self, mut stream: SendableArrayStream) -> VortexResult<()> {
        let arrays = self.arrays.clone().vortex_expect("missing arrays sender");
        let stream_fut = async move {
            while let Some(chunk) = stream.next().await {
                arrays.send(chunk).await?;
            }
            Ok::<_, kanal::SendError>(())
        }
        .fuse();
        pin_mut!(stream_fut);

        // We poll the writer future to continue writing bytes to the output, while waiting for
        // enough room to push the stream into the channel.
        select! {
            result = stream_fut => {
                if let Err(_send_err) = result {
                    // If the send future failed, the writer has failed or panicked.
                    return Err(self.handle_failed_task().await);
                }
            }

            result = &mut self.future => {
                // Under normal operation, the writer future should never complete until
                // finish() is called. Therefore, we can assume the writer has failed.
                // The writer future has failed, we need to propagate the error.
                match result {
                    Ok(_) => vortex_bail!("Internal error: writer future completed early"),
                    Err(e) => return Err(e),
                }
            }
        }

        Ok(())
    }

    /// Returns the number of bytes written to the file so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
    }

    /// Returns the number of bytes currently buffered by the layout writers.
    pub fn buffered_bytes(&self) -> u64 {
        self.buffered_bytes.buffered_bytes()
    }

    /// Finish writing the Vortex file, flushing any remaining buffers and returning the
    /// new file's footer.
    pub async fn finish(mut self) -> VortexResult<WriteSummary> {
        // Drop the input channel to signal EOF.
        drop(self.arrays.take());

        // Await the future task.
        self.future.await
    }

    /// Assuming the writer task has failed, await it to get the error.
    async fn handle_failed_task(&mut self) -> VortexError {
        match (&mut self.future).await {
            Ok(_) => vortex_err!(
                "Internal error: writer task completed successfully but write future finished early"
            ),
            Err(e) => e,
        }
    }
}

/// Blocking adapter for [`VortexWriteOptions`].
pub struct BlockingWrite<'rt, B: BlockingRuntime> {
    options: VortexWriteOptions,
    runtime: &'rt B,
}

impl<'rt, B: BlockingRuntime> BlockingWrite<'rt, B> {
    /// Write a Vortex file into the given `Write` sink.
    ///
    /// The iterator is converted to an [`ArrayStream`] and driven to completion on
    /// the configured blocking runtime.
    pub fn write<W: Write + Unpin + Send>(
        self,
        write: W,
        iter: impl ArrayIterator + Send + 'static,
    ) -> VortexResult<WriteSummary> {
        self.runtime.block_on(async move {
            self.options
                .write(BlockingWriteAdapter(write), iter.into_array_stream())
                .await
        })
    }

    /// Create a blocking push-based writer for chunks with dtype `dtype`.
    pub fn writer<'w, W: Write + Unpin + Send + 'w>(
        self,
        write: W,
        dtype: DType,
    ) -> BlockingWriter<'rt, 'w, B> {
        BlockingWriter {
            writer: self.options.writer(BlockingWriteAdapter(write), dtype),
            runtime: self.runtime,
        }
    }
}

/// A blocking adapter around a [`Writer`], allowing incremental writing of arrays to a Vortex file.
pub struct BlockingWriter<'rt, 'w, B: BlockingRuntime> {
    runtime: &'rt B,
    writer: Writer<'w>,
}

impl<B: BlockingRuntime> BlockingWriter<'_, '_, B> {
    /// Push one array chunk into the file.
    pub fn push(&mut self, chunk: ArrayRef) -> VortexResult<()> {
        self.runtime.block_on(self.writer.push(chunk))
    }

    /// Returns the number of bytes written to the sink so far.
    pub fn bytes_written(&self) -> u64 {
        self.writer.bytes_written()
    }

    /// Returns the number of bytes currently buffered by layout strategies.
    pub fn buffered_bytes(&self) -> u64 {
        self.writer.buffered_bytes()
    }

    /// Finish writing and return the written file summary.
    pub fn finish(self) -> VortexResult<WriteSummary> {
        self.runtime.block_on(self.writer.finish())
    }
}

// TODO(ngates): this blocking API may change, for now we just run blocking I/O inline.
struct BlockingWriteAdapter<W>(W);

impl<W: Write + Unpin + Send> VortexWrite for BlockingWriteAdapter<W> {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        self.0.write_all(buffer.as_slice())?;
        Ok(buffer)
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> + Send {
        ready(self.0.flush())
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> + Send {
        ready(Ok(()))
    }
}

/// Summary returned after a Vortex file is written.
pub struct WriteSummary {
    footer: Footer,
    size: u64,
    // TODO(ngates): add a checksum
}

impl WriteSummary {
    /// The footer of the written Vortex file.
    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    /// The total size of the written Vortex file in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The total number of rows in the written Vortex file.
    pub fn row_count(&self) -> u64 {
        self.footer.row_count()
    }

    /// Returns the compressed size in bytes of each top-level column in schema order.
    ///
    /// A column's size includes every physical segment attributed to its layout subtree,
    /// including auxiliary segments such as zone maps and dictionaries; see
    /// [`Footer::compressed_field_sizes`] for the exact attribution semantics and for sizes of
    /// nested fields. Bytes not attributable to a specific column (e.g. top-level struct
    /// validity) are not included in any column's size.
    ///
    /// For a non-struct file, the returned vector contains a single entry for the root column.
    pub fn compressed_column_sizes(&self) -> VortexResult<Vec<u64>> {
        let sizes = self.footer.compressed_field_sizes()?;
        let Some(fields) = self.footer.dtype().as_struct_fields_opt() else {
            return Ok(vec![sizes.total()]);
        };
        Ok(fields
            .names()
            .iter()
            .map(|name| {
                sizes
                    .get(&FieldPath::from_name(name.clone()))
                    .unwrap_or_default()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::VTable;
    use vortex_array::array_session;
    use vortex_array::arrays::Bool;
    use vortex_array::arrays::Primitive;
    use vortex_buffer::ByteBuffer;
    use vortex_edition::ComponentKind;
    use vortex_edition::Edition;
    use vortex_edition::EditionDeclaration;
    use vortex_edition::EditionId;
    use vortex_edition::EditionInclusion;
    use vortex_edition::EditionMember;
    use vortex_edition::EditionSession;
    use vortex_edition::EditionSessionExt;

    use super::*;

    #[test]
    fn array_context_only_permits_enabled_encodings() -> Result<(), vortex_edition::EditionError> {
        const EDITION: EditionId = EditionId::new("test", 2026, 7, 0);
        static DECLARATION: EditionDeclaration = EditionDeclaration {
            edition: Edition {
                id: EDITION,
                min_vortex_version: None,
            },
            added: &[EditionMember::array(&"vortex.primitive")],
        };

        let session = array_session().with::<EditionSession>();
        session.register_edition(&DECLARATION)?;
        session.enable_edition(EDITION)?;

        let enabled_encoding_ids = session.enabled_component_ids(ComponentKind::Array);
        let ctx = ArrayContext::new(enabled_encoding_ids.clone())
            .with_allowed_ids(enabled_encoding_ids.into_iter().collect());
        assert_eq!(ctx.to_ids(), [Primitive.id()]);
        assert!(ctx.intern(&Bool.id()).is_none());
        Ok(())
    }

    /// This test edition declares only arrays, so every other kind must forbid all components.
    #[test]
    fn kind_filters_are_active_when_empty() -> Result<(), vortex_edition::EditionError> {
        const EDITION: EditionId = EditionId::new("test", 2026, 8, 0);
        static ARRAYS_ONLY: EditionDeclaration = EditionDeclaration {
            edition: Edition {
                id: EDITION,
                min_vortex_version: None,
            },
            added: &[EditionMember::array(&"vortex.primitive")],
        };

        let session = array_session().with::<EditionSession>();
        session.register_edition(&ARRAYS_ONLY)?;
        session.enable_edition(EDITION)?;
        assert!(edition_filter(&session, ComponentKind::Layout).is_empty());
        assert!(edition_filter(&session, ComponentKind::DType).is_empty());
        assert!(edition_filter(&session, ComponentKind::Aggregate).is_empty());
        assert!(
            new_layout_context(&session)
                .intern(&"vortex.flat".into())
                .is_none()
        );

        session.editions().declare_inclusion(EditionInclusion::new(
            ComponentKind::Aggregate,
            "vortex.min",
            EDITION,
        ))?;
        let allowed = edition_filter(&session, ComponentKind::Aggregate);
        assert_eq!(allowed.len(), 1);
        assert!(allowed.contains(&Id::from("vortex.min")));
        Ok(())
    }

    #[test]
    fn dtype_filter_checks_nested_extension_dtypes() -> VortexResult<()> {
        use vortex_array::dtype::Nullability;
        use vortex_array::extension::datetime::Date;
        use vortex_array::extension::datetime::Time;
        use vortex_array::extension::datetime::TimeUnit;

        const EDITION: EditionId = EditionId::new("test", 2026, 8, 0);
        static DECLARATION: EditionDeclaration = EditionDeclaration {
            edition: Edition {
                id: EDITION,
                min_vortex_version: None,
            },
            added: &[EditionMember::dtype(&"vortex.date")],
        };

        let session = array_session().with::<EditionSession>();
        session
            .register_edition(&DECLARATION)
            .map_err(|error| vortex_err!("{error}"))?;
        session
            .enable_edition(EDITION)
            .map_err(|error| vortex_err!("{error}"))?;

        let date = DType::Extension(Date::new(TimeUnit::Days, Nullability::NonNullable).erased());
        let nested = DType::struct_([("date", date)], Nullability::NonNullable);
        validate_dtype_editions(&session, &nested)?;

        let time =
            DType::Extension(Time::new(TimeUnit::Seconds, Nullability::NonNullable).erased());
        let error = validate_dtype_editions(&session, &time)
            .expect_err("vortex.time is not in the enabled edition");
        assert!(error.to_string().contains("vortex.time"));
        Ok(())
    }

    fn write_options_with_keys(keys: &[String]) -> VortexWriteOptions {
        array_session().write_options().with_metadata_segments(
            keys.iter()
                .map(|key| (key.clone(), ByteBuffer::copy_from(b"value"))),
        )
    }

    #[rstest]
    #[case::empty_key(vec![String::new()], "non-empty")]
    #[case::oversized_key(vec!["k".repeat(MAX_METADATA_KEY_BYTES + 1)], "keys must be at most")]
    // The cap is on bytes, not characters.
    #[case::oversized_multibyte_key(
        vec!["é".repeat(MAX_METADATA_KEY_BYTES / "é".len() + 1)],
        "keys must be at most"
        )]
    #[case::too_many_segments(
        (0..=MAX_METADATA_SEGMENTS).map(|idx| format!("key-{idx}")).collect(),
        "at most 16 metadata segments"
        )]
    fn validate_metadata_rejects(#[case] keys: Vec<String>, #[case] expected: &str) {
        let Err(error) = write_options_with_keys(&keys).validate_metadata() else {
            panic!("metadata must be rejected for {keys:?}");
        };
        assert!(
            error.to_string().contains(expected),
            "error should mention {expected:?}, got: {error}"
        );
    }

    #[test]
    fn validate_metadata_accepts_the_limits() -> VortexResult<()> {
        // Distinct keys, each exactly at the key-length cap.
        let keys = (0..MAX_METADATA_SEGMENTS)
            .map(|idx| format!("{idx:0>width$}", width = MAX_METADATA_KEY_BYTES))
            .collect::<Vec<_>>();
        write_options_with_keys(&keys).validate_metadata()
    }

    #[test]
    fn validate_metadata_accepts_no_metadata() -> VortexResult<()> {
        write_options_with_keys(&[]).validate_metadata()
    }
}
