// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::io;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Poll;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::poll_fn;
use futures::future::ready;
use futures::pin_mut;
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
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::ByteBuffer;
use vortex_edition::ComponentKind;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::IoBuf;
use vortex_io::VortexWrite;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::BufferedBytesTracker;
use vortex_layout::LayoutContext;
use vortex_layout::LayoutStrategy;
use vortex_layout::LayoutWriterActor;
use vortex_layout::LayoutWriterContext;
use vortex_layout::layouts::file_stats::FileStatsAccumulator;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequencePointer;
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
/// All write strategies are restricted to the components in the session's enabled editions: an array,
/// layout, or zone-map aggregate outside them fails the write. A kind the editions declare nothing of
/// is left unrestricted.
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
    /// The set of encodings permitted in the file is snapshotted from the session's array registry
    /// here, so encodings registered after this call are not written.
    pub async fn write<W: VortexWrite + Unpin, S: ArrayStream + Send + 'static>(
        self,
        write: W,
        stream: S,
    ) -> VortexResult<WriteSummary> {
        let dtype = stream.dtype().clone();
        let mut writer = self.writer(write, dtype)?;
        pin_mut!(stream);
        while let Some(chunk) = stream.next().await {
            writer.write(chunk?).await?;
        }
        writer.close().await
    }

    /// Create a push-based [`Writer`] that can be used to incrementally write arrays to the file.
    ///
    /// This follows the same lifecycle as other columnar file writers: call [`Writer::write`] for
    /// each chunk, then call [`Writer::close`] to flush remaining buffers and receive the
    /// [`WriteSummary`]. Each chunk must have dtype `dtype`.
    pub fn writer<W: VortexWrite + Unpin>(self, write: W, dtype: DType) -> VortexResult<Writer<W>> {
        validate_metadata_segments(&self.metadata)?;
        let mut ctx = LayoutWriterContext::new(new_array_context(&self.session))
            .with_buffered_bytes_tracker(self.buffered_bytes.clone());
        if let Some(allowed) = edition_filter(&self.session, ComponentKind::Aggregate) {
            ctx = ctx.with_allowed_aggregates(allowed);
        }
        let layout_ctx = new_layout_context(&self.session);
        let (buffers_send, buffers) = kanal::bounded_async(1);
        let segment_sink = Arc::new(BufferedSegmentSink::new(
            buffers_send,
            MAGIC_BYTES.len() as u64,
        ));
        let layout = self.strategy.new_writer(
            ctx.clone(),
            Arc::<BufferedSegmentSink>::clone(&segment_sink),
            dtype.clone(),
            &self.session,
        )?;
        let layout =
            LayoutWriterActor::spawn(layout, self.buffered_bytes.clone(), &self.session.handle());
        let sequence = SequenceId::root();
        let file_stats = FileStatsAccumulator::new(
            &dtype,
            self.file_statistics.clone().into(),
            self.max_variable_length_statistics_size,
            &self.session,
        );
        let write = CountingVortexWrite::new(write);
        let bytes_written = write.counter();
        Ok(Writer {
            write,
            buffers,
            segment_sink,
            layout: Some(layout),
            sequence,
            ctx,
            layout_ctx,
            dtype,
            file_stats,
            file_statistics: self.file_statistics,
            metadata: self.metadata,
            exclude_dtype: self.exclude_dtype,
            position: 0,
            bytes_written,
            buffered_bytes: self.buffered_bytes,
        })
    }
}

fn new_array_context(session: &VortexSession) -> ArrayContext {
    // NOTE(os): Setup an array context that already has all known encodings pre-populated.
    // This is preferred for now over having an empty context here, because only the
    // serialised array order is deterministic. The serialisation of arrays are done
    // parallel and with an empty context they can register their encodings to the context
    // in different order, changing the written bytes from run to run.
    let enabled_encoding_ids = session.enabled_component_ids(ComponentKind::Array);
    ArrayContext::new(enabled_encoding_ids.iter().cloned().sorted().collect())
        // Only permit encodings known to the session.
        .with_allowed_ids(enabled_encoding_ids.into_iter().collect())
}

/// The ids of `kind` the enabled editions permit, or `None` when they declare none.
///
/// Editions declare array members today. A kind with no declared members carries no guarantee
/// to enforce, so its filter stays disarmed rather than forbidding every layout or aggregate;
/// declaring the first member of a kind arms it. Arrays are always restricted to the enabled
/// set, since that is the set the file format's read-forever guarantee is written against.
fn edition_filter(session: &VortexSession, kind: ComponentKind) -> Option<HashSet<Id>> {
    let ids: HashSet<Id> = session.enabled_component_ids(kind).into_iter().collect();
    (!ids.is_empty()).then_some(ids)
}

/// The context every layout in the file is interned through, restricted to the layouts the
/// enabled editions permit.
fn new_layout_context(session: &VortexSession) -> LayoutContext {
    match edition_filter(session, ComponentKind::Layout) {
        Some(allowed) => LayoutContext::default().with_allowed_ids(allowed),
        None => LayoutContext::default(),
    }
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
pub struct Writer<W> {
    write: CountingVortexWrite<W>,
    buffers: kanal::AsyncReceiver<ByteBuffer>,
    segment_sink: Arc<BufferedSegmentSink>,
    layout: Option<LayoutWriterActor>,
    sequence: SequencePointer,
    ctx: LayoutWriterContext,
    layout_ctx: LayoutContext,
    dtype: DType,
    file_stats: FileStatsAccumulator,
    file_statistics: Vec<Stat>,
    metadata: HashMap<String, ByteBuffer>,
    exclude_dtype: bool,
    position: u64,
    bytes_written: Arc<AtomicU64>,
    buffered_bytes: BufferedBytesTracker,
}

impl<W: VortexWrite + Unpin> Writer<W> {
    async fn ensure_started(&mut self) -> VortexResult<()> {
        if self.position == 0 {
            self.write
                .write_all(ByteBuffer::copy_from(MAGIC_BYTES))
                .await?;
            self.position = MAGIC_BYTES.len() as u64;
        }
        Ok(())
    }

    async fn write_buffer(
        write: &mut CountingVortexWrite<W>,
        position: &mut u64,
        buffer: ByteBuffer,
    ) -> VortexResult<()> {
        if !buffer.is_empty() {
            *position += buffer.len() as u64;
            write.write_all(buffer).await?;
        }
        Ok(())
    }

    async fn drive_layout<T>(
        write: &mut CountingVortexWrite<W>,
        buffers: &kanal::AsyncReceiver<ByteBuffer>,
        position: &mut u64,
        operation: impl Future<Output = VortexResult<T>>,
        channel_closed_message: &'static str,
    ) -> VortexResult<T> {
        enum Event<B> {
            Buffer(B),
            Done,
        }

        let operation = operation.fuse();
        pin_mut!(operation);
        let mut completed = None;

        loop {
            let receive = buffers.recv().fuse();
            pin_mut!(receive);
            let event = poll_fn(|cx| {
                if let Poll::Ready(buffer) = receive.as_mut().poll(cx) {
                    return Poll::Ready(Event::Buffer(buffer));
                }

                if completed.is_none()
                    && let Poll::Ready(result) = operation.as_mut().poll(cx)
                {
                    completed = Some(result);
                }

                // Poll again because polling the layout operation may have completed a send into
                // the receive future that was pending immediately above.
                if let Poll::Ready(buffer) = receive.as_mut().poll(cx) {
                    return Poll::Ready(Event::Buffer(buffer));
                }

                if completed.is_some() {
                    Poll::Ready(Event::Done)
                } else {
                    Poll::Pending
                }
            })
            .await;

            match event {
                Event::Buffer(buffer) => {
                    let buffer = buffer.map_err(|_| vortex_err!("{channel_closed_message}"))?;
                    Self::write_buffer(write, position, buffer).await?;
                }
                Event::Done => {
                    return completed.take().vortex_expect("layout operation completed");
                }
            }
        }
    }

    /// Write a new chunk.
    ///
    /// Returns an error without writing the chunk if its dtype does not match the dtype used to
    /// construct the writer.
    pub async fn write(&mut self, chunk: ArrayRef) -> VortexResult<()> {
        if chunk.dtype() != &self.dtype {
            vortex_bail!(
                "Writer expected array with dtype {}, but received {}",
                self.dtype,
                chunk.dtype()
            );
        }

        if chunk.is_empty() {
            return Ok(());
        }
        self.ensure_started().await?;
        self.file_stats.push(&chunk)?;

        let sequence_id = self.sequence.advance();
        let layout = self.layout.as_mut().vortex_expect("layout writer present");
        let write = &mut self.write;
        let buffers = &self.buffers;
        let position = &mut self.position;
        Self::drive_layout(
            write,
            buffers,
            position,
            layout.write(sequence_id, chunk),
            "segment buffer channel closed while writing",
        )
        .await?;
        while let Ok(Some(buffer)) = buffers.try_recv() {
            Self::write_buffer(write, position, buffer).await?;
        }
        Ok(())
    }

    /// Push a new chunk into the writer.
    ///
    /// This is an alias for [`Self::write`].
    pub async fn push(&mut self, chunk: ArrayRef) -> VortexResult<()> {
        self.write(chunk).await
    }

    /// Push an entire [`ArrayStream`] into the writer, consuming it.
    pub async fn push_stream(&mut self, mut stream: SendableArrayStream) -> VortexResult<()> {
        while let Some(chunk) = stream.next().await {
            self.write(chunk?).await?;
        }
        Ok(())
    }

    /// Returns the number of bytes written to the file so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
    }

    /// Returns the logical byte size of arrays currently retained by layout strategies.
    ///
    /// This includes arrays queued for asynchronous layout work. It does not include allocator
    /// overhead, statistics-builder state, or buffering performed by the output sink.
    pub fn buffered_bytes(&self) -> u64 {
        self.buffered_bytes.buffered_bytes()
    }

    /// Finish writing the Vortex file, flushing any remaining buffers and returning the
    /// new file's footer.
    pub async fn finish(mut self) -> VortexResult<WriteSummary> {
        self.ensure_started().await?;
        let mut layout = self.layout.take().vortex_expect("layout writer present");
        let sequence_id = self.sequence.advance();
        let write = &mut self.write;
        let buffers = &self.buffers;
        let position = &mut self.position;
        let layout = Self::drive_layout(
            write,
            buffers,
            position,
            async move {
                layout.finish(sequence_id).await?;
                layout.take_layout()
            },
            "segment buffer channel closed while closing",
        )
        .await?;
        while let Ok(Some(buffer)) = buffers.try_recv() {
            Self::write_buffer(write, position, buffer).await?;
        }

        let statistics = if self.file_statistics.is_empty() {
            None
        } else {
            Some(FileStatistics::new_with_dtype(
                self.file_stats.stats_sets().into(),
                &self.dtype,
            ))
        };
        let mut footer = Footer::new(
            layout,
            self.segment_sink.segment_specs(),
            statistics,
            ReadContext::new(self.ctx.array_ctx().to_ids()),
        );
        let (footer_buffers, metadata, approx_byte_size) = footer
            .clone()
            .into_serializer()
            .with_layout_context(self.layout_ctx)
            .with_metadata_segments(self.metadata)
            .with_offset(self.position)
            .with_exclude_dtype(self.exclude_dtype)
            .serialize_with_metadata()?;
        footer = footer
            .with_metadata_segments(metadata)
            .with_approx_byte_size(approx_byte_size);
        for buffer in footer_buffers {
            Self::write_buffer(&mut self.write, &mut self.position, buffer).await?;
        }
        self.write.flush().await?;
        Ok(WriteSummary {
            footer,
            size: self.position,
        })
    }

    /// Close the writer, flushing any remaining buffers and returning the file summary.
    ///
    /// This is an alias for [`Self::finish`].
    pub async fn close(self) -> VortexResult<WriteSummary> {
        self.finish().await
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
    pub fn writer<W: Write + Unpin + Send>(
        self,
        write: W,
        dtype: DType,
    ) -> VortexResult<BlockingWriter<'rt, B, W>> {
        Ok(BlockingWriter {
            writer: self.options.writer(BlockingWriteAdapter(write), dtype)?,
            runtime: self.runtime,
        })
    }
}

/// A blocking adapter around a [`Writer`], allowing incremental writing of arrays to a Vortex file.
pub struct BlockingWriter<'rt, B: BlockingRuntime, W> {
    runtime: &'rt B,
    writer: Writer<BlockingWriteAdapter<W>>,
}

impl<B: BlockingRuntime, W: Write + Unpin + Send> BlockingWriter<'_, B, W> {
    /// Push one array chunk into the file.
    pub fn push(&mut self, chunk: ArrayRef) -> VortexResult<()> {
        self.runtime.block_on(self.writer.push(chunk))
    }

    /// Returns the number of bytes written to the sink so far.
    pub fn bytes_written(&self) -> u64 {
        self.writer.bytes_written()
    }

    /// Returns the logical byte size of arrays currently retained by layout strategies.
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

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(self.0.flush())
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
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
    fn push_writer_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Writer<io::Cursor<Vec<u8>>>>();
    }

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

    /// Editions declare array members today, so the layout and aggregate filters must stay
    /// disarmed until something declares members of those kinds — arming them on an empty
    /// set would forbid every layout and drop every zone-map aggregate.
    #[test]
    fn kind_filters_arm_on_declaration() -> Result<(), vortex_edition::EditionError> {
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
        assert!(edition_filter(&session, ComponentKind::Layout).is_none());
        assert!(edition_filter(&session, ComponentKind::Aggregate).is_none());
        assert!(
            new_layout_context(&session)
                .intern(&"vortex.flat".into())
                .is_some()
        );

        session.editions().declare_inclusion(EditionInclusion::new(
            ComponentKind::Aggregate,
            "vortex.min",
            EDITION,
        ))?;
        let allowed = edition_filter(&session, ComponentKind::Aggregate)
            .vortex_expect("aggregate member is declared");
        assert_eq!(allowed.len(), 1);
        assert!(allowed.contains(&Id::from("vortex.min")));
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
