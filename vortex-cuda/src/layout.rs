// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A CUDA-optimized flat layout that inlines small constant array buffers into layout metadata.

use std::collections::BTreeSet;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use flatbuffers::root;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::ArrayVisitor;
use vortex_array::ArrayVisitorExt;
use vortex_array::DeserializeMetadata;
use vortex_array::MaskFuture;
use vortex_array::ProstMetadata;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantVTable;
use vortex_array::buffer::BufferHandle;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::normalize::NormalizeOptions;
use vortex_array::normalize::Operation;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarTruncation;
use vortex_array::scalar::lower_bound;
use vortex_array::scalar::upper_bound;
use vortex_array::serde::ArrayParts;
use vortex_array::serde::SerializeOptions;
use vortex_array::session::ArrayRegistry;
use vortex_array::stats::StatsSetRef;
use vortex_buffer::Alignment;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_dtype::FieldMask;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::array as fba;
use vortex_layout::IntoLayout;
use vortex_layout::LayoutChildType;
use vortex_layout::LayoutChildren;
use vortex_layout::LayoutEncodingRef;
use vortex_layout::LayoutId;
use vortex_layout::LayoutReader;
use vortex_layout::LayoutReaderRef;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::VTable;
use vortex_layout::layouts::SharedArrayFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSinkRef;
use vortex_layout::segments::SegmentSource;
use vortex_layout::sequence::SendableSequentialStream;
use vortex_layout::sequence::SequencePointer;
use vortex_layout::vtable;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

/// A buffer inlined into layout metadata for host-side access.
#[derive(Clone, prost::Message)]
pub struct InlinedBuffer {
    #[prost(uint32, tag = "1")]
    pub buffer_index: u32,
    #[prost(bytes, tag = "2")]
    pub data: Vec<u8>,
}

/// Protobuf metadata for [`CudaFlatLayout`].
#[derive(prost::Message)]
pub struct CudaFlatLayoutMetadata {
    #[prost(bytes, tag = "1")]
    pub array_encoding_tree: Vec<u8>,
    #[prost(message, repeated, tag = "2")]
    pub host_buffers: Vec<InlinedBuffer>,
}

vtable!(CudaFlat);

#[derive(Debug)]
pub struct CudaFlatLayoutEncoding;

#[derive(Clone, Debug)]
pub struct CudaFlatLayout {
    row_count: u64,
    dtype: DType,
    segment_id: SegmentId,
    ctx: ArrayContext,
    array_tree: ByteBuffer,
    /// Small buffers kept on host, keyed by global buffer index.
    host_buffers: Arc<HashMap<u32, ByteBuffer>>,
}

impl CudaFlatLayout {
    #[inline]
    pub fn segment_id(&self) -> SegmentId {
        self.segment_id
    }

    #[inline]
    pub fn array_ctx(&self) -> &ArrayContext {
        &self.ctx
    }

    #[inline]
    pub fn array_tree(&self) -> &ByteBuffer {
        &self.array_tree
    }

    #[inline]
    pub fn host_buffers(&self) -> &Arc<HashMap<u32, ByteBuffer>> {
        &self.host_buffers
    }
}

impl VTable for CudaFlatVTable {
    type Layout = CudaFlatLayout;
    type Encoding = CudaFlatLayoutEncoding;
    type Metadata = ProstMetadata<CudaFlatLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.cuda_flat")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(CudaFlatLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        ProstMetadata(CudaFlatLayoutMetadata {
            array_encoding_tree: layout.array_tree.to_vec(),
            host_buffers: layout
                .host_buffers
                .iter()
                .map(|(&idx, buf)| InlinedBuffer {
                    buffer_index: idx,
                    data: buf.to_vec(),
                })
                .collect(),
        })
    }

    fn segment_ids(layout: &Self::Layout) -> Vec<SegmentId> {
        vec![layout.segment_id]
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        0
    }

    fn child(_layout: &Self::Layout, _idx: usize) -> VortexResult<LayoutRef> {
        vortex_bail!("CudaFlatLayout has no children");
    }

    fn child_type(_layout: &Self::Layout, _idx: usize) -> LayoutChildType {
        vortex_panic!("CudaFlatLayout has no children");
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(CudaFlatReader {
            layout: layout.clone(),
            name,
            segment_source,
            session: session.clone(),
        }))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        _children: &dyn LayoutChildren,
        ctx: &ArrayContext,
    ) -> VortexResult<Self::Layout> {
        if segment_ids.len() != 1 {
            vortex_bail!("CudaFlatLayout must have exactly one segment ID");
        }
        let host_buffers: HashMap<u32, ByteBuffer> = metadata
            .host_buffers
            .iter()
            .map(|hb| (hb.buffer_index, ByteBuffer::from(hb.data.clone())))
            .collect();
        Ok(CudaFlatLayout {
            row_count,
            dtype: dtype.clone(),
            segment_id: segment_ids[0],
            ctx: ctx.clone(),
            array_tree: ByteBuffer::from(metadata.array_encoding_tree.clone()),
            host_buffers: Arc::new(host_buffers),
        })
    }

    fn with_children(_layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        if !children.is_empty() {
            vortex_bail!("CudaFlatLayout has no children, got {}", children.len());
        }
        Ok(())
    }
}

// Threshold to order filter and apply expression, copied from FlatLayout.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

pub struct CudaFlatReader {
    layout: CudaFlatLayout,
    name: Arc<str>,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
}

impl CudaFlatReader {
    fn array_future(&self) -> SharedArrayFuture {
        let row_count =
            usize::try_from(self.layout.row_count).vortex_expect("row count must fit in usize");

        let segment_fut = self.segment_source.request(self.layout.segment_id);

        let ctx = self.layout.ctx.clone();
        let session = self.session.clone();
        let dtype = self.layout.dtype.clone();
        let array_tree = self.layout.array_tree.clone();
        let host_buffers = self.layout.host_buffers.clone();

        async move {
            let segment = segment_fut.await?;
            let parts = if host_buffers.is_empty() {
                ArrayParts::from_flatbuffer_and_segment(array_tree, segment)?
            } else {
                let buffers =
                    resolve_buffers_with_host_overrides(&array_tree, &segment, &host_buffers)?;
                ArrayParts::from_flatbuffer_with_buffers(array_tree, buffers)?
            };
            parts
                .decode(&dtype, row_count, &ctx, &session)
                .map_err(Arc::new)
        }
        .boxed()
        .shared()
    }
}

impl LayoutReader for CudaFlatReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        &self.layout.dtype
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        splits.insert(row_range.start + self.layout.row_count);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within CudaFlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within CudaFlatLayout size");
        let name = self.name.clone();
        let array = self.array_future();
        let expr = expr.clone();
        let session = self.session.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            let mut array = array.clone().await?;
            let mask = mask.await?;

            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone())?;
            }

            let array_mask = if mask.density() < EXPR_EVAL_THRESHOLD {
                let array = array.apply(&expr)?;
                let array = array.filter(mask.clone())?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.execute::<Mask>(&mut ctx)?;
                mask.intersect_by_rank(&array_mask)
            } else {
                let array = array.apply(&expr)?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.execute::<Mask>(&mut ctx)?;
                mask.bitand(&array_mask)
            };

            tracing::debug!(
                "CudaFlat mask evaluation {} - {} (mask = {}) => {}",
                name,
                expr,
                mask.density(),
                array_mask.density(),
            );

            Ok(array_mask)
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within CudaFlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within CudaFlatLayout size");
        let name = self.name.clone();
        let array = self.array_future();
        let expr = expr.clone();

        Ok(async move {
            tracing::debug!("CudaFlat array evaluation {} - {}", name, expr);

            let mut array = array.clone().await?;
            let mask = mask.await?;

            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone())?;
            }

            if !mask.all_true() {
                array = array.filter(mask)?;
            }

            array = array.apply(&expr)?;

            Ok(array)
        }
        .boxed())
    }
}

/// Resolve buffers from the array tree flatbuffer, substituting host overrides for specific
/// buffer indices.
fn resolve_buffers_with_host_overrides(
    array_tree: &ByteBuffer,
    segment: &BufferHandle,
    host_overrides: &HashMap<u32, ByteBuffer>,
) -> VortexResult<Vec<BufferHandle>> {
    let segment = segment.clone().ensure_aligned(Alignment::none())?;
    let fb_buffer = FlatBuffer::align_from(array_tree.clone());
    let fb_array = root::<fba::Array>(fb_buffer.as_ref())?;

    let mut offset = 0usize;
    fb_array
        .buffers()
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(idx, fb_buf)| {
            offset += fb_buf.padding() as usize;
            let buffer_len = fb_buf.length() as usize;

            let idx = u32::try_from(idx).vortex_expect("buffer count must fit in u32");
            let alignment = Alignment::from_exponent(fb_buf.alignment_exponent());
            let handle = if let Some(host_data) = host_overrides.get(&idx) {
                // Inlined host buffers lose segment padding alignment after protobuf
                // round-trip. Re-align here so downstream aligned casts are safe.
                BufferHandle::new_host(host_data.clone()).ensure_aligned(alignment)?
            } else {
                let buffer = segment.slice(offset..(offset + buffer_len));
                buffer.ensure_aligned(alignment)?
            };

            offset += buffer_len;
            Ok(handle)
        })
        .collect()
}

/// A [`LayoutStrategy`] that writes a [`CudaFlatLayout`] with constant array buffers inlined
/// into layout metadata for host-side access during GPU reads.
#[derive(Clone)]
pub struct CudaFlatLayoutStrategy {
    /// Whether to include padding for memory-mapped reads.
    pub include_padding: bool,
    /// Maximum length of variable length statistics.
    pub max_variable_length_statistics_size: usize,
    /// Optional set of allowed array encodings for normalization.
    pub allowed_encodings: Option<ArrayRegistry>,
}

impl Default for CudaFlatLayoutStrategy {
    fn default() -> Self {
        Self {
            include_padding: true,
            max_variable_length_statistics_size: 64,
            allowed_encodings: None,
        }
    }
}

impl CudaFlatLayoutStrategy {
    pub fn with_include_padding(mut self, include_padding: bool) -> Self {
        self.include_padding = include_padding;
        self
    }

    pub fn with_max_variable_length_statistics_size(mut self, size: usize) -> Self {
        self.max_variable_length_statistics_size = size;
        self
    }

    pub fn with_allow_encodings(mut self, allow_encodings: ArrayRegistry) -> Self {
        self.allowed_encodings = Some(allow_encodings);
        self
    }
}

fn truncate_scalar_stat<F: Fn(Scalar) -> Option<(Scalar, bool)>>(
    statistics: StatsSetRef<'_>,
    stat: Stat,
    truncation: F,
) {
    if let Some(sv) = statistics.get(stat) {
        if let Some((truncated_value, truncated)) = truncation(sv.into_inner()) {
            if truncated && let Some(v) = truncated_value.into_value() {
                statistics.set(stat, Precision::Inexact(v));
            }
        } else {
            statistics.clear(stat)
        }
    }
}

#[async_trait]
impl LayoutStrategy for CudaFlatLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        _eof: SequencePointer,
        _handle: vortex_io::runtime::Handle,
    ) -> VortexResult<LayoutRef> {
        let ctx = ctx.clone();
        let options = self.clone();
        let Some(chunk) = stream.next().await else {
            vortex_bail!("CudaFlatLayoutStrategy needs a single chunk");
        };
        let (sequence_id, chunk) = chunk?;
        let row_count = chunk.len() as u64;

        match chunk.dtype() {
            DType::Utf8(n) => {
                truncate_scalar_stat(chunk.statistics(), Stat::Min, |v| {
                    lower_bound(
                        BufferString::from_scalar(v)
                            .vortex_expect("utf8 scalar must be a BufferString"),
                        self.max_variable_length_statistics_size,
                        *n,
                    )
                });
                truncate_scalar_stat(chunk.statistics(), Stat::Max, |v| {
                    upper_bound(
                        BufferString::from_scalar(v)
                            .vortex_expect("utf8 scalar must be a BufferString"),
                        self.max_variable_length_statistics_size,
                        *n,
                    )
                });
            }
            DType::Binary(n) => {
                truncate_scalar_stat(chunk.statistics(), Stat::Min, |v| {
                    lower_bound(
                        ByteBuffer::from_scalar(v)
                            .vortex_expect("binary scalar must be a ByteBuffer"),
                        self.max_variable_length_statistics_size,
                        *n,
                    )
                });
                truncate_scalar_stat(chunk.statistics(), Stat::Max, |v| {
                    upper_bound(
                        ByteBuffer::from_scalar(v)
                            .vortex_expect("binary scalar must be a ByteBuffer"),
                        self.max_variable_length_statistics_size,
                        *n,
                    )
                });
            }
            _ => {}
        }

        let chunk = if let Some(allowed) = &options.allowed_encodings {
            chunk.normalize(&mut NormalizeOptions {
                allowed,
                operation: Operation::Error,
            })?
        } else {
            chunk
        };

        // Scan for constant array buffers before serialization (while data is still on host).
        let host_buffers = extract_constant_buffers(&*chunk);

        let buffers = chunk.serialize(
            &ctx,
            &SerializeOptions {
                offset: 0,
                include_padding: options.include_padding,
            },
        )?;
        assert!(buffers.len() >= 2);

        // Always store the array tree inline (the cuda path requires it for planning).
        let array_tree = buffers[buffers.len() - 2].clone();

        let segment_id = segment_sink.write(sequence_id, buffers).await?;

        let None = stream.next().await else {
            vortex_bail!("CudaFlatLayoutStrategy received stream with more than a single chunk");
        };

        let host_buffer_map: HashMap<u32, ByteBuffer> = host_buffers
            .iter()
            .map(|hb| (hb.buffer_index, ByteBuffer::from(hb.data.clone())))
            .collect();

        Ok(CudaFlatLayout {
            row_count,
            dtype: stream.dtype().clone(),
            segment_id,
            ctx: ctx.clone(),
            array_tree,
            host_buffers: Arc::new(host_buffer_map),
        }
        .into_layout())
    }
}

/// Walk the array tree depth-first and extract buffer data for all `ConstantArray` nodes.
///
/// The buffer ordering matches `Array::serialize()` because both use depth-first traversal.
fn extract_constant_buffers(chunk: &dyn Array) -> Vec<InlinedBuffer> {
    let mut result = Vec::new();
    let mut buffer_idx = 0u32;
    for array in chunk.depth_first_traversal() {
        let n = array.nbuffers();
        if array.encoding_id() == ConstantVTable::ID {
            for buf in array.buffers() {
                result.push(InlinedBuffer {
                    buffer_index: buffer_idx,
                    data: buf.to_vec(),
                });
                buffer_idx += 1;
            }
        } else {
            buffer_idx += u32::try_from(n).vortex_expect("buffer count must fit in u32");
        }
    }
    result
}

/// Register the [`CudaFlatLayoutEncoding`] in the session's layout registry.
///
/// Call this alongside [`crate::initialize_cuda`] when setting up a CUDA-enabled session.
pub fn register_cuda_layout(session: &VortexSession) {
    use vortex_layout::session::LayoutSessionExt;
    session
        .layouts()
        .register(LayoutEncodingRef::new_ref(CudaFlatLayoutEncoding.as_ref()));
}
