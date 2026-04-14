// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A CUDA-optimized flat layout that inlines small constant array buffers into layout metadata.

use std::collections::BTreeSet;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use vortex::array::ArrayContext;
use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::DeserializeMetadata;
use vortex::array::MaskFuture;
use vortex::array::ProstMetadata;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Constant;
use vortex::array::expr::Expression;
use vortex::array::expr::stats::Precision;
use vortex::array::expr::stats::Stat;
use vortex::array::expr::stats::StatsProvider;
use vortex::array::normalize::NormalizeOptions;
use vortex::array::normalize::Operation;
use vortex::array::serde::SerializeOptions;
use vortex::array::serde::SerializedArray;
use vortex::array::stats::StatsSetRef;
use vortex::buffer::BufferString;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::dtype::FieldMask;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_panic;
use vortex::layout::IntoLayout;
use vortex::layout::LayoutChildType;
use vortex::layout::LayoutChildren;
use vortex::layout::LayoutEncodingRef;
use vortex::layout::LayoutId;
use vortex::layout::LayoutReader;
use vortex::layout::LayoutReaderRef;
use vortex::layout::LayoutRef;
use vortex::layout::LayoutStrategy;
use vortex::layout::VTable;
use vortex::layout::layouts::SharedArrayFuture;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSinkRef;
use vortex::layout::segments::SegmentSource;
use vortex::layout::sequence::SendableSequentialStream;
use vortex::layout::sequence::SequencePointer;
use vortex::layout::vtable;
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarTruncation;
use vortex::scalar::lower_bound;
use vortex::scalar::upper_bound;
use vortex::session::VortexSession;
use vortex::session::registry::ReadContext;
use vortex::utils::aliases::hash_map::HashMap;
use vortex::utils::aliases::hash_set::HashSet;

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
    ctx: ReadContext,
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
    pub fn array_ctx(&self) -> &ReadContext {
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

impl VTable for CudaFlat {
    type Layout = CudaFlatLayout;
    type Encoding = CudaFlatLayoutEncoding;
    type Metadata = ProstMetadata<CudaFlatLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new("vortex.cuda_flat")
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
            array: Default::default(),
        }))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        _children: &dyn LayoutChildren,
        ctx: &ReadContext,
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
    array: OnceLock<SharedArrayFuture>,
}

impl CudaFlatReader {
    fn array_future(&self) -> SharedArrayFuture {
        self.array
            .get_or_init(|| {
                let row_count = usize::try_from(self.layout.row_count)
                    .vortex_expect("row count must fit in usize");

                let segment_fut = self.segment_source.request(self.layout.segment_id);

                let ctx = self.layout.ctx.clone();
                let session = self.session.clone();
                let dtype = self.layout.dtype.clone();
                let array_tree = self.layout.array_tree.clone();
                let host_buffers = Arc::clone(&self.layout.host_buffers);

                async move {
                    let segment = segment_fut.await?;
                    let parts = SerializedArray::from_flatbuffer_and_segment_with_overrides(
                        array_tree,
                        segment,
                        &host_buffers,
                    )?;
                    parts
                        .decode(&dtype, row_count, &ctx, &session)
                        .map_err(Arc::new)
                }
                .boxed()
                .shared()
            })
            .clone()
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
        let name = Arc::clone(&self.name);
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
        let name = Arc::clone(&self.name);
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

/// A [`LayoutStrategy`] that writes a [`CudaFlatLayout`] with constant array buffers inlined
/// into layout metadata for host-side access during GPU reads.
#[derive(Clone)]
pub struct CudaFlatLayoutStrategy {
    /// Whether to include padding for memory-mapped reads.
    pub include_padding: bool,
    /// Maximum length of variable length statistics.
    pub max_variable_length_statistics_size: usize,
    /// Optional set of allowed array encodings for normalization.
    pub allowed_encodings: Option<HashSet<ArrayId>>,
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

    pub fn with_allow_encodings(mut self, allow_encodings: HashSet<ArrayId>) -> Self {
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
        session: &VortexSession,
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
        let host_buffers = extract_constant_buffers(&chunk);

        let buffers = chunk.serialize(
            &ctx,
            session,
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
            ctx: ReadContext::new(ctx.to_ids()),
            array_tree,
            host_buffers: Arc::new(host_buffer_map),
        }
        .into_layout())
    }
}

/// Walk the array tree depth-first and extract buffer data for all `ConstantArray` nodes.
///
/// The buffer ordering matches `Array::serialize()` because both use depth-first traversal.
fn extract_constant_buffers(chunk: &ArrayRef) -> Vec<InlinedBuffer> {
    let mut result = Vec::new();
    let mut buffer_idx = 0u32;
    for array in chunk.depth_first_traversal() {
        let n = array.nbuffers();
        if array.encoding_id() == Constant.id() {
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
    use vortex::layout::session::LayoutSessionExt;
    session
        .layouts()
        .register(LayoutEncodingRef::new_ref(CudaFlatLayoutEncoding.as_ref()));
}
