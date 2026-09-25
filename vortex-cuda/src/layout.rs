// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A CUDA-optimized flat layout that inlines small constant array buffers into layout metadata.

use std::any::Any;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Once;
use std::sync::OnceLock;

use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::MaskFuture;
use vortex::array::ProstMetadata;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Constant;
use vortex::array::expr::BoundExpression;
use vortex::array::expr::stats::Precision;
use vortex::array::expr::stats::Stat;
use vortex::array::expr::stats::StatsProvider;
use vortex::array::serde::SerializeOptions;
use vortex::array::serde::SerializedArray;
use vortex::array::stats::StatsSetRef;
use vortex::buffer::BufferString;
use vortex::buffer::ByteBuffer;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::DType;
use vortex::dtype::FieldMask;
use vortex::editions::ComponentKind;
use vortex::editions::Edition;
use vortex::editions::EditionDeclaration;
use vortex::editions::EditionFamily;
use vortex::editions::EditionId;
use vortex::editions::EditionMember;
use vortex::editions::EditionSessionExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_panic;
use vortex::file::WriteStrategyBuilder;
use vortex::layout::Layout;
use vortex::layout::LayoutChildType;
use vortex::layout::LayoutDeserializeArgs;
use vortex::layout::LayoutEncodingRef;
use vortex::layout::LayoutId;
use vortex::layout::LayoutParts;
use vortex::layout::LayoutReader;
use vortex::layout::LayoutReaderRef;
use vortex::layout::LayoutRef;
use vortex::layout::LayoutStrategy;
use vortex::layout::LayoutWriterContext;
use vortex::layout::RowSplits;
use vortex::layout::SplitRange;
use vortex::layout::VTable;
use vortex::layout::layout_children;
use vortex::layout::layouts::SharedArrayFuture;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSinkRef;
use vortex::layout::segments::SegmentSource;
use vortex::layout::sequence::SendableSequentialStream;
use vortex::layout::sequence::SequencePointer;
use vortex::layout::session::LayoutSessionExt;
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarTruncation;
use vortex::scalar::lower_bound;
use vortex::scalar::upper_bound;
use vortex::session::SessionExt;
use vortex::session::SessionVar;
use vortex::session::VortexSession;
use vortex::session::registry::CachedId;
use vortex::session::registry::ReadContext;
use vortex::utils::aliases::hash_map::HashMap;

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

/// CUDA flat layout vtable.
#[derive(Clone, Debug)]
pub struct CudaFlat;

/// Backwards-compatible plugin name.
pub use CudaFlat as CudaFlatLayoutEncoding;

/// CUDA-flat-specific data.
#[derive(Clone, Debug)]
pub struct CudaFlatData {
    segment_id: SegmentId,
    ctx: ReadContext,
    array_tree: ByteBuffer,
    /// Small buffers kept on host, keyed by global buffer index.
    host_buffers: Arc<HashMap<u32, ByteBuffer>>,
}

/// A CUDA-optimized terminal layout.
pub type CudaFlatLayout = Layout<CudaFlat>;

impl CudaFlatData {
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
    type LayoutData = CudaFlatData;
    type Metadata = ProstMetadata<CudaFlatLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.cuda_flat");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
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

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &CudaFlatLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        if args.segment_ids.len() != 1 {
            vortex_bail!("CudaFlatLayout must have exactly one segment ID");
        }
        if args.children.nchildren() != 0 {
            vortex_bail!("CudaFlatLayout must not have children");
        }
        let host_buffers = metadata
            .host_buffers
            .iter()
            .map(|hb| (hb.buffer_index, ByteBuffer::from(hb.data.clone())))
            .collect();
        Ok(CudaFlatData {
            segment_id: args.segment_ids[0],
            ctx: args.array_read_ctx.clone(),
            array_tree: ByteBuffer::from(metadata.array_encoding_tree.clone()),
            host_buffers: Arc::new(host_buffers),
        })
    }

    fn child_dtype(_layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        vortex_bail!("CudaFlatLayout has no child {idx}");
    }

    fn child_type(_layout: &Layout<Self>, _idx: usize) -> LayoutChildType {
        vortex_panic!("CudaFlatLayout has no children");
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        _ctx: &vortex::layout::LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(CudaFlatReader {
            layout: layout.clone(),
            name,
            segment_source,
            session: session.clone(),
            array: Default::default(),
        }))
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
                let row_count = usize::try_from(self.layout.row_count())
                    .vortex_expect("row count must fit in usize");

                let segment_fut = self.segment_source.request(self.layout.segment_id);

                let ctx = self.layout.ctx.clone();
                let session = self.session.clone();
                let dtype = self.layout.dtype().clone();
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
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        split_range.check_bounds(self.layout.row_count())?;
        splits.push(split_range.root_row_range().end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
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

            let mask_density = mask.density();
            let array_mask = if mask_density < EXPR_EVAL_THRESHOLD {
                let array = array.apply(&expr)?;
                let array = array.filter(mask.clone())?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.null_as_false().execute(&mut ctx)?;
                mask.intersect_by_rank(&array_mask)
            } else {
                let array = array.apply(&expr)?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.null_as_false().execute(&mut ctx)?;
                mask.bitand(&array_mask)
            };

            tracing::debug!(
                "CudaFlat mask evaluation {} - {} (mask = {}) => {}",
                name,
                expr,
                mask_density,
                array_mask.density(),
            );

            Ok(array_mask)
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
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

    fn as_any(&self) -> &dyn Any {
        self
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
}

impl Default for CudaFlatLayoutStrategy {
    fn default() -> Self {
        Self {
            include_padding: true,
            max_variable_length_statistics_size: 64,
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
}

fn truncate_scalar_stat<F: Fn(Scalar) -> Option<(Scalar, bool)>>(
    statistics: StatsSetRef<'_>,
    stat: Stat,
    truncation: F,
) {
    if let Some(sv) = statistics.get(stat).into_inner() {
        if let Some((truncated_value, truncated)) = truncation(sv) {
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
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        _eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
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

        // Scan for constant array buffers before serialization (while data is still on host).
        let host_buffers = extract_constant_buffers(&chunk);

        let buffers = chunk.serialize(
            ctx.array_ctx(),
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

        Ok(LayoutParts::new(
            CudaFlat,
            stream.dtype().clone(),
            row_count,
            vec![segment_id],
            layout_children(Vec::new()),
            CudaFlatData {
                segment_id,
                ctx: ReadContext::new(ctx.array_ctx().to_ids()),
                array_tree,
                host_buffers: Arc::new(host_buffer_map),
            },
        )
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

/// Build a CUDA-flat writer using only CUDA-compatible, session-enabled array encodings.
///
/// Requires [`register_cuda_layout`]. Zero `block_rows` uses default sizing and dictionary policy;
/// nonzero sets row blocks without outer dictionaries or byte coalescing, retaining per-block
/// dictionary compression.
pub fn cuda_write_strategy(session: &VortexSession, block_rows: usize) -> Arc<dyn LayoutStrategy> {
    let allowed_encodings = session
        .enabled_component_ids(ComponentKind::Array)
        .into_iter()
        .collect();
    let builder = BtrBlocksCompressorBuilder::default()
        .only_cuda_compatible()
        .retain_allowed_encodings(&allowed_encodings);
    let strategy = WriteStrategyBuilder::default()
        .with_flat_strategy(Arc::new(CudaFlatLayoutStrategy::default()));
    if block_rows == 0 {
        strategy.with_btrblocks_builder(builder).build()
    } else {
        // An opaque compressor keeps IntDict; disabling the probe avoids u16-sized outer blocks.
        strategy
            .with_compressor(builder.build())
            .with_probe_compressor(BtrBlocksCompressorBuilder::empty().build())
            .with_row_block_size(block_rows)
            .with_data_block_target_bytes(None)
            .build()
    }
}

#[derive(Clone, Debug)]
struct CudaLayoutRegistration(Arc<Once>);

impl Default for CudaLayoutRegistration {
    fn default() -> Self {
        Self(Arc::new(Once::new()))
    }
}

impl SessionVar for CudaLayoutRegistration {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

const CUDA_EDITION_FAMILY: EditionFamily = EditionFamily {
    name: "cuda",
    origin: "vortex-cuda",
    doc: "CUDA-readable layouts, enabled only when CUDA layout support is registered.",
};
const CUDA_EDITION: EditionId = EditionId::new("cuda", 2026, 9, 0);
static CUDA_EDITION_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CUDA_EDITION,
        min_library_version: None,
    },
    added: &[EditionMember::layout(&"vortex.cuda_flat")],
};

/// Register [`CudaFlat`] and its draft `cuda` edition once per session.
///
/// Enables a newly registered edition only if no `cuda` edition is selected; otherwise preserves
/// writer policy, including on repeated calls. The draft has no cross-version compatibility
/// guarantee. Readers must also register the layout.
///
/// Call alongside [`crate::initialize_cuda`]; registration itself needs no GPU.
pub fn register_cuda_layout(session: &VortexSession) {
    // Editions are published before their members; concurrent callers must wait for both.
    session.get::<CudaLayoutRegistration>().0.call_once(|| {
        session
            .layouts()
            .register(LayoutEncodingRef::new_ref(&CudaFlat));
        if session.editions().find(&CUDA_EDITION).is_some() {
            return;
        }
        if session.editions().find_family("cuda").is_none() {
            session
                .editions()
                .declare_family(&CUDA_EDITION_FAMILY)
                .vortex_expect("CUDA edition family is valid");
        }
        session
            .register_edition(&CUDA_EDITION_DECLARATION)
            .vortex_expect("CUDA edition declaration is valid");
        if !session
            .enabled_editions()
            .editions()
            .iter()
            .any(|edition| edition.family == CUDA_EDITION.family)
        {
            session
                .enable_edition(CUDA_EDITION)
                .vortex_expect("CUDA edition is registered");
        }
    });
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::Dict;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::struct_::StructArrayExt;
    use vortex::array::assert_arrays_eq;
    use vortex::buffer::ByteBufferMut;
    use vortex::editions::CORE_2025_05_0;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::VortexFile;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::io::runtime::current::CurrentThreadRuntime;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::layout::scan::split_by::SplitBy;

    use super::*;

    fn repeated_ids(unique: i64, rows: usize) -> VortexResult<ArrayRef> {
        // Wide, shuffled values favor dictionaries over bitpacking and FoR.
        let ids = PrimitiveArray::from_iter(
            (0..unique)
                .cycle()
                .take(rows)
                .map(|id| (id * 7_919 % unique).wrapping_mul(0x5851_f42d_4c95_7f2d)),
        );
        Ok(StructArray::from_fields(&[("ids", ids.into_array())])?.into_array())
    }

    async fn write_file(
        session: &VortexSession,
        array: ArrayRef,
        block_rows: usize,
    ) -> VortexResult<VortexFile> {
        let mut buffer = ByteBufferMut::empty();
        session
            .write_options()
            .with_strategy(cuda_write_strategy(session, block_rows))
            .write(&mut buffer, array.to_array_stream())
            .await?;
        session.open_options().open_buffer(buffer.freeze())
    }

    fn data_block_rows(layout: &LayoutRef) -> VortexResult<Vec<u64>> {
        let mut rows = Vec::new();
        if layout.is::<CudaFlat>() {
            rows.push(layout.row_count());
        }
        for (kind, child) in layout.child_types().zip(layout.children()?) {
            // Exclude zone maps and dictionary values from data row counts.
            if !matches!(kind, LayoutChildType::Auxiliary(_)) {
                rows.extend(data_block_rows(&child)?);
            }
        }
        Ok(rows)
    }

    #[test]
    fn test_cuda_write_strategy_preserves_integer_dictionary_compression() -> VortexResult<()> {
        let block_rows = 1024;
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());
        register_cuda_layout(&session);
        runtime.block_on(async {
            let input = repeated_ids(8, 2 * block_rows + 137)?;
            let file = write_file(&session, input.clone(), block_rows).await?;

            let batches: Vec<_> = file
                .scan()?
                .with_split_by(SplitBy::Layout)
                .into_array_stream()?
                .try_collect()
                .await?;
            assert_eq!(
                batches.iter().map(|batch| batch.len()).collect::<Vec<_>>(),
                [block_rows, block_rows, 137]
            );
            let mut ctx = session.create_execution_ctx();
            let mut offset = 0;
            for batch in batches {
                // Keep the child encoded to detect loss of IntDict compression.
                let batch = batch.execute::<StructArray>(&mut ctx)?;
                assert!(batch.unmasked_field(0).is::<Dict>());
                let end = offset + batch.len();
                assert_arrays_eq!(batch.into_array(), input.slice(offset..end)?, &mut ctx);
                offset = end;
            }

            Ok(())
        })
    }

    #[test]
    fn test_cuda_write_strategy_preserves_high_cardinality_row_blocks() -> VortexResult<()> {
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());
        register_cuda_layout(&session);
        runtime.block_on(async {
            // Exceed u16 cardinality while remaining eligible for outer dictionaries.
            let block_rows = 70_000 * 8;
            let input = repeated_ids(70_000, block_rows)?;
            let file = write_file(&session, input, block_rows).await?;
            assert_eq!(
                data_block_rows(file.footer().layout())?,
                [block_rows as u64]
            );
            Ok(())
        })
    }

    #[test]
    fn test_concurrent_cuda_registration_preserves_edition_policy() -> VortexResult<()> {
        let session = VortexSession::default();
        session.enable_edition(CORE_2025_05_0)?;
        let mut expected_editions = session.enabled_editions().editions();
        expected_editions.push(CUDA_EDITION);
        expected_editions.sort_unstable();
        let expected_arrays = session.enabled_component_ids(ComponentKind::Array);
        let barrier = std::sync::Barrier::new(4);

        std::thread::scope(|scope| {
            for _ in 0..4 {
                let session = session.clone();
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    register_cuda_layout(&session);
                    assert!(
                        session
                            .enabled_component_ids(ComponentKind::Layout)
                            .contains(&CudaFlat.id())
                    );
                });
            }
        });

        let mut enabled_editions = session.enabled_editions().editions();
        enabled_editions.sort_unstable();
        assert_eq!(enabled_editions, expected_editions);
        assert_eq!(
            session.enabled_component_ids(ComponentKind::Array),
            expected_arrays
        );
        Ok(())
    }

    #[rstest]
    fn test_cuda_registration_preserves_selected_cuda_edition(
        #[values(false, true)] register_first: bool,
    ) -> VortexResult<()> {
        const OTHER_CUDA_EDITION: EditionId = EditionId::new("cuda", 2026, 8, 0);
        let session = VortexSession::default();
        if register_first {
            register_cuda_layout(&session);
        } else {
            session.editions().declare_family(&CUDA_EDITION_FAMILY)?;
        }
        session.register_edition(&EditionDeclaration {
            edition: Edition {
                id: OTHER_CUDA_EDITION,
                min_library_version: None,
            },
            added: &[],
        })?;
        session.enable_edition(OTHER_CUDA_EDITION)?;
        let mut expected_editions = session.enabled_editions().editions();
        expected_editions.sort_unstable();
        let expected_layouts = session.enabled_component_ids(ComponentKind::Layout);

        register_cuda_layout(&session);

        let mut enabled_editions = session.enabled_editions().editions();
        enabled_editions.sort_unstable();
        assert_eq!(enabled_editions, expected_editions);
        assert_eq!(
            session.enabled_component_ids(ComponentKind::Layout),
            expected_layouts
        );
        Ok(())
    }

    #[test]
    fn test_cuda_registration_preserves_disabled_pre_registered_edition() -> VortexResult<()> {
        let session = VortexSession::default();
        session.editions().declare_family(&CUDA_EDITION_FAMILY)?;
        session.register_edition(&CUDA_EDITION_DECLARATION)?;
        let mut expected_editions = session.enabled_editions().editions();
        expected_editions.sort_unstable();
        let expected_layouts = session.enabled_component_ids(ComponentKind::Layout);

        register_cuda_layout(&session);

        let mut enabled_editions = session.enabled_editions().editions();
        enabled_editions.sort_unstable();
        assert_eq!(enabled_editions, expected_editions);
        assert_eq!(
            session.enabled_component_ids(ComponentKind::Layout),
            expected_layouts
        );
        Ok(())
    }
}
