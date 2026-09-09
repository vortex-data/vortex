// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A CUDA-optimized flat layout that inlines small constant array buffers into layout metadata.

use std::any::Any;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use parking_lot::Mutex;
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
use vortex::dtype::DType;
use vortex::dtype::FieldMask;
use vortex::editions::Edition;
use vortex::editions::EditionDeclaration;
use vortex::editions::EditionFamily;
use vortex::editions::EditionId;
use vortex::editions::EditionMember;
use vortex::editions::EditionSessionExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;
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
                let array = array.apply_bound(&expr)?;
                let array = array.filter(mask.clone())?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.null_as_false().execute(&mut ctx)?;
                mask.intersect_by_rank(&array_mask)
            } else {
                let array = array.apply_bound(&expr)?;
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

            array = array.apply_bound(&expr)?;

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

const CUDA_EDITION: EditionId = EditionId::new("cuda", 2026, 9, 0);
static CUDA_EDITION_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CUDA_EDITION,
        min_library_version: None,
    },
    added: &[EditionMember::layout(&"vortex.cuda_flat")],
};

/// Register the [`CudaFlatLayoutEncoding`] and enable its draft `cuda` edition for writing.
///
/// This opts into only the CUDA-flat layout; the session's other edition selections and checks
/// remain unchanged. The draft has no cross-version compatibility guarantee, and readers must
/// register the CUDA layout to read these files.
///
/// Call this alongside [`crate::initialize_cuda`] when setting up a CUDA-enabled session.
/// Registration itself does not require a GPU.
pub fn register_cuda_layout(session: &VortexSession) {
    session
        .layouts()
        .register(LayoutEncodingRef::new_ref(&CudaFlat));

    // CUDA FFI entry points register repeatedly, including through concurrent session clones.
    // Edition declaration rejects duplicates, so serialize the check and registration.
    static REGISTRATION_LOCK: Mutex<()> = Mutex::new(());
    let _guard = REGISTRATION_LOCK.lock();
    if session.editions().find(&CUDA_EDITION).is_none() {
        session
            .editions()
            .declare_family(&EditionFamily {
                name: "cuda",
                origin: "vortex-cuda",
                doc: "CUDA-readable layouts, enabled only when CUDA layout support is registered.",
            })
            .map_err(|error| vortex_err!("{error}"))
            .vortex_expect("CUDA edition family is valid");
        session
            .register_edition(&CUDA_EDITION_DECLARATION)
            .map_err(|error| vortex_err!("{error}"))
            .vortex_expect("CUDA edition declaration is valid");
    }
    session
        .enable_edition(CUDA_EDITION)
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("CUDA edition is registered");
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::stream::ArrayStreamExt;
    use vortex::buffer::ByteBufferMut;
    use vortex::buffer::buffer;
    use vortex::compressor::BtrBlocksCompressorBuilder;
    use vortex::editions::CORE_2025_05_0;
    use vortex::editions::ComponentKind;
    use vortex::editions::DEFAULT_CORE_EDITION;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::file::WriteStrategyBuilder;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::io::runtime::current::CurrentThreadRuntime;
    use vortex::io::session::RuntimeSessionExt;

    use super::*;

    #[test]
    fn test_cuda_compatible_write_on_host() -> VortexResult<()> {
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());
        runtime.block_on(async {
            register_cuda_layout(&session);
            let array =
                StructArray::from_fields(&[("numbers", buffer![1i32, 4, 9, 16].into_array())])?
                    .into_array();
            let strategy = WriteStrategyBuilder::default()
                .with_btrblocks_builder(
                    BtrBlocksCompressorBuilder::default().only_cuda_compatible(),
                )
                .with_flat_strategy(Arc::new(CudaFlatLayoutStrategy::default()))
                .build();

            let mut buffer = ByteBufferMut::empty();
            session
                .write_options()
                .with_strategy(strategy)
                .write(&mut buffer, array.to_array_stream())
                .await?;

            let file = session.open_options().open_buffer(buffer)?;
            let mut layouts = vec![file.footer().layout().to_layout()];
            let mut has_cuda_flat = false;
            while let Some(layout) = layouts.pop() {
                has_cuda_flat |= layout.encoding_id() == CudaFlat.id();
                layouts.extend(layout.children()?);
            }
            assert!(has_cuda_flat);
            let result = file.scan()?.into_array_stream()?.read_all().await?;
            assert_arrays_eq!(array, result, &mut session.create_execution_ctx());
            Ok(())
        })
    }

    #[rstest]
    fn test_cuda_registration_preserves_edition_policy(
        #[values(DEFAULT_CORE_EDITION, CORE_2025_05_0)] core: EditionId,
    ) -> VortexResult<()> {
        let session = VortexSession::default();
        session
            .enable_edition(core)
            .map_err(|error| vortex_err!("{error}"))?;
        let kinds = [
            ComponentKind::Array,
            ComponentKind::Layout,
            ComponentKind::DType,
            ComponentKind::Aggregate,
        ];
        let expected_ids = kinds.map(|kind| {
            let mut ids = session.enabled_component_ids(kind);
            if kind == ComponentKind::Layout {
                assert!(!ids.contains(&CudaFlat.id()));
                ids.push(CudaFlat.id());
                ids.sort_unstable();
            }
            ids
        });
        let mut expected_editions = session.enabled_editions().editions();
        expected_editions.push(CUDA_EDITION);
        expected_editions.sort_unstable();

        std::thread::scope(|scope| {
            for _ in 0..4 {
                let session = session.clone();
                scope.spawn(move || register_cuda_layout(&session));
            }
        });
        register_cuda_layout(&session);

        for (kind, expected) in kinds.into_iter().zip(expected_ids) {
            assert_eq!(session.enabled_component_ids(kind), expected);
        }
        let mut enabled_editions = session.enabled_editions().editions();
        enabled_editions.sort_unstable();
        assert_eq!(enabled_editions, expected_editions);
        session
            .editions()
            .validate()
            .map_err(|error| vortex_err!("{error}"))?;
        assert!(
            !VortexSession::default()
                .enabled_component_ids(ComponentKind::Layout)
                .contains(&CudaFlat.id())
        );
        Ok(())
    }

    #[test]
    fn test_registry_alone_does_not_permit_cuda_flat() -> VortexResult<()> {
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());
        session
            .layouts()
            .register(LayoutEncodingRef::new_ref(&CudaFlat));
        runtime.block_on(async {
            let mut buffer = ByteBufferMut::empty();
            let error = session
                .write_options()
                .with_strategy(Arc::new(CudaFlatLayoutStrategy::default()))
                .write(
                    &mut buffer,
                    buffer![1i32, 4, 9, 16].into_array().to_array_stream(),
                )
                .await
                .err()
                .ok_or_else(|| vortex_err!("write permitted an uneditioned CUDA layout"))?;
            assert!(
                error
                    .to_string()
                    .contains("Layout encoding vortex.cuda_flat not permitted by ctx"),
                "unexpected error: {error}"
            );
            Ok(())
        })
    }
}
