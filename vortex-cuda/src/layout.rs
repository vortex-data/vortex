// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A CUDA-optimized flat layout that inlines small constant array buffers into layout metadata.
//!
//! Bounded, complete logical value domains allow host pruning before payload registration.
//! Missing or unsupported domains never exclude rows. `VORTEX_CUDA_DOMAIN_PRUNING=0` disables
//! writing and reading this optional metadata.

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
use prost::Message;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::MaskFuture;
use vortex::array::ProstMetadata;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Constant;
use vortex::array::arrays::Dict;
use vortex::array::arrays::Extension;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBin;
use vortex::array::arrays::VarBinView;
use vortex::array::arrays::dict::DictArraySlotsExt;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::builders::builder_with_capacity_in;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::expr::BoundExpression;
use vortex::array::expr::stats::Precision;
use vortex::array::expr::stats::Stat;
use vortex::array::expr::stats::StatsProvider;
use vortex::array::extension::datetime::AnyTemporal;
use vortex::array::memory::BufferAllocatorRef;
use vortex::array::scalar_fn::fns::between::Between;
use vortex::array::scalar_fn::fns::binary::Binary;
use vortex::array::scalar_fn::fns::literal::Literal;
use vortex::array::scalar_fn::fns::operators::Operator;
use vortex::array::serde::SerializeOptions;
use vortex::array::serde::SerializedArray;
use vortex::array::stats::StatsSetRef;
use vortex::buffer::BufferString;
use vortex::buffer::ByteBuffer;
use vortex::compressor::BtrBlocksCompressorBuilder;
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
use vortex::error::vortex_panic;
use vortex::file::WriteStrategyBuilder;
use vortex::layout::Layout;
use vortex::layout::LayoutChildType;
use vortex::layout::LayoutDeserializeArgs;
use vortex::layout::LayoutEncodingRef;
use vortex::layout::LayoutId;
use vortex::layout::LayoutParts;
use vortex::layout::LayoutReader;
use vortex::layout::LayoutReaderContext;
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
use vortex::scalar::ScalarValue;
use vortex::scalar::lower_bound;
use vortex::scalar::upper_bound;
use vortex::session::SessionExt;
use vortex::session::SessionVar;
use vortex::session::VortexSession;
use vortex::session::registry::CachedId;
use vortex::session::registry::ReadContext;
use vortex::utils::aliases::hash_map::HashMap;

use crate::CudaSession;
use crate::DictionaryExport;
use crate::executor::CudaArrayExt;

pub(crate) static CUDA_SCAN_DICTIONARY_EXPORT: CachedId =
    CachedId::new("vortex.cuda.scan_dictionary_export");

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
    /// Optional complete logical domain; absent in files written without domain pruning.
    #[prost(message, optional, tag = "3")]
    pub logical_value_domain: Option<LogicalValueDomain>,
}

/// A bounded superset of a leaf's logical values, never a sample or a truncated dictionary.
///
/// Values use the leaf dtype and the scalar-value protobuf format. Nullable leaves include null
/// even when only dictionary codes, rather than dictionary values, might be null.
#[derive(Clone, prost::Message)]
pub struct LogicalValueDomain {
    /// Unsupported versions are ignored by the reader.
    #[prost(uint32, tag = "1")]
    pub version: u32,
    /// Complete scalar values, including any conservative extra values.
    #[prost(bytes = "vec", repeated, tag = "2")]
    pub values: Vec<Vec<u8>>,
}

const DOMAIN_VERSION: u32 = 1;
const MAX_DOMAIN_VALUES: usize = 256;
const MAX_DOMAIN_BYTES: usize = 16 * 1024;
const MAX_DOMAIN_NODES: usize = 32;
const MAX_DOMAIN_EXPR_NODES: usize = 64;

fn domain_pruning_enabled() -> bool {
    !std::env::var("VORTEX_CUDA_DOMAIN_PRUNING").is_ok_and(|value| value == "0")
}

fn domain_dtype_supported(dtype: &DType) -> bool {
    match dtype {
        DType::Bool(_) | DType::Utf8(_) | DType::Binary(_) => true,
        // Float comparison semantics (NaNs and signed zero) need a separate compatibility audit.
        DType::Primitive(ptype, _) => ptype.is_int(),
        DType::Extension(ext) => ext.is::<AnyTemporal>(),
        _ => false,
    }
}

fn domain_predicate_supported(expr: &BoundExpression, remaining: &mut usize) -> bool {
    if *remaining == 0 {
        return false;
    }
    *remaining -= 1;
    // Expression optimization combines a same-field range conjunction into Between.
    if expr.is::<Between>() {
        return expr.child(0).is_root()
            && expr.child(1).is::<Literal>()
            && expr.child(2).is::<Literal>();
    }
    match expr.as_opt::<Binary>() {
        Some(Operator::And) => {
            domain_predicate_supported(expr.child(0), remaining)
                && domain_predicate_supported(expr.child(1), remaining)
        }
        Some(
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::Lte
            | Operator::Gt
            | Operator::Gte,
        ) => {
            (expr.child(0).is_root() && expr.child(1).is::<Literal>())
                || (expr.child(1).is_root() && expr.child(0).is::<Literal>())
        }
        _ => false,
    }
}

impl LogicalValueDomain {
    fn within_budget(&self) -> bool {
        self.version == DOMAIN_VERSION
            && !self.values.is_empty()
            && self.values.len() <= MAX_DOMAIN_VALUES
            && self.encoded_len() <= MAX_DOMAIN_BYTES
    }

    fn rejects(
        &self,
        dtype: &DType,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<bool> {
        let mut remaining = MAX_DOMAIN_EXPR_NODES;
        if !self.within_budget()
            || !domain_dtype_supported(dtype)
            || !domain_predicate_supported(expr, &mut remaining)
            || !expr.is_root_bound_to(dtype)
        {
            return Ok(false);
        }
        let mut builder =
            builder_with_capacity_in(dtype, self.values.len(), BufferAllocatorRef::static_ref());
        for bytes in &self.values {
            let value = ScalarValue::from_proto_bytes(bytes, dtype, session)?;
            builder.append_scalar(&Scalar::try_new(dtype.clone(), value)?)?;
        }
        // Evaluate the whole predicate on each value, not its conjuncts independently. Separate
        // bounds can each match a dictionary entry while their intersection matches none.
        let mut ctx = session.create_execution_ctx();
        Ok(builder
            .finish()
            .apply_bound(expr)?
            .fill_null(false)?
            .execute::<Mask>(&mut ctx)?
            .all_false())
    }
}

fn logical_value_domain(
    array: &ArrayRef,
    enabled: bool,
    session: &VortexSession,
) -> Option<LogicalValueDomain> {
    if !enabled || !domain_dtype_supported(array.dtype()) {
        return None;
    }
    // Optional metadata must not turn an otherwise readable/writable encoding into an error.
    try_logical_value_domain(array, session).ok().flatten()
}

fn try_logical_value_domain(
    array: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<Option<LogicalValueDomain>> {
    let storage = array
        .as_opt::<Extension>()
        .map_or_else(|| array.clone(), |ext| ext.storage_array().clone());
    let candidates = if storage.is::<Constant>() {
        storage
    } else if let Some(dict) = storage.as_opt::<Dict>() {
        dict.values().clone()
    } else {
        // In particular, a dictionary nested under ALP/FoR/patches is not the root value domain.
        return Ok(None);
    };
    let candidates = if let Some(ext) = candidates.as_opt::<Extension>() {
        ext.storage_array().clone()
    } else {
        candidates
    };
    let mut ctx = session.create_execution_ctx();
    let scalars = if let Some(constant) = candidates.as_opt::<Constant>() {
        vec![constant.scalar().clone()]
    } else {
        if candidates.len() > MAX_DOMAIN_VALUES {
            return Ok(None);
        }
        // Fixed-width values may be compressed, but never execute an unbounded child tree.
        // Compressed strings can expand arbitrarily; only canonical string buffers are admitted.
        if matches!(candidates.dtype(), DType::Utf8(_) | DType::Binary(_))
            && !candidates.is::<VarBin>()
            && !candidates.is::<VarBinView>()
        {
            return Ok(None);
        }
        let mut bytes = 0usize;
        for (index, node) in candidates.depth_first_traversal().enumerate() {
            if index >= MAX_DOMAIN_NODES || node.len() > MAX_DOMAIN_VALUES || !node.is_host() {
                return Ok(None);
            }
            for buffer in node.buffers() {
                if buffer.len() > MAX_DOMAIN_BYTES - bytes {
                    return Ok(None);
                }
                bytes += buffer.len();
            }
        }
        let values = candidates.execute::<Canonical>(&mut ctx)?.into_array();
        // Canonicalize once before this bounded loop, rather than decoding a dictionary per value.
        (0..values.len())
            .map(|index| values.execute_scalar(index, &mut ctx))
            .collect::<VortexResult<Vec<_>>>()?
    };
    let mut domain = LogicalValueDomain {
        version: DOMAIN_VERSION,
        values: Vec::with_capacity(scalars.len() + usize::from(array.dtype().is_nullable())),
    };
    let mut bytes = 0usize;
    for scalar in scalars {
        // Reserve protobuf overhead before serializing variable-width values, so even constants
        // with very large strings cannot allocate an oversized domain only to discard it.
        let size = scalar.approx_nbytes().saturating_add(16);
        if size > MAX_DOMAIN_BYTES - bytes {
            return Ok(None);
        }
        bytes += size;
        domain
            .values
            .push(ScalarValue::to_proto_bytes(scalar.value()));
    }
    if array.dtype().is_nullable() {
        domain.values.push(ScalarValue::to_proto_bytes(None));
    }
    Ok(domain.within_budget().then_some(domain))
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
    logical_value_domain: Option<LogicalValueDomain>,
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
            logical_value_domain: layout.logical_value_domain.clone(),
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
            logical_value_domain: metadata
                .logical_value_domain
                .as_ref()
                .filter(|domain| domain.within_budget())
                .cloned(),
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
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        // Only a decoded CUDA scan may canonicalize dictionaries before Arrow export.
        // Ordinary readers keep their encodings and never initialize CUDA here.
        let eager_decode = ctx
            .get::<DictionaryExport>(*CUDA_SCAN_DICTIONARY_EXPORT)
            .is_some_and(|policy| *policy == DictionaryExport::Decode)
            && session.get_opt::<CudaSession>().is_some();
        Ok(Arc::new(CudaFlatReader {
            layout: layout.clone(),
            name,
            segment_source,
            session: session.clone(),
            array: Default::default(),
            eager_decode,
            domain_pruning: domain_pruning_enabled(),
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
    eager_decode: bool,
    domain_pruning: bool,
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
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // CUDA scan preflight invokes pruning before constructing projection evaluations. Do not
        // call array_future here: even registering that future makes the payload coalescible.
        let rejected = self.domain_pruning
            && self
                .layout
                .logical_value_domain
                .as_ref()
                .is_some_and(|domain| {
                    domain
                        .rejects(self.dtype(), expr, &self.session)
                        .unwrap_or(false)
                });
        Ok(MaskFuture::ready(if rejected {
            Mask::new_false(mask.len())
        } else {
            mask
        }))
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
                let array_mask = array.fill_null(false)?.execute::<Mask>(&mut ctx)?;
                mask.intersect_by_rank(&array_mask)
            } else {
                let array = array.apply_bound(&expr)?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.fill_null(false)?.execute::<Mask>(&mut ctx)?;
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
        let session = self.session.clone();
        let eager_decode = self.eager_decode;

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

            // Submit numeric field packs before the partitioned layout joins all inputs.
            // Raw primitive projections stay lazy: auxiliary arrays such as list offsets
            // can still be needed for host-side scalar planning.
            // Device buffer events order decoded writes with the later Arrow export.
            if eager_decode {
                array = eagerly_execute_numeric_projection(array, &session).await?;
            }

            Ok(array)
        }
        .boxed())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

async fn eagerly_execute_numeric_projection(
    array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let DType::Struct(fields, _) = array.dtype() else {
        return Ok(array);
    };
    if fields.nfields() == 0
        || !fields
            .fields()
            .all(|dtype| matches!(dtype, DType::Primitive(..)))
    {
        return Ok(array);
    }

    let mut ctx = CudaSession::create_execution_ctx(session)?
        .with_dictionary_export(DictionaryExport::Decode);
    // Resolve structural wrappers such as select/pack without decoding their field values.
    // They have no CUDA kernel, but their fields may already contain device buffers.
    let array = array
        .execute::<StructArray>(ctx.execution_ctx())?
        .into_array();
    Ok(array.execute_cuda(&mut ctx).await?.into_array())
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
        let logical_value_domain = logical_value_domain(&chunk, domain_pruning_enabled(), session);

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
                logical_value_domain,
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
    let builder = BtrBlocksCompressorBuilder::from_session(session).only_cuda_compatible();
    let strategy = WriteStrategyBuilder::from_session(session)
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
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::TryStreamExt;
    use roaring::RoaringTreemap;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::ArrayContext;
    use vortex::array::Canonical;
    use vortex::array::IntoArray;
    use vortex::array::arrays::ConstantArray;
    use vortex::array::arrays::Dict;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::ExtensionArray;
    use vortex::array::arrays::Primitive;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::Struct;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::arrays::struct_::StructArrayExt;
    use vortex::array::assert_arrays_eq;
    use vortex::array::buffer::BufferHandle;
    use vortex::array::expr::and;
    use vortex::array::expr::between;
    use vortex::array::expr::eq;
    use vortex::array::expr::gt_eq;
    use vortex::array::expr::is_null;
    use vortex::array::expr::lit;
    use vortex::array::expr::lt;
    use vortex::array::expr::or;
    use vortex::array::expr::pack;
    use vortex::array::expr::root;
    use vortex::array::extension::datetime::Date;
    use vortex::array::extension::datetime::TimeUnit;
    use vortex::array::scalar_fn::fns::between::BetweenOptions;
    use vortex::array::scalar_fn::fns::between::StrictComparison;
    use vortex::array::validity::Validity;
    use vortex::buffer::ByteBufferMut;
    use vortex::dtype::FieldNames;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::editions::CORE_2025_05_0;
    use vortex::editions::ComponentKind;
    use vortex::encodings::fastlanes::BitPacked;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::VortexFile;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::io::runtime::current::CurrentThreadRuntime;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::layout::scan::scan_builder::ScanBuilder;
    use vortex::layout::scan::split_by::SplitBy;
    use vortex::layout::segments::SegmentFuture;
    use vortex::scan::selection::Selection;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::CudaFileScanExt;
    use crate::scan_cuda_pruned;

    struct DomainRecordingSource {
        source: Arc<dyn SegmentSource>,
        requests: AtomicUsize,
    }

    impl SegmentSource for DomainRecordingSource {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.requests.fetch_add(1, Ordering::Relaxed);
            self.source.request(id)
        }
    }

    struct DomainBufferSource(ByteBuffer);

    impl SegmentSource for DomainBufferSource {
        fn request(&self, _id: SegmentId) -> SegmentFuture {
            let buffer = self.0.clone();
            async move { Ok(BufferHandle::new_host(buffer)) }.boxed()
        }
    }

    fn domain_dictionary() -> VortexResult<ArrayRef> {
        Ok(DictArray::try_new(
            PrimitiveArray::from_iter([1u8, 0, 1, 0]).into_array(),
            PrimitiveArray::from_iter([0i32, 100]).into_array(),
        )?
        .into_array())
    }

    fn domain_gap(dtype: &DType, optimize: bool) -> VortexResult<BoundExpression> {
        let expr = if optimize {
            // find_between optimizes named fields before StructReader rebases them to root.
            between(
                root(),
                lit(40i32),
                lit(60i32),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                },
            )
        } else {
            and(gt_eq(root(), lit(40i32)), lt(root(), lit(60i32)))
        };
        expr.bind(dtype)
    }

    // Explicit per-reader flags avoid mutating the process environment in concurrent tests.
    fn domain_reader(
        array: &ArrayRef,
        domain: Option<LogicalValueDomain>,
        enabled: bool,
        session: &VortexSession,
    ) -> VortexResult<(Arc<CudaFlatReader>, Arc<DomainRecordingSource>)> {
        let array_ctx = ArrayContext::empty();
        let buffers = array.serialize(
            &array_ctx,
            session,
            &SerializeOptions {
                offset: 0,
                include_padding: true,
            },
        )?;
        let metadata = CudaFlatLayoutMetadata {
            array_encoding_tree: buffers[buffers.len() - 2].to_vec(),
            host_buffers: extract_constant_buffers(array),
            logical_value_domain: domain,
        };
        let metadata = CudaFlatLayoutMetadata::decode(metadata.encode_to_vec().as_slice())?;
        let mut buffer = ByteBufferMut::empty();
        for part in buffers {
            buffer.extend_from_slice(&part);
        }
        let source = Arc::new(DomainRecordingSource {
            source: Arc::new(DomainBufferSource(buffer.freeze())),
            requests: AtomicUsize::new(0),
        });
        let segment_id = SegmentId::from(0);
        let layout = LayoutParts::new(
            CudaFlat,
            array.dtype().clone(),
            array.len() as u64,
            vec![segment_id],
            layout_children(Vec::new()),
            CudaFlatData {
                segment_id,
                ctx: ReadContext::new(array_ctx.to_ids()),
                array_tree: ByteBuffer::from(metadata.array_encoding_tree),
                host_buffers: Arc::new(
                    metadata
                        .host_buffers
                        .into_iter()
                        .map(|buffer| (buffer.buffer_index, ByteBuffer::from(buffer.data)))
                        .collect(),
                ),
                logical_value_domain: metadata.logical_value_domain,
            },
        )
        .into_typed();
        let reader = Arc::new(CudaFlatReader {
            layout,
            name: "arbitrary".into(),
            segment_source: Arc::clone(&source) as _,
            session: session.clone(),
            array: Default::default(),
            eager_decode: false,
            domain_pruning: enabled,
        });
        Ok((reader, source))
    }

    #[rstest]
    #[case::conjunction(false)]
    #[case::optimized_between(true)]
    fn test_domain_preflight_registers_no_rejected_payload(
        #[case] optimize: bool,
    ) -> VortexResult<()> {
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());
        runtime.block_on(async {
            let input = domain_dictionary()?;
            let domain = logical_value_domain(&input, true, &session).unwrap();
            for bound in [gt_eq(root(), lit(40i32)), lt(root(), lit(60i32))] {
                assert!(!domain.rejects(input.dtype(), &bound.bind(input.dtype())?, &session)?);
            }
            let (reader, source) = domain_reader(&input, Some(domain), true, &session)?;
            let expr = domain_gap(input.dtype(), optimize)?;
            if optimize {
                assert!(expr.is::<Between>());
            }
            let mask = reader
                .pruning_evaluation(&(0..input.len() as u64), &expr, Mask::new_true(input.len()))?
                .await?;
            assert!(mask.all_false());
            assert_eq!(source.requests.load(Ordering::Relaxed), 0);

            // Match scan_cuda_pruned: reject during preflight, before ScanBuilder creates any
            // projection future. Ordinary with_filter scans can register projections earlier.
            let mut excluded = RoaringTreemap::new();
            excluded.insert_range(0..mask.len() as u64);
            let batches: Vec<_> = ScanBuilder::new(session.clone(), reader)
                .with_selection(Selection::ExcludeRoaring(excluded))
                .into_array_stream()?
                .try_collect()
                .await?;
            assert!(batches.is_empty());
            assert_eq!(source.requests.load(Ordering::Relaxed), 0);
            Ok(())
        })
    }

    #[test]
    fn test_domain_cuda_scan_environment_gate() -> VortexResult<()> {
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());
        register_cuda_layout(&session);
        runtime.block_on(async {
            let input = domain_dictionary()?;
            let mut bytes = ByteBufferMut::empty();
            session
                .write_options()
                .with_strategy(Arc::new(CudaFlatLayoutStrategy::default()))
                .write(&mut bytes, input.to_array_stream())
                .await?;
            let file = session.open_options().open_buffer(bytes.freeze())?;
            let source = Arc::new(DomainRecordingSource {
                source: file.segment_source(),
                requests: AtomicUsize::new(0),
            });
            let file = file.with_segment_source(Arc::clone(&source) as _);
            let predicate = and(gt_eq(root(), lit(40i32)), lt(root(), lit(60i32)));
            let batches: Vec<_> = scan_cuda_pruned(&file, DictionaryExport::Decode, &predicate)
                .await?
                .into_array_stream()?
                .try_collect()
                .await?;
            let enabled = domain_pruning_enabled();
            assert_eq!(batches.is_empty(), enabled);
            assert_eq!(
                source.requests.load(Ordering::Relaxed),
                usize::from(!enabled)
            );
            assert!(session.get_opt::<CudaSession>().is_none());
            Ok(())
        })
    }

    #[rstest]
    #[case::missing(0)]
    #[case::disabled(1)]
    #[case::too_many_values(2)]
    #[case::too_many_bytes(3)]
    #[case::unknown_version(4)]
    #[case::invalid_scalar(5)]
    #[case::empty_domain(6)]
    fn test_domain_metadata_fallback(#[case] case: usize) -> VortexResult<()> {
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());
        runtime.block_on(async {
            let input = domain_dictionary()?;
            let mut domain = logical_value_domain(&input, true, &session);
            match case {
                0 => domain = None,
                1 => {}
                2 => domain.as_mut().unwrap().values = vec![vec![0]; MAX_DOMAIN_VALUES + 1],
                3 => domain.as_mut().unwrap().values = vec![vec![0; MAX_DOMAIN_BYTES + 1]],
                4 => domain.as_mut().unwrap().version += 1,
                5 => domain.as_mut().unwrap().values = vec![vec![0xff]],
                6 => domain.as_mut().unwrap().values.clear(),
                _ => unreachable!(),
            }
            let (reader, source) = domain_reader(&input, domain, case != 1, &session)?;
            let mask = reader
                .pruning_evaluation(
                    &(0..input.len() as u64),
                    &domain_gap(input.dtype(), true)?,
                    Mask::new_true(input.len()),
                )?
                .await?;
            assert!(mask.all_true());
            assert_eq!(source.requests.load(Ordering::Relaxed), 0);
            let batches: Vec<_> = ScanBuilder::new(session.clone(), reader)
                .into_array_stream()?
                .try_collect()
                .await?;
            assert_eq!(batches.len(), 1);
            assert_arrays_eq!(batches[0], input, &mut session.create_execution_ctx());
            assert_eq!(source.requests.load(Ordering::Relaxed), 1);
            Ok(())
        })
    }

    #[tokio::test]
    async fn test_domain_keeps_existing_mask_without_requests() -> VortexResult<()> {
        let session = VortexSession::default();
        let input = domain_dictionary()?;
        let domain = logical_value_domain(&input, true, &session);
        let (reader, source) = domain_reader(&input, domain, true, &session)?;
        let mask = Mask::from_iter([false, true]);
        let result = reader
            .pruning_evaluation(
                &(1..3),
                &eq(root(), lit(0i32)).bind(input.dtype())?,
                mask.clone(),
            )?
            .await?;
        assert_eq!(result, mask);
        assert_eq!(source.requests.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn test_domain_metadata_wire_compatibility() -> VortexResult<()> {
        #[derive(prost::Message)]
        struct LegacyMetadata {
            #[prost(bytes, tag = "1")]
            array_encoding_tree: Vec<u8>,
            #[prost(message, repeated, tag = "2")]
            host_buffers: Vec<InlinedBuffer>,
        }

        let old = LegacyMetadata {
            array_encoding_tree: vec![1, 2, 3],
            host_buffers: Vec::new(),
        };
        let mut new = CudaFlatLayoutMetadata::decode(old.encode_to_vec().as_slice())?;
        assert!(new.logical_value_domain.is_none());
        new.logical_value_domain =
            logical_value_domain(&domain_dictionary()?, true, &VortexSession::default());
        assert!(new.logical_value_domain.is_some());
        let decoded = LegacyMetadata::decode(new.encode_to_vec().as_slice())?;
        assert_eq!(decoded.array_encoding_tree, old.array_encoding_tree);
        assert!(decoded.host_buffers.is_empty());
        Ok(())
    }

    #[test]
    fn test_domain_writer_bounds_and_unsupported_encodings() -> VortexResult<()> {
        let session = VortexSession::default();
        let input = domain_dictionary()?;
        assert!(logical_value_domain(&input, false, &session).is_none());
        let large = DictArray::try_new(
            PrimitiveArray::from_iter([0u16]).into_array(),
            PrimitiveArray::from_iter(0..=i32::try_from(MAX_DOMAIN_VALUES)?).into_array(),
        )?
        .into_array();
        assert!(logical_value_domain(&large, true, &session).is_none());
        let large = ConstantArray::new("x".repeat(MAX_DOMAIN_BYTES), 100).into_array();
        assert!(logical_value_domain(&large, true, &session).is_none());
        let primitive = PrimitiveArray::from_iter([0u32, 100]).into_array();
        assert!(logical_value_domain(&primitive, true, &session).is_none());
        let packed =
            BitPacked::encode(&primitive, 7, &mut session.create_execution_ctx())?.into_array();
        assert!(logical_value_domain(&packed, true, &session).is_none());
        let float = ConstantArray::new(f64::NAN, 10).into_array();
        assert!(logical_value_domain(&float, true, &session).is_none());
        Ok(())
    }

    #[rstest]
    #[case::null_codes(true, false)]
    #[case::null_values(false, true)]
    #[case::both(true, true)]
    fn test_domain_nulls_and_unsupported_predicates(
        #[case] null_codes: bool,
        #[case] null_values: bool,
    ) -> VortexResult<()> {
        let session = VortexSession::default();
        let codes =
            PrimitiveArray::from_option_iter([Some(0u8), (!null_codes).then_some(1)]).into_array();
        let values = PrimitiveArray::from_option_iter([Some(0i32), (!null_values).then_some(100)])
            .into_array();
        let input = DictArray::try_new(codes, values)?.into_array();
        let domain = logical_value_domain(&input, true, &session).unwrap();
        assert!(domain.values.iter().any(|bytes| {
            ScalarValue::from_proto_bytes(bytes, input.dtype(), &session)
                .is_ok_and(|value| value.is_none())
        }));
        assert!(domain.rejects(input.dtype(), &domain_gap(input.dtype(), true)?, &session)?);
        assert!(!domain.rejects(
            input.dtype(),
            &is_null(root()).bind(input.dtype())?,
            &session,
        )?);
        let unsupported = or(eq(root(), lit(40i32)), eq(root(), lit(60i32)));
        assert!(!domain.rejects(input.dtype(), &unsupported.bind(input.dtype())?, &session)?);
        let null_literal = eq(root(), lit(Scalar::null(input.dtype().clone())));
        assert!(domain.rejects(input.dtype(), &null_literal.bind(input.dtype())?, &session)?);
        Ok(())
    }

    #[test]
    fn test_domain_constants_and_strings() -> VortexResult<()> {
        let session = VortexSession::default();
        for input in [
            ConstantArray::new(0i32, 10).into_array(),
            ConstantArray::new(
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                10,
            )
            .into_array(),
        ] {
            let domain = logical_value_domain(&input, true, &session).unwrap();
            assert!(domain.rejects(input.dtype(), &domain_gap(input.dtype(), true)?, &session)?);
        }
        let strings = DictArray::try_new(
            PrimitiveArray::from_iter([0u8, 1]).into_array(),
            VarBinViewArray::from_iter_str(["alpha", "omega"]).into_array(),
        )?
        .into_array();
        let domain = logical_value_domain(&strings, true, &session).unwrap();
        assert!(domain.rejects(
            strings.dtype(),
            &eq(root(), lit("absent")).bind(strings.dtype())?,
            &session,
        )?);
        assert!(!domain.rejects(
            strings.dtype(),
            &eq(root(), lit("omega")).bind(strings.dtype())?,
            &session,
        )?);
        Ok(())
    }

    #[test]
    fn test_domain_temporal_wrapper_and_unused_values() -> VortexResult<()> {
        let session = VortexSession::default();
        let dtype = Date::new(TimeUnit::Days, Nullability::NonNullable).erased();
        let input = ExtensionArray::new(dtype.clone(), domain_dictionary()?).into_array();
        let domain = logical_value_domain(&input, true, &session).unwrap();
        let bound = |value: i32| lit(Scalar::extension_ref(dtype.clone(), Scalar::from(value)));
        let expr = and(gt_eq(root(), bound(40)), lt(root(), bound(60)))
            .bind(input.dtype())?
            .optimize()?;
        assert!(domain.rejects(input.dtype(), &expr, &session)?);

        let input = DictArray::try_new(
            PrimitiveArray::from_iter([0u8]).into_array(),
            PrimitiveArray::from_iter([0i32, 50, 100]).into_array(),
        )?
        .into_array();
        let domain = logical_value_domain(&input, true, &session).unwrap();
        // An unused matching value may prevent pruning; inspecting codes to remove it is forbidden.
        assert!(!domain.rejects(input.dtype(), &domain_gap(input.dtype(), true)?, &session)?);
        Ok(())
    }

    #[rstest]
    #[case::struct_pack(0..8)]
    #[case::struct_pack_slice(1..7)]
    #[crate::test]
    async fn test_eager_numeric_projection_decodes_before_export(
        #[case] row_range: Range<usize>,
    ) -> VortexResult<()> {
        let session = vortex::array::array_session().with_some(CudaSession::try_default()?);
        let mut ctx = session.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter([
            Some(7u32),
            Some(1),
            None,
            Some(3),
            Some(15),
            None,
            Some(2),
            Some(8),
        ])
        .into_array();
        let encoded = BitPacked::encode(&values, 4, &mut ctx)?.into_array();
        assert!(encoded.is::<BitPacked>());
        let projection =
            pack([("values", root())], Nullability::NonNullable).bind(encoded.dtype())?;
        let projected = encoded.slice(row_range.clone())?.apply_bound(&projection)?;
        let expected =
            StructArray::from_fields(&[("values", values.slice(row_range)?)])?.into_array();

        let result = eagerly_execute_numeric_projection(projected, &session).await?;
        let child = result.as_::<Struct>().unmasked_field(0).clone();
        assert!(child.is::<Primitive>(), "projected child is still encoded");
        assert!(
            child
                .as_::<Primitive>()
                .buffer_handle()
                .as_device_opt()
                .is_some(),
            "projected child was not decoded onto the device"
        );
        let host = result
            .execute::<Canonical>(&mut ctx)?
            .into_host()
            .await?
            .into_array();
        assert_arrays_eq!(host, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_eager_numeric_projection_preserves_empty_structs(
        #[values(0, 8)] rows: usize,
    ) -> VortexResult<()> {
        let session = vortex::array::array_session();
        let input = StructArray::try_new(FieldNames::empty(), [], rows, Validity::NonNullable)?
            .into_array();
        let result = eagerly_execute_numeric_projection(input.clone(), &session).await?;
        assert!(ArrayRef::ptr_eq(&result, &input));
        assert_eq!(result.len(), rows);
        assert!(session.get_opt::<CudaSession>().is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_eager_numeric_projection_preserves_raw_primitives() -> VortexResult<()> {
        let session = vortex::array::array_session();
        let mut ctx = session.create_execution_ctx();
        let values = PrimitiveArray::from_iter([1u32, 3, 7, 15]).into_array();
        let input = BitPacked::encode(&values, 4, &mut ctx)?.into_array();
        let result = eagerly_execute_numeric_projection(input.clone(), &session).await?;
        assert!(ArrayRef::ptr_eq(&result, &input));
        assert!(session.get_opt::<CudaSession>().is_none());
        Ok(())
    }

    #[rstest]
    #[case::mixed(true)]
    #[case::strings_only(false)]
    #[tokio::test]
    async fn test_eager_numeric_projection_preserves_non_numeric_structs(
        #[case] mixed: bool,
    ) -> VortexResult<()> {
        let session = vortex::array::array_session();
        let mut ctx = session.create_execution_ctx();
        let strings = VarBinViewArray::from_iter_str(["a", "b", "c", "d"]).into_array();
        let mut fields = vec![("text", strings)];
        if mixed {
            let values = PrimitiveArray::from_iter([1u32, 3, 7, 15]).into_array();
            fields.push((
                "number",
                BitPacked::encode(&values, 4, &mut ctx)?.into_array(),
            ));
        }
        let input = StructArray::from_fields(&fields)?.into_array();
        let result = eagerly_execute_numeric_projection(input.clone(), &session).await?;
        assert!(ArrayRef::ptr_eq(&result, &input));
        for (idx, (_, original)) in fields.iter().enumerate() {
            assert!(ArrayRef::ptr_eq(
                result.as_::<Struct>().unmasked_field(idx),
                original
            ));
        }
        assert!(session.get_opt::<CudaSession>().is_none());
        Ok(())
    }

    #[test]
    fn test_decoded_scan_does_not_initialize_cuda() -> VortexResult<()> {
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());
        register_cuda_layout(&session);
        runtime.block_on(async {
            let input = repeated_ids(8, 1024)?;
            let file = write_file(&session, input.clone(), 1024).await?;
            assert!(session.get_opt::<CudaSession>().is_none());
            let batches: Vec<_> = file
                .scan_cuda(DictionaryExport::Decode)?
                .with_split_by(SplitBy::Layout)
                .into_array_stream()?
                .try_collect()
                .await?;
            assert!(session.get_opt::<CudaSession>().is_none());
            assert_eq!(batches.len(), 1);
            let mut ctx = session.create_execution_ctx();
            let batch = batches[0].clone().execute::<StructArray>(&mut ctx)?;
            assert!(batch.unmasked_field(0).is::<Dict>());
            assert_arrays_eq!(batch.into_array(), input, &mut ctx);
            Ok(())
        })
    }

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
