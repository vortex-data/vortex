// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exact row selection for CUDA scans whose filter otherwise only prunes zones.
//!
//! [`scan_cuda_pruned`](crate::scan_cuda_pruned) skips zones a filter rules out, but returns kept
//! zones whole, so a consumer that filters afterwards copies and evaluates every row of a kept
//! zone. [`select_rows`] additionally evaluates the filter's comparisons of a field with a
//! literal on the GPU, and drops the failing rows from each decoded batch before export.
//!
//! The default shared path fuses the eligible conjunction and reuses stable tile-local
//! selection positions across columns in one scatter launch. Batches outside its bounds
//! fall back to per-column CUB selection. `VORTEX_CUDA_EXACT_FILTER_IMPL=cub` forces the
//! baseline for comparison; `shared` explicitly selects the default. Both paths preserve
//! the consumer's final filter and the existing scan/export copy sizes.
//! Single-field conjunctions use typed, ballot-count masks by default.
//! `VORTEX_CUDA_TYPED_FILTER=0` restores the generic mask; all other values retain the default.
//! The CUB override takes precedence over this setting.

use std::env;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use futures::StreamExt;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::Primitive;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::array::expr::Expression;
use vortex::array::match_each_native_simd_ptype;
use vortex::array::scalar_fn::fns::binary::Binary;
use vortex::array::scalar_fn::fns::get_item::GetItem;
use vortex::array::scalar_fn::fns::literal::Literal;
use vortex::array::scalar_fn::fns::operators::Operator;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::SendableArrayStream;
use vortex::array::validity::Validity;
use vortex::buffer::Alignment;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;
use vortex_cub::filter::CubFilterable;
use vortex_cub::filter::cudaStream_t;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::CudaSession;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecutionCtx;
use crate::file::coerce_comparison_literals;

/// Batches keeping more than this fraction of their rows are returned whole: compacting them
/// costs a pass over every column, but saves the consumer little.
const SELECTED_FRACTION_MAX: f64 = 0.5;

// Keep these limits and the descriptor layouts in sync with exact_filter.cu.
const FILTER_THREADS: usize = 256;
const FILTER_TILE_ROWS: usize = 2048;
const FILTER_MAX_PREDICATES: usize = 16;
const FILTER_MAX_COLUMNS: usize = 32;
const FILTER_MAX_TILES: usize = FILTER_THREADS * FILTER_THREADS;

/// Shared selection with typed single-group masks is the default; CUB remains a control/fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectionImpl {
    Cub,
    Shared,
    SharedTyped,
}

impl SelectionImpl {
    fn from_env() -> Self {
        Self::from_settings(
            env::var("VORTEX_CUDA_EXACT_FILTER_IMPL").ok().as_deref(),
            env::var("VORTEX_CUDA_TYPED_FILTER").ok().as_deref(),
        )
    }

    fn from_settings(implementation: Option<&str>, typed_filter: Option<&str>) -> Self {
        match (implementation, typed_filter) {
            (Some("cub"), _) => Self::Cub,
            (_, Some("0")) => Self::Shared,
            _ => Self::SharedTyped,
        }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct FilterComparison {
    input: u64,
    literal: u64,
    ptype: u32,
    operator: u32,
}

#[repr(C)]
struct FilterPredicates {
    comparisons: [FilterComparison; FILTER_MAX_PREDICATES],
    // Only group starts are read. Comparisons of one field share a single typed load.
    group_ends: [u32; FILTER_MAX_PREDICATES],
}

// SAFETY: A C-layout array of C-layout integer-only descriptors, matching FilterPredicates
// in exact_filter.cu. Input pointers are recorded and kept alive across each kernel launch.
unsafe impl DeviceRepr for FilterPredicates {}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct FilterColumn {
    input: u64,
    output: u64,
    width: u64,
}

#[repr(C)]
struct FilterColumns {
    columns: [FilterColumn; FILTER_MAX_COLUMNS],
}

// SAFETY: Matches FilterColumns in exact_filter.cu with no padding or Rust references.
// All embedded input/output pointers are separately recorded on the launch stream.
unsafe impl DeviceRepr for FilterColumns {}

struct SelectionMask {
    words: CudaSlice<u32>,
    tile_offsets: Option<CudaSlice<u64>>,
    group_offsets: Option<CudaSlice<u64>>,
    selected: usize,
}

/// One conjunct of a scan filter: a top-level field compared with a non-null literal.
#[derive(Clone, Debug)]
pub struct FieldComparison {
    field: FieldName,
    operator: CompareOp,
    /// The literal as a scalar of the field's primitive storage type.
    literal: Scalar,
    ptype: PType,
}

/// Comparison operators, numbered as `CompareOp` in `compare.cu`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum CompareOp {
    Eq = 0,
    NotEq = 1,
    Gt = 2,
    Gte = 3,
    Lt = 4,
    Lte = 5,
}

impl CompareOp {
    fn new(operator: Operator) -> Option<Self> {
        Some(match operator {
            Operator::Eq => Self::Eq,
            Operator::NotEq => Self::NotEq,
            Operator::Gt => Self::Gt,
            Operator::Gte => Self::Gte,
            Operator::Lt => Self::Lt,
            Operator::Lte => Self::Lte,
            _ => return None,
        })
    }

    /// The operator for swapped operands: `a < b` is `b > a`.
    fn swapped(self) -> Self {
        match self {
            Self::Eq | Self::NotEq => self,
            Self::Gt => Self::Lt,
            Self::Gte => Self::Lte,
            Self::Lt => Self::Gt,
            Self::Lte => Self::Gte,
        }
    }
}

/// The conjuncts of `filter` that compare a top-level primitive or temporal field of `scope`
/// with a literal. The others are left to the consumer.
pub fn field_comparisons(filter: &Expression, scope: &DType) -> VortexResult<Vec<FieldComparison>> {
    let filter = coerce_comparison_literals(filter.clone(), scope)?;
    let mut comparisons = Vec::new();
    collect_comparisons(&filter, scope, &mut comparisons);
    Ok(comparisons)
}

fn collect_comparisons(expr: &Expression, scope: &DType, comparisons: &mut Vec<FieldComparison>) {
    let Some(&operator) = expr.as_opt::<Binary>() else {
        return;
    };
    if operator == Operator::And {
        collect_comparisons(expr.child(0), scope, comparisons);
        collect_comparisons(expr.child(1), scope, comparisons);
        return;
    }
    let Some(operator) = CompareOp::new(operator) else {
        return;
    };
    let (lhs, rhs) = (expr.child(0), expr.child(1));
    let (field, literal, operator) = match (field_of(lhs), rhs.as_opt::<Literal>()) {
        (Some(field), Some(literal)) => (field, literal, operator),
        _ => match (field_of(rhs), lhs.as_opt::<Literal>()) {
            (Some(field), Some(literal)) => (field, literal, operator.swapped()),
            _ => return,
        },
    };
    let Some(field_dtype) = scope
        .as_struct_fields_opt()
        .and_then(|fields| fields.field(field))
    else {
        return;
    };
    let storage = match &field_dtype {
        DType::Extension(ext) => ext.storage_dtype().clone(),
        dtype => dtype.clone(),
    };
    let DType::Primitive(ptype, _) = storage else {
        return;
    };
    let literal = match literal.dtype() {
        DType::Extension(_) => literal.as_extension().to_storage_scalar(),
        _ => literal.clone(),
    };
    // Literals were cast to the field's type; anything else, like a null, stays unselected.
    if literal.is_null() || !matches!(literal.dtype(), DType::Primitive(p, _) if *p == ptype) {
        return;
    }
    if ptype == PType::F16 {
        return;
    }
    comparisons.push(FieldComparison {
        field: field.clone(),
        operator,
        literal,
        ptype,
    });
}

/// The field `expr` reads from the scope root, if it is a plain field reference.
fn field_of(expr: &Expression) -> Option<&FieldName> {
    let field = expr.as_opt::<GetItem>()?;
    expr.child(0).is_root().then_some(field)
}

/// Drop the rows of each batch of `stream` that fail any of `comparisons`, evaluated on the
/// GPU. Comparisons of fields the batches do not hold, or that hold nulls, select nothing, so
/// the result is a superset of the rows satisfying the comparisons, as the CUDA scan's filter
/// contract allows. Batches that would keep most rows, or hold columns other than non-null
/// primitives and temporals, are returned whole.
///
/// Up to `concurrency` batches are selected at once, matching the scan's split concurrency, so
/// selecting one batch does not hold back decoding the next.
pub fn select_rows(
    stream: SendableArrayStream,
    comparisons: Vec<FieldComparison>,
    session: VortexSession,
    concurrency: usize,
) -> SendableArrayStream {
    let dtype = stream.dtype().clone();
    let implementation = SelectionImpl::from_env();
    let comparisons: Arc<[FieldComparison]> = comparisons.into();
    let batches = stream
        .map(move |batch| {
            let comparisons = Arc::clone(&comparisons);
            let session = session.clone();
            async move {
                let mut ctx = CudaSession::create_execution_ctx(&session)?;
                select_batch(batch?, &comparisons, implementation, &mut ctx).await
            }
        })
        .buffered(concurrency.max(1));
    Box::pin(ArrayStreamAdapter::new(dtype, batches))
}

async fn select_batch(
    batch: ArrayRef,
    comparisons: &[FieldComparison],
    implementation: SelectionImpl,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = batch.len();
    let Some(batch_struct) = batch.as_opt::<Struct>() else {
        return Ok(batch);
    };
    let validity = batch_struct.validity()?;
    // Only primitive and temporal columns are compacted. Others, like strings, keep their
    // batch whole before any work, so they also keep their direct export paths.
    let compactable = batch_struct
        .struct_fields()
        .fields()
        .all(|dtype| primitive_storage(&dtype).is_some());
    if len == 0 || !compactable || !matches!(validity, Validity::NonNullable | Validity::AllValid) {
        return Ok(batch);
    }
    let names = batch_struct.struct_fields().names().clone();
    let mut fields: Vec<ArrayRef> = batch_struct.iter_unmasked_fields().cloned().collect();
    let whole = |fields: Vec<ArrayRef>| -> VortexResult<ArrayRef> {
        Ok(StructArray::try_new(names.clone(), fields, len, validity.clone())?.into_array())
    };

    let fused =
        implementation != SelectionImpl::Cub && supports_fused_filter(len, comparisons.len());
    let mut fused_inputs = Vec::new();
    let mut words: Option<CudaSlice<u32>> = None;
    for comparison in comparisons {
        let Some(index) = names.iter().position(|name| *name == comparison.field) else {
            continue;
        };
        // Keep the decoded column so selection and export do not decode it again.
        let canonical = fields[index].clone().execute_cuda(ctx).await?;
        fields[index] = canonical.clone().into_array();
        let Some(values) = comparable_values(canonical, comparison.ptype) else {
            continue;
        };
        if fused {
            fused_inputs.push((values, comparison));
            continue;
        }
        let combine = words.is_some();
        let target = match &mut words {
            Some(words) => words,
            None => words.insert(ctx.device_alloc::<u32>(len.div_ceil(32))?),
        };
        launch_compare(values, len, comparison, target, combine, ctx).await?;
    }
    let mask = if !fused_inputs.is_empty() {
        fused_selection_mask(
            fused_inputs,
            len,
            implementation == SelectionImpl::SharedTyped,
            ctx,
        )
        .await?
    } else if let Some(words) = words {
        let selected = count_selected(&words, ctx).await?;
        SelectionMask {
            words,
            tile_offsets: None,
            group_offsets: None,
            selected,
        }
    } else {
        return whole(fields);
    };
    let selected = mask.selected;
    if selected as f64 > SELECTED_FRACTION_MAX * len as f64 {
        return whole(fields);
    }

    let mut decoded = Vec::with_capacity(fields.len());
    for field in &fields {
        decoded.push(field.clone().execute_cuda(ctx).await?);
    }
    // Columns holding nulls would need their validity selected too.
    if !decoded
        .iter()
        .all(|field| selectable_values(field.clone()).is_some())
    {
        return whole(decoded.into_iter().map(Canonical::into_array).collect());
    }
    let selected_fields = if implementation != SelectionImpl::Cub
        && mask.tile_offsets.is_some()
        && decoded.len() <= FILTER_MAX_COLUMNS
    {
        select_fields_shared(decoded, &mask, len, ctx).await?
    } else {
        let mut selected_fields = Vec::with_capacity(decoded.len());
        for field in decoded {
            selected_fields.push(select_field(field, &mask.words, len, selected, ctx).await?);
        }
        selected_fields
    };
    Ok(StructArray::try_new(names, selected_fields, selected, validity)?.into_array())
}

/// The primitive type of a primitive column, or of a temporal column's storage.
fn primitive_storage(dtype: &DType) -> Option<PType> {
    let ptype = match dtype {
        DType::Primitive(ptype, _) => *ptype,
        DType::Extension(ext) => match ext.storage_dtype() {
            DType::Primitive(ptype, _) => *ptype,
            _ => return None,
        },
        _ => return None,
    };
    (ptype != PType::F16).then_some(ptype)
}

/// The number of set bits in `words`, read back without blocking the calling thread.
async fn count_selected(words: &CudaSlice<u32>, ctx: &mut CudaExecutionCtx) -> VortexResult<usize> {
    let mut count = ctx
        .stream()
        .alloc_zeros::<u64>(1)
        .map_err(|error| vortex_err!("Failed to allocate selection count: {error}"))?;
    let num_words = words.len() as u64;
    let function = ctx.load_function_with_suffixes("count_bits", &[])?;
    ctx.launch_kernel(&function, words.len(), |args| {
        args.arg(words).arg(&num_words).arg(&mut count);
    })?;
    read_selected_count(count).await
}

async fn read_selected_count(count: CudaSlice<u64>) -> VortexResult<usize> {
    let count = CudaDeviceBuffer::new(count)
        .copy_to_host(Alignment::of::<u64>())?
        .await?;
    let count = u64::from_le_bytes(
        count
            .as_slice()
            .try_into()
            .map_err(|_| vortex_err!("selection count has {} bytes", count.len()))?,
    );
    Ok(usize::try_from(count)?)
}

fn supports_fused_filter(len: usize, predicates: usize) -> bool {
    len > 0
        && (1..=FILTER_MAX_PREDICATES).contains(&predicates)
        && len.div_ceil(FILTER_TILE_ROWS) <= FILTER_MAX_TILES
}

fn filter_launch_config(blocks: usize) -> VortexResult<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (u32::try_from(blocks)?, 1, 1),
        block_dim: (u32::try_from(FILTER_THREADS)?, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn comparison_descriptor(comparison: &FieldComparison) -> VortexResult<FilterComparison> {
    let ptype = match comparison.ptype {
        PType::U8 => 0,
        PType::U16 => 1,
        PType::U32 => 2,
        PType::U64 => 3,
        PType::I8 => 4,
        PType::I16 => 5,
        PType::I32 => 6,
        PType::I64 => 7,
        PType::F32 => 8,
        PType::F64 => 9,
        PType::F16 => return Err(vortex_err!("f16 comparisons have no CUDA kernel")),
    };
    let literal = match_each_native_simd_ptype!(comparison.ptype, |T| {
        let value = comparison
            .literal
            .as_primitive()
            .typed_value::<T>()
            .ok_or_else(|| vortex_err!("comparison literal does not match its field"))?;
        let bytes = value.to_le_bytes();
        let mut padded = [0u8; 8];
        padded[..bytes.len()].copy_from_slice(&bytes);
        u64::from_le_bytes(padded)
    });
    Ok(FilterComparison {
        input: 0,
        literal,
        ptype,
        operator: comparison.operator as u32,
    })
}

/// Fuse the eligible conjunction and count tiles, then scan tile counts in launch-ordered
/// passes. The second prefix level is only needed above 256 tiles (524288 rows).
/// Comparisons referring to the same field must refer to the same batch's values.
async fn fused_selection_mask(
    mut comparisons: Vec<(PrimitiveArray, &FieldComparison)>,
    len: usize,
    typed_mask: bool,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<SelectionMask> {
    if !supports_fused_filter(len, comparisons.len()) {
        return Err(vortex_err!("unsupported fused filter dimensions"));
    }
    // Ranges and other repeated-field conjuncts load each decoded value only once. Preserve
    // first-field order for short-circuiting; never prioritize fields by name or schema.
    // This bounded sort handles at most 16 predicates, independently of the row count.
    let field_order: Vec<_> = comparisons
        .iter()
        .map(|(_, comparison)| comparison.field.clone())
        .collect();
    comparisons.sort_by_key(|(_, comparison)| {
        field_order
            .iter()
            .position(|field| field == comparison.field)
    });
    let mut predicates = FilterPredicates {
        comparisons: [FilterComparison::default(); FILTER_MAX_PREDICATES],
        group_ends: [0; FILTER_MAX_PREDICATES],
    };
    let num_predicates = u32::try_from(comparisons.len())?;
    let mut inputs = Vec::new();
    let mut groups = Vec::new();
    let mut start = 0;
    for group in
        comparisons.chunk_by(|(_, lhs), (_, rhs)| lhs.field == rhs.field && lhs.ptype == rhs.ptype)
    {
        let end = start + group.len();
        for (descriptor, (_, comparison)) in
            predicates.comparisons[start..end].iter_mut().zip(group)
        {
            *descriptor = comparison_descriptor(comparison)?;
        }
        predicates.group_ends[start] = u32::try_from(end)?;
        let buffer = group[0].0.clone().into_data_parts().buffer;
        inputs.push(ctx.ensure_on_device(buffer).await?);
        groups.push(start..end);
        start = end;
    }
    let views = inputs
        .iter()
        .map(|input| input.cuda_view::<u8>())
        .collect::<VortexResult<Vec<_>>>()?;
    let stream = ctx.stream().clone();
    let mut records = Vec::with_capacity(views.len());
    for (group, view) in groups.into_iter().zip(&views) {
        let (pointer, record) = view.device_ptr(&stream);
        for descriptor in &mut predicates.comparisons[group] {
            descriptor.input = pointer;
        }
        records.push(record);
    }
    let tiles = len.div_ceil(FILTER_TILE_ROWS);
    let mut words = ctx.device_alloc::<u32>(len.div_ceil(32))?;
    let mut counts = ctx.device_alloc::<u64>(tiles)?;
    let function = if typed_mask && predicates.group_ends[0] == num_predicates {
        // Group construction guarantees one primitive type and one tracked input pointer.
        let ptype = comparisons[0].1.ptype.to_string();
        ctx.load_function_with_suffixes("exact_filter", &["mask", &ptype])?
    } else {
        ctx.load_function_with_suffixes("exact_filter", &["mask"])?
    };
    let len_u64 = u64::try_from(len)?;
    ctx.launch_kernel_config(&function, filter_launch_config(tiles)?, len, |args| {
        args.arg(&predicates)
            .arg(&num_predicates)
            .arg(&len_u64)
            .arg(&mut words)
            .arg(&mut counts);
    })?;
    // Embedded pointers do not participate in LaunchArgs tracking automatically.
    drop(records);

    let (tile_offsets, group_counts) = prefix_counts(&counts, ctx)?;
    let (group_offsets, count) = if group_counts.len() > 1 {
        let (offsets, count) = prefix_counts(&group_counts, ctx)?;
        (Some(offsets), count)
    } else {
        (None, group_counts)
    };
    let selected = read_selected_count(count).await?;
    Ok(SelectionMask {
        words,
        tile_offsets: Some(tile_offsets),
        group_offsets,
        selected,
    })
}

fn prefix_counts(
    counts: &CudaSlice<u64>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(CudaSlice<u64>, CudaSlice<u64>)> {
    let groups = counts.len().div_ceil(FILTER_THREADS);
    let mut offsets = ctx.device_alloc::<u64>(counts.len())?;
    let mut group_counts = ctx.device_alloc::<u64>(groups)?;
    let len = u64::try_from(counts.len())?;
    let function = ctx.load_function_with_suffixes("exact_filter", &["prefix"])?;
    ctx.launch_kernel_config(
        &function,
        filter_launch_config(groups)?,
        counts.len(),
        |args| {
            args.arg(counts)
                .arg(&len)
                .arg(&mut offsets)
                .arg(&mut group_counts);
        },
    )?;
    Ok((offsets, group_counts))
}

/// One launch compacts every supported column using tile-local indices shared across columns.
/// Raw descriptor pointers are recorded on the stream, including slice offsets and outputs.
async fn select_fields_shared(
    fields: Vec<Canonical>,
    mask: &SelectionMask,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Vec<ArrayRef>> {
    let mut columns = FilterColumns {
        columns: [FilterColumn::default(); FILTER_MAX_COLUMNS],
    };
    let num_columns = u32::try_from(fields.len())?;
    let mut inputs = Vec::with_capacity(fields.len());
    let mut outputs = Vec::with_capacity(fields.len());
    let mut metadata = Vec::with_capacity(fields.len());
    for (index, field) in fields.into_iter().enumerate() {
        let ext_dtype = match &field {
            Canonical::Extension(ext) => Some(ext.ext_dtype().clone()),
            _ => None,
        };
        let values = selectable_values(field)
            .ok_or_else(|| vortex_err!("selected column is not a non-null primitive"))?;
        let ptype = values.ptype();
        let PrimitiveDataParts {
            buffer, validity, ..
        } = values.into_data_parts();
        let byte_len = mask
            .selected
            .checked_mul(ptype.byte_width())
            .ok_or_else(|| vortex_err!("selected column size overflow"))?;
        columns.columns[index].width = u64::try_from(ptype.byte_width())?;
        // Allocate at 8-byte alignment for all widths, with no per-column scratch space.
        outputs.push(ctx.device_alloc::<u64>(byte_len.div_ceil(8).max(1))?);
        metadata.push((ptype, validity, ext_dtype, byte_len));
        if mask.selected > 0 {
            inputs.push(ctx.ensure_on_device(buffer).await?);
        }
    }
    if mask.selected > 0 && num_columns > 0 {
        let views = inputs
            .iter()
            .map(|input| input.cuda_view::<u8>())
            .collect::<VortexResult<Vec<_>>>()?;
        let stream = ctx.stream().clone();
        let mut records = Vec::with_capacity(views.len() * 2);
        for ((descriptor, view), output) in columns.columns.iter_mut().zip(&views).zip(&mut outputs)
        {
            let (pointer, record) = view.device_ptr(&stream);
            descriptor.input = pointer;
            records.push(record);
            let (pointer, record) = output.device_ptr_mut(&stream);
            descriptor.output = pointer;
            records.push(record);
        }
        let tile_offsets = mask
            .tile_offsets
            .as_ref()
            .ok_or_else(|| vortex_err!("shared selection requires tile offsets"))?;
        // The kernel does not read group offsets when there is only one group.
        let group_offsets = mask.group_offsets.as_ref().unwrap_or(tile_offsets);
        let function = ctx.load_function_with_suffixes("exact_filter", &["scatter"])?;
        let len_u64 = u64::try_from(len)?;
        let config = filter_launch_config(len.div_ceil(FILTER_TILE_ROWS))?;
        ctx.launch_kernel_config(&function, config, len, |args| {
            args.arg(&columns)
                .arg(&num_columns)
                .arg(&len_u64)
                .arg(&mask.words)
                .arg(tile_offsets)
                .arg(group_offsets);
        })?;
        drop(records);
    }
    Ok(outputs
        .into_iter()
        .zip(metadata)
        .map(|(output, (ptype, validity, ext_dtype, byte_len))| {
            let output = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(output)))
                .slice(0..byte_len);
            let values = PrimitiveArray::from_buffer_handle(output, ptype, validity).into_array();
            match ext_dtype {
                Some(ext_dtype) => ExtensionArray::new(ext_dtype, values).into_array(),
                None => values,
            }
        })
        .collect())
}

/// The non-null primitive values of a decoded primitive or temporal column.
fn selectable_values(canonical: Canonical) -> Option<PrimitiveArray> {
    let ptype = match &canonical {
        Canonical::Primitive(values) => values.ptype(),
        Canonical::Extension(ext) => primitive_storage(ext.storage_array().dtype())?,
        _ => return None,
    };
    comparable_values(canonical, ptype).filter(|values| values.ptype() != PType::F16)
}

/// Keep the rows of a decoded column whose bit is set in `words`.
async fn select_field(
    field: Canonical,
    words: &CudaSlice<u32>,
    len: usize,
    selected: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ext_dtype = match &field {
        Canonical::Extension(ext) => Some(ext.ext_dtype().clone()),
        _ => None,
    };
    let values = selectable_values(field)
        .ok_or_else(|| vortex_err!("selected column is not a non-null primitive"))?;
    let values = match_each_native_simd_ptype!(values.ptype(), |T| {
        select_primitive::<T>(values, words, len, selected, ctx).await?
    });
    Ok(match ext_dtype {
        Some(ext_dtype) => ExtensionArray::new(ext_dtype, values.into_array()).into_array(),
        None => values.into_array(),
    })
}

async fn select_primitive<T: NativePType + DeviceRepr + CubFilterable + Send + Sync + 'static>(
    values: PrimitiveArray,
    words: &CudaSlice<u32>,
    len: usize,
    selected: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let PrimitiveDataParts {
        buffer, validity, ..
    } = values.into_data_parts();
    let input = ctx.ensure_on_device(buffer).await?;
    let temp_bytes = T::get_temp_size(i64::try_from(len)?)
        .map_err(|error| vortex_err!("CUB select temp size failed: {error}"))?;
    let mut temp = ctx.device_alloc::<u8>(temp_bytes.max(1))?;
    let mut output = ctx.device_alloc::<T>(selected.max(1))?;
    let mut num_selected = ctx.device_alloc::<i64>(1)?;

    let stream = ctx.stream();
    let input = input.cuda_view::<T>()?;
    let (input_ptr, record_input) = input.device_ptr(stream);
    let (words_ptr, record_words) = words.device_ptr(stream);
    let (temp_ptr, record_temp) = temp.device_ptr_mut(stream);
    let (output_ptr, record_output) = output.device_ptr_mut(stream);
    let (num_selected_ptr, record_num_selected) = num_selected.device_ptr_mut(stream);
    let stream_ptr = stream.cu_stream() as cudaStream_t;
    let num_items = i64::try_from(len)?;
    ctx.launch_external(selected, || {
        // SAFETY: Every pointer is a live device allocation of the stated size, recorded on
        // this stream: `len` input values, `len` bits of words, `selected.max(1)` outputs.
        unsafe {
            T::filter_bitmask(
                temp_ptr as *mut c_void,
                temp_bytes,
                input_ptr as *const T,
                words_ptr as *const u8,
                0,
                output_ptr as *mut T,
                num_selected_ptr as *mut i64,
                num_items,
                stream_ptr,
            )
        }
        .map_err(|error| vortex_err!("CUB select failed: {error}"))
    })?;
    drop((
        record_input,
        record_words,
        record_temp,
        record_output,
        record_num_selected,
    ));

    let output = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(output)))
        .slice(0..selected * size_of::<T>());
    Ok(PrimitiveArray::from_buffer_handle(
        output,
        T::PTYPE,
        validity,
    ))
}

/// The primitive values of a decoded field, if no value is null.
fn comparable_values(canonical: Canonical, ptype: PType) -> Option<PrimitiveArray> {
    let values = match canonical {
        Canonical::Primitive(values) => values,
        Canonical::Extension(ext) => ext.storage_array().as_opt::<Primitive>()?.into_owned(),
        _ => return None,
    };
    if values.ptype() != ptype {
        return None;
    }
    matches!(
        values.validity(),
        Ok(Validity::NonNullable | Validity::AllValid)
    )
    .then_some(values)
}

async fn launch_compare(
    values: PrimitiveArray,
    len: usize,
    comparison: &FieldComparison,
    words: &mut CudaSlice<u32>,
    combine: bool,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    macro_rules! launch {
        ($T:ty) => {
            launch_compare_typed::<$T>(values, len, comparison, words, combine, ctx).await
        };
    }
    match comparison.ptype {
        PType::U8 => launch!(u8),
        PType::U16 => launch!(u16),
        PType::U32 => launch!(u32),
        PType::U64 => launch!(u64),
        PType::I8 => launch!(i8),
        PType::I16 => launch!(i16),
        PType::I32 => launch!(i32),
        PType::I64 => launch!(i64),
        PType::F32 => launch!(f32),
        PType::F64 => launch!(f64),
        PType::F16 => Err(vortex_err!("f16 comparisons have no CUDA kernel")),
    }
}

async fn launch_compare_typed<T: NativePType + DeviceRepr + Send + Sync + 'static>(
    values: PrimitiveArray,
    len: usize,
    comparison: &FieldComparison,
    words: &mut CudaSlice<u32>,
    combine: bool,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    let literal = comparison
        .literal
        .as_primitive()
        .typed_value::<T>()
        .ok_or_else(|| vortex_err!("comparison literal does not match its field"))?;
    let PrimitiveDataParts { buffer, .. } = values.into_data_parts();
    let buffer = ctx.ensure_on_device(buffer).await?;
    let values = buffer.cuda_view::<T>()?;
    let len_u64 = len as u64;
    let operator = comparison.operator as u32;
    let combine = u32::from(combine);
    let function = ctx.load_function("compare", &[T::PTYPE])?;
    ctx.launch_kernel(&function, len, |args| {
        args.arg(&values)
            .arg(&len_u64)
            .arg(&operator)
            .arg(&literal)
            .arg(words)
            .arg(&combine);
    })
}

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use rstest::rstest;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::TemporalArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::expr::and;
    use vortex::array::expr::col;
    use vortex::array::expr::eq;
    use vortex::array::expr::gt;
    use vortex::array::expr::gt_eq;
    use vortex::array::expr::lit;
    use vortex::array::expr::lt;
    use vortex::array::expr::or;
    use vortex::array::extension::datetime::TimeUnit;
    use vortex::array::session::ArraySession;
    use vortex::error::VortexExpect;
    use vortex::mask::Mask;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::CudaSession;
    use crate::cuda_session;

    /// Columns `x = 0..5000` and `y = x / 2` as f64.
    fn batch() -> VortexResult<ArrayRef> {
        let x = PrimitiveArray::from_iter(0i32..5000).into_array();
        let y = PrimitiveArray::from_iter((0..5000).map(|x| f64::from(x) / 2.0)).into_array();
        Ok(StructArray::from_fields(&[("x", x), ("y", y)])?.into_array())
    }

    async fn select(batch: ArrayRef, filter: Expression) -> VortexResult<ArrayRef> {
        select_with_impl(batch, filter, SelectionImpl::Shared).await
    }

    async fn select_with_impl(
        batch: ArrayRef,
        filter: Expression,
        implementation: SelectionImpl,
    ) -> VortexResult<ArrayRef> {
        let comparisons = field_comparisons(&filter, batch.dtype())?;
        let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
        let selected = select_batch(batch, &comparisons, implementation, &mut ctx).await?;
        Ok(selected
            .execute_cuda(&mut ctx)
            .await?
            .into_host()
            .await?
            .into_array())
    }

    /// A selective conjunction keeps exactly the matching rows of every column.
    #[crate::test]
    async fn selective_comparisons_drop_failing_rows() -> VortexResult<()> {
        let filter = and(gt_eq(col("x"), lit(1000i64)), lt(col("x"), lit(1500i32)));
        let selected = select(batch()?, filter).await?;
        let x = PrimitiveArray::from_iter(1000i32..1500).into_array();
        let y = PrimitiveArray::from_iter((1000..1500).map(|x| f64::from(x) / 2.0)).into_array();
        let expected = StructArray::from_fields(&[("x", x), ("y", y)])?.into_array();
        let mut ctx = VortexSession::empty()
            .with::<ArraySession>()
            .create_execution_ctx();
        assert_arrays_eq!(expected, selected, &mut ctx);
        Ok(())
    }

    /// A literal on the left swaps the operator; a float column compares as floats.
    #[crate::test]
    async fn literal_on_the_left_swaps_the_operator() -> VortexResult<()> {
        let selected = select(batch()?, gt(lit(100.0f64), col("y"))).await?;
        assert_eq!(selected.len(), 200);
        Ok(())
    }

    /// Batches that would keep most rows are returned whole.
    #[crate::test]
    async fn unselective_comparisons_keep_the_batch() -> VortexResult<()> {
        let selected = select(batch()?, lt(col("x"), lit(4000i32))).await?;
        assert_eq!(selected.len(), 5000);
        Ok(())
    }

    /// Dates compare on their day-count storage.
    #[rstest]
    #[crate::test]
    async fn dates_compare_on_storage(
        #[values(SelectionImpl::Shared, SelectionImpl::SharedTyped)] implementation: SelectionImpl,
    ) -> VortexResult<()> {
        let days = PrimitiveArray::from_iter(0i32..5000).into_array();
        let date = TemporalArray::new_date(days, TimeUnit::Days).into_array();
        let batch = StructArray::from_fields(&[("d", date)])?.into_array();
        let selected =
            select_with_impl(batch, gt_eq(col("d"), lit(4500i32)), implementation).await?;
        assert_eq!(selected.len(), 500);
        Ok(())
    }

    /// Columns other than non-null primitives are not compacted, so their batches stay whole.
    #[crate::test]
    async fn batches_with_strings_stay_whole() -> VortexResult<()> {
        let x = PrimitiveArray::from_iter(0i32..5000).into_array();
        let s = VarBinViewArray::from_iter_str((0..5000).map(|i| format!("s{i}"))).into_array();
        let batch = StructArray::from_fields(&[("x", x), ("s", s)])?.into_array();
        let selected = select(batch, lt(col("x"), lit(100i32))).await?;
        assert_eq!(selected.len(), 5000);
        Ok(())
    }

    /// Disjunctions and computed operands are not comparisons of a field with a literal.
    #[test]
    fn other_predicates_select_nothing() -> VortexResult<()> {
        let dtype = batch()?.dtype().clone();
        let disjunction = or(lt(col("x"), lit(1i32)), gt(col("x"), lit(4998i32)));
        assert!(field_comparisons(&disjunction, &dtype)?.is_empty());
        let columns = lt(col("x"), col("y"));
        assert!(field_comparisons(&columns, &dtype)?.is_empty());
        Ok(())
    }

    #[rstest]
    #[case(None, None, SelectionImpl::SharedTyped)]
    #[case(None, Some("0"), SelectionImpl::Shared)]
    #[case(None, Some("true"), SelectionImpl::SharedTyped)]
    #[case(None, Some(""), SelectionImpl::SharedTyped)]
    #[case(None, Some("1"), SelectionImpl::SharedTyped)]
    #[case(Some("shared"), None, SelectionImpl::SharedTyped)]
    #[case(Some("shared"), Some("0"), SelectionImpl::Shared)]
    #[case(Some("shared"), Some("1"), SelectionImpl::SharedTyped)]
    #[case(Some("cub"), None, SelectionImpl::Cub)]
    #[case(Some("cub"), Some("0"), SelectionImpl::Cub)]
    #[case(Some("cub"), Some("1"), SelectionImpl::Cub)]
    fn test_selection_settings(
        #[case] implementation: Option<&str>,
        #[case] typed_filter: Option<&str>,
        #[case] expected: SelectionImpl,
    ) {
        assert_eq!(
            SelectionImpl::from_settings(implementation, typed_filter),
            expected
        );
    }

    #[test]
    fn test_inline_descriptor_layouts() {
        assert_eq!(size_of::<FilterComparison>(), 24);
        assert_eq!(size_of::<FilterPredicates>(), 448);
        assert_eq!(size_of::<FilterColumn>(), 24);
        assert_eq!(size_of::<FilterColumns>(), 768);
        assert_eq!(offset_of!(FilterComparison, literal), 8);
        assert_eq!(offset_of!(FilterComparison, ptype), 16);
        assert_eq!(offset_of!(FilterComparison, operator), 20);
        assert_eq!(offset_of!(FilterPredicates, group_ends), 384);
        assert_eq!(offset_of!(FilterColumn, output), 8);
        assert_eq!(offset_of!(FilterColumn, width), 16);
    }

    #[test]
    fn test_fused_capacity_bounds() {
        assert!(!supports_fused_filter(0, 1));
        assert!(!supports_fused_filter(1, 0));
        assert!(supports_fused_filter(1, 1));
        assert!(supports_fused_filter(
            FILTER_TILE_ROWS * FILTER_MAX_TILES,
            FILTER_MAX_PREDICATES,
        ));
        assert!(!supports_fused_filter(
            FILTER_TILE_ROWS * FILTER_MAX_TILES + 1,
            1,
        ));
        assert!(!supports_fused_filter(1, FILTER_MAX_PREDICATES + 1));
        assert!(!supports_fused_filter(usize::MAX, 1));
    }

    #[rstest]
    #[case(Scalar::from(-2i8), PType::I8, 0xfe)]
    #[case(Scalar::from(-2i16), PType::I16, 0xfffe)]
    #[case(Scalar::from(-2i32), PType::I32, 0xffff_fffe)]
    #[case(Scalar::from(-2i64), PType::I64, 0xffff_ffff_ffff_fffe)]
    #[case(Scalar::from(u64::MAX), PType::U64, u64::MAX)]
    #[case(Scalar::from(-0.0f32), PType::F32, 0x8000_0000)]
    #[case(Scalar::from(-0.0f64), PType::F64, 0x8000_0000_0000_0000)]
    fn test_literal_descriptor_bits(
        #[case] literal: Scalar,
        #[case] ptype: PType,
        #[case] expected: u64,
    ) -> VortexResult<()> {
        let descriptor = comparison_descriptor(&FieldComparison {
            field: "value".into(),
            operator: CompareOp::Gte,
            literal,
            ptype,
        })?;
        assert_eq!(descriptor.literal, expected);
        assert_eq!(descriptor.operator, 3);
        Ok(())
    }

    /// Every primitive width, signed/unsigned/float bit patterns, nullable-but-all-valid,
    /// and temporal storage are copied together using the same tile-local row indices.
    fn mixed_batch(len: i32) -> VortexResult<ArrayRef> {
        let rows = 0..len;
        let key = PrimitiveArray::from_iter(rows.clone().map(|row| row % 7)).into_array();
        let u8s = PrimitiveArray::from_option_iter(
            rows.clone()
                .map(|row| Some(u8::try_from(row % 251).vortex_expect("bounded remainder"))),
        )
        .into_array();
        let i8s = PrimitiveArray::from_iter(
            rows.clone()
                .map(|row| i8::try_from(row % 127 - 63).vortex_expect("bounded remainder")),
        )
        .into_array();
        let u16s = PrimitiveArray::from_iter(
            rows.clone()
                .map(|row| u16::try_from(row % 65536).vortex_expect("bounded remainder")),
        )
        .into_array();
        let i16s = PrimitiveArray::from_iter(
            rows.clone()
                .map(|row| i16::try_from(row % 1000 - 500).vortex_expect("bounded remainder")),
        )
        .into_array();
        let u32s = PrimitiveArray::from_iter(rows.clone().map(|row| 4_000_000_000u32 + row as u32))
            .into_array();
        let i64s = PrimitiveArray::from_iter(rows.clone().map(|row| i64::MIN + i64::from(row)))
            .into_array();
        let u64s =
            PrimitiveArray::from_iter(rows.clone().map(|row| u64::MAX - row as u64)).into_array();
        let f32s =
            PrimitiveArray::from_iter(rows.clone().map(|row| -(row as f32) / 10.0)).into_array();
        let f64s =
            PrimitiveArray::from_iter(rows.clone().map(|row| f64::from(row) / 10.0)).into_array();
        let dates =
            TemporalArray::new_date(PrimitiveArray::from_iter(rows).into_array(), TimeUnit::Days)
                .into_array();
        Ok(StructArray::from_fields(&[
            ("key", key),
            ("u8", u8s),
            ("i8", i8s),
            ("u16", u16s),
            ("i16", i16s),
            ("u32", u32s),
            ("i64", i64s),
            ("u64", u64s),
            ("f32", f32s),
            ("f64", f64s),
            ("d", dates),
        ])?
        .into_array())
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(31)]
    #[case(32)]
    #[case(33)]
    #[case(2047)]
    #[case(2048)]
    #[case(2049)]
    #[case(524289)]
    #[crate::test]
    async fn test_multi_column_widths_and_prefix_boundaries(
        #[case] len: i32,
        #[values(SelectionImpl::Cub, SelectionImpl::Shared, SelectionImpl::SharedTyped)]
        implementation: SelectionImpl,
    ) -> VortexResult<()> {
        let batch = mixed_batch(len)?;
        let expected = batch.filter(Mask::from_iter((0..len).map(|row| row % 7 == 6)))?;
        let filter = and(
            eq(col("key"), lit(6i32)),
            and(
                gt_eq(col("u32"), lit(4_000_000_000u32)),
                lt(col("f64"), lit(1.0e12f64)),
            ),
        );
        let selected = select_with_impl(batch, filter, implementation).await?;
        let mut ctx = VortexSession::empty()
            .with::<ArraySession>()
            .create_execution_ctx();
        assert_arrays_eq!(expected, selected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(0, 0)]
    #[case(2500, 2500)]
    #[case(2501, 5000)]
    #[case(5000, 5000)]
    #[crate::test]
    async fn test_empty_half_and_dense_masks(
        #[case] cutoff: i32,
        #[case] expected_len: usize,
        #[values(SelectionImpl::Cub, SelectionImpl::Shared, SelectionImpl::SharedTyped)]
        implementation: SelectionImpl,
    ) -> VortexResult<()> {
        let batch = batch()?;
        let expected = batch.slice(0..expected_len)?;
        let selected = select_with_impl(batch, lt(col("x"), lit(cutoff)), implementation).await?;
        let mut ctx = VortexSession::empty()
            .with::<ArraySession>()
            .create_execution_ctx();
        assert_arrays_eq!(expected, selected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(PType::U8)]
    #[case(PType::U16)]
    #[case(PType::U32)]
    #[case(PType::U64)]
    #[case(PType::I8)]
    #[case(PType::I16)]
    #[case(PType::I32)]
    #[case(PType::I64)]
    #[case(PType::F32)]
    #[case(PType::F64)]
    #[crate::test]
    async fn test_fused_comparison_types_and_operators(
        #[case] ptype: PType,
        #[values(false, true)] typed_mask: bool,
    ) -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
        for operator in [
            CompareOp::Eq,
            CompareOp::NotEq,
            CompareOp::Gt,
            CompareOp::Gte,
            CompareOp::Lt,
            CompareOp::Lte,
        ] {
            let (values, literal, expected) = match_each_native_simd_ptype!(ptype, |T| {
                let values = [T::MIN, 0 as T, 1 as T, T::MAX];
                let literal = 1 as T;
                let expected = values
                    .iter()
                    .enumerate()
                    .fold(0u32, |mask, (index, value)| {
                        let keep = match operator {
                            CompareOp::Eq => *value == literal,
                            CompareOp::NotEq => *value != literal,
                            CompareOp::Gt => *value > literal,
                            CompareOp::Gte => *value >= literal,
                            CompareOp::Lt => *value < literal,
                            CompareOp::Lte => *value <= literal,
                        };
                        mask | (u32::from(keep) << index)
                    });
                (
                    PrimitiveArray::from_iter(values),
                    Scalar::from(literal),
                    expected,
                )
            });
            let comparison = FieldComparison {
                field: "x".into(),
                operator,
                literal,
                ptype,
            };
            let mask =
                fused_selection_mask(vec![(values, &comparison)], 4, typed_mask, &mut ctx).await?;
            let words = ctx
                .stream()
                .clone_dtoh(&mask.words)
                .map_err(|error| vortex_err!("Failed to read selection mask: {error}"))?;
            assert_eq!(words, vec![expected]);
            assert_eq!(mask.selected, expected.count_ones() as usize);
        }
        Ok(())
    }

    #[rstest]
    #[case(PType::I8, i64::from(i8::MIN), i64::from(i8::MAX))]
    #[case(PType::I16, i64::from(i16::MIN), i64::from(i16::MAX))]
    #[case(PType::I32, i64::from(i32::MIN), i64::from(i32::MAX))]
    #[case(PType::I64, i64::MIN, i64::MAX)]
    #[crate::test]
    async fn test_signed_literal_reconstruction_all_operators(
        #[case] ptype: PType,
        #[case] min: i64,
        #[case] max: i64,
        #[values(false, true)] typed_mask: bool,
    ) -> VortexResult<()> {
        // The oracle compares original signed CPU values, not serialized descriptor bits.
        let values = [min, min + 1, -3, -2, -1, 0, 1, 2, max - 1, max];
        let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
        for literal in [min, min + 1, -2, -1, 0, 1, max - 1, max] {
            macro_rules! narrow_input {
                ($T:ty) => {
                    (
                        PrimitiveArray::from_iter(values.map(|value| {
                            <$T>::try_from(value).vortex_expect("test value fits signed type")
                        })),
                        Scalar::from(
                            <$T>::try_from(literal).vortex_expect("test literal fits signed type"),
                        ),
                    )
                };
            }
            let (input, scalar) = match ptype {
                PType::I8 => narrow_input!(i8),
                PType::I16 => narrow_input!(i16),
                PType::I32 => narrow_input!(i32),
                PType::I64 => (PrimitiveArray::from_iter(values), Scalar::from(literal)),
                _ => unreachable!("signed integer cases only"),
            };
            for operator in [
                CompareOp::Eq,
                CompareOp::NotEq,
                CompareOp::Gt,
                CompareOp::Gte,
                CompareOp::Lt,
                CompareOp::Lte,
            ] {
                let expected = values
                    .iter()
                    .enumerate()
                    .fold(0u32, |word, (index, &value)| {
                        let keep = match operator {
                            CompareOp::Eq => value == literal,
                            CompareOp::NotEq => value != literal,
                            CompareOp::Gt => value > literal,
                            CompareOp::Gte => value >= literal,
                            CompareOp::Lt => value < literal,
                            CompareOp::Lte => value <= literal,
                        };
                        word | (u32::from(keep) << index)
                    });
                let comparison = FieldComparison {
                    field: "x".into(),
                    operator,
                    literal: scalar.clone(),
                    ptype,
                };
                let mask = fused_selection_mask(
                    vec![(input.clone(), &comparison)],
                    values.len(),
                    typed_mask,
                    &mut ctx,
                )
                .await?;
                let words = ctx
                    .stream()
                    .clone_dtoh(&mask.words)
                    .map_err(|error| vortex_err!("Failed to read selection mask: {error}"))?;
                assert_eq!(
                    words,
                    vec![expected],
                    "{ptype} {operator:?} {literal}, typed={typed_mask}"
                );
                assert_eq!(
                    mask.selected,
                    expected.count_ones() as usize,
                    "{ptype} {operator:?} {literal}, typed={typed_mask}"
                );
            }
        }
        Ok(())
    }

    #[rstest]
    #[crate::test]
    async fn test_single_group_comparison_counts(
        #[values(false, true)] typed_mask: bool,
    ) -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0i32..4099);
        let len = values.len();
        let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
        for count in 1..=FILTER_MAX_PREDICATES {
            let comparisons: Vec<_> = (0..count)
                .map(|index| {
                    let (operator, literal) = match index % 3 {
                        0 => (CompareOp::Gte, 123i32),
                        1 => (CompareOp::Lt, 333i32),
                        _ => (CompareOp::NotEq, 200i32),
                    };
                    FieldComparison {
                        field: "x".into(),
                        operator,
                        literal: Scalar::from(literal),
                        ptype: PType::I32,
                    }
                })
                .collect();
            let inputs = comparisons
                .iter()
                .map(|comparison| (values.clone(), comparison))
                .collect();
            let mask = fused_selection_mask(inputs, len, typed_mask, &mut ctx).await?;
            let mut expected = vec![0u32; len.div_ceil(32)];
            for row in 0..len {
                if row >= 123 && (count < 2 || row < 333) && (count < 3 || row != 200) {
                    expected[row / 32] |= 1u32 << (row % 32);
                }
            }
            let words = ctx
                .stream()
                .clone_dtoh(&mask.words)
                .map_err(|error| vortex_err!("Failed to read selection mask: {error}"))?;
            assert_eq!(words, expected, "comparison count {count}");
            assert_eq!(
                mask.selected,
                expected
                    .iter()
                    .map(|word| word.count_ones() as usize)
                    .sum::<usize>()
            );
        }
        Ok(())
    }

    /// Interleaved predicates must form complete groups even at the descriptor capacity.
    #[rstest]
    #[case(SelectionImpl::Cub)]
    #[case(SelectionImpl::Shared)]
    #[case(SelectionImpl::SharedTyped)]
    #[crate::test]
    async fn test_repeated_fields_at_predicate_capacity(
        #[case] implementation: SelectionImpl,
    ) -> VortexResult<()> {
        let batch = batch()?;
        let expected = batch.slice(123..333)?;
        let filter = and(
            and(gt_eq(col("y"), lit(0.0f64)), gt_eq(col("x"), lit(123i32))),
            and(lt(col("y"), lit(200.0f64)), lt(col("x"), lit(333i32))),
        );
        let comparisons = field_comparisons(&filter, batch.dtype())?;
        let comparisons: Vec<_> = comparisons
            .into_iter()
            .cycle()
            .take(FILTER_MAX_PREDICATES)
            .collect();
        let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
        let selected = select_batch(batch, &comparisons, implementation, &mut ctx).await?;
        let selected = selected
            .execute_cuda(&mut ctx)
            .await?
            .into_host()
            .await?
            .into_array();
        let mut ctx = VortexSession::empty()
            .with::<ArraySession>()
            .create_execution_ctx();
        assert_arrays_eq!(expected, selected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(PType::F32)]
    #[case(PType::F64)]
    #[crate::test]
    async fn test_float_mask_all_operators_and_ambiguous_literals(
        #[case] ptype: PType,
        #[values(false, true)] typed_mask: bool,
    ) -> VortexResult<()> {
        // All values (including the subnormal) are represented exactly by both float widths.
        let values = [
            -f64::NAN,
            f64::NAN,
            -0.0,
            0.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
            -2.0,
            -1.0,
            1.0,
            2.0,
            f64::from(f32::from_bits(1)),
        ];
        let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
        for literal in [
            -f64::NAN,
            f64::NAN,
            -0.0,
            0.0,
            -1.0,
            1.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
        ] {
            for operator in [
                CompareOp::Eq,
                CompareOp::NotEq,
                CompareOp::Gt,
                CompareOp::Gte,
                CompareOp::Lt,
                CompareOp::Lte,
            ] {
                let expected = values
                    .iter()
                    .enumerate()
                    .fold(0u32, |mask, (index, &value)| {
                        let ambiguous =
                            value.is_nan() || literal.is_nan() || (value == 0.0 && literal == 0.0);
                        let keep = ambiguous
                            || match operator {
                                CompareOp::Eq => value == literal,
                                CompareOp::NotEq => value != literal,
                                CompareOp::Gt => value > literal,
                                CompareOp::Gte => value >= literal,
                                CompareOp::Lt => value < literal,
                                CompareOp::Lte => value <= literal,
                            };
                        mask | (u32::from(keep) << index)
                    });
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "Finite test values are exactly representable in f32"
                )]
                let (input, literal) = match ptype {
                    PType::F32 => (
                        PrimitiveArray::from_iter(values.map(|value| value as f32)),
                        Scalar::from(literal as f32),
                    ),
                    PType::F64 => (PrimitiveArray::from_iter(values), Scalar::from(literal)),
                    _ => unreachable!("float cases only"),
                };
                let comparison = FieldComparison {
                    field: "value".into(),
                    operator,
                    literal,
                    ptype,
                };
                let mask = fused_selection_mask(
                    vec![(input.clone(), &comparison)],
                    values.len(),
                    typed_mask,
                    &mut ctx,
                )
                .await?;
                let words = ctx
                    .stream()
                    .clone_dtoh(&mask.words)
                    .map_err(|error| vortex_err!("Failed to read fused selection mask: {error}"))?;
                assert_eq!(words, vec![expected]);
                assert_eq!(mask.selected, expected.count_ones() as usize);

                // The standalone baseline must obey the same contract, including tails.
                let mut baseline = ctx.device_alloc::<u32>(1)?;
                launch_compare(
                    input,
                    values.len(),
                    &comparison,
                    &mut baseline,
                    false,
                    &mut ctx,
                )
                .await?;
                let words = ctx.stream().clone_dtoh(&baseline).map_err(|error| {
                    vortex_err!("Failed to read baseline selection mask: {error}")
                })?;
                assert_eq!(words, vec![expected]);
            }
        }
        Ok(())
    }

    #[rstest]
    #[crate::test]
    async fn test_device_slice_offsets_and_cross_stream_lifetimes(
        #[values(SelectionImpl::Shared, SelectionImpl::SharedTyped)] implementation: SelectionImpl,
    ) -> VortexResult<()> {
        let session = cuda_session();
        let producer = CudaSession::create_execution_ctx(&session)?;
        let x = producer
            .copy_to_device((0i32..5003).collect::<Vec<_>>())?
            .await?;
        let y = producer
            .copy_to_device((0u64..5003).map(|row| u64::MAX - row).collect::<Vec<_>>())?
            .await?;
        let x = PrimitiveArray::from_buffer_handle(
            x.slice(3 * 4..5003 * 4),
            PType::I32,
            Validity::AllValid,
        )
        .into_array();
        let y = PrimitiveArray::from_buffer_handle(
            y.slice(3 * 8..5003 * 8),
            PType::U64,
            Validity::NonNullable,
        )
        .into_array();
        let batch = StructArray::from_fields(&[("x", x), ("y", y)])?.into_array();
        let filter = and(gt_eq(col("x"), lit(1000i32)), lt(col("x"), lit(1100i32)));
        let comparisons = field_comparisons(&filter, batch.dtype())?;
        let mut consumer = CudaSession::create_execution_ctx(&session)?;
        let selected = select_batch(batch, &comparisons, implementation, &mut consumer).await?;
        // Drop both contexts and all source handles before a third stream consumes outputs.
        drop((producer, consumer));
        let mut reader = CudaSession::create_execution_ctx(&session)?;
        let selected = selected
            .execute_cuda(&mut reader)
            .await?
            .into_host()
            .await?
            .into_array();
        let expected = StructArray::from_fields(&[
            (
                "x",
                PrimitiveArray::from_option_iter((1000i32..1100).map(Some)).into_array(),
            ),
            (
                "y",
                PrimitiveArray::from_iter((1000u64..1100).map(|row| u64::MAX - row)).into_array(),
            ),
        ])?
        .into_array();
        let mut ctx = VortexSession::empty()
            .with::<ArraySession>()
            .create_execution_ctx();
        assert_arrays_eq!(expected, selected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[crate::test]
    async fn test_nullable_columns_keep_whole_batch(
        #[values(SelectionImpl::Shared, SelectionImpl::SharedTyped)] implementation: SelectionImpl,
    ) -> VortexResult<()> {
        let x = PrimitiveArray::from_iter(0i32..100).into_array();
        let y = PrimitiveArray::from_option_iter((0i32..100).map(|row| (row != 50).then_some(row)))
            .into_array();
        let batch = StructArray::from_fields(&[("x", x), ("y", y)])?.into_array();
        for filter in [lt(col("x"), lit(5i32)), lt(col("y"), lit(5i32))] {
            let selected = select_with_impl(batch.clone(), filter, implementation).await?;
            let mut ctx = VortexSession::empty()
                .with::<ArraySession>()
                .create_execution_ctx();
            assert_arrays_eq!(batch, selected, &mut ctx);
        }
        Ok(())
    }

    #[rstest]
    #[case(SelectionImpl::Cub)]
    #[case(SelectionImpl::Shared)]
    #[case(SelectionImpl::SharedTyped)]
    #[crate::test]
    async fn test_float_ambiguities_preserve_cpu_superset(
        #[case] implementation: SelectionImpl,
    ) -> VortexResult<()> {
        let values = PrimitiveArray::from_iter([f64::NAN, -0.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]);
        let batch = StructArray::from_fields(&[("x", values.into_array())])?.into_array();
        // Vortex orders NaN above finite values and +0 above -0; IEEE consumers differ.
        // Retaining all three ambiguous rows is safe for either consumer.
        let expected = batch.slice(0..3)?;
        let selected = select_with_impl(batch, gt(col("x"), lit(-0.0f64)), implementation).await?;
        let mut ctx = VortexSession::empty()
            .with::<ArraySession>()
            .create_execution_ctx();
        assert_arrays_eq!(expected, selected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[crate::test]
    async fn test_predicate_capacity_falls_back_to_cub(
        #[values(SelectionImpl::Shared, SelectionImpl::SharedTyped)] implementation: SelectionImpl,
    ) -> VortexResult<()> {
        let batch = batch()?;
        let expected = batch.slice(0..100)?;
        let comparison = field_comparisons(&lt(col("x"), lit(100i32)), batch.dtype())?;
        let comparisons = vec![comparison[0].clone(); FILTER_MAX_PREDICATES + 1];
        let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
        let selected = select_batch(batch, &comparisons, implementation, &mut ctx).await?;
        let selected = selected
            .execute_cuda(&mut ctx)
            .await?
            .into_host()
            .await?
            .into_array();
        let mut ctx = VortexSession::empty()
            .with::<ArraySession>()
            .create_execution_ctx();
        assert_arrays_eq!(expected, selected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(8193, 2048, 4096)]
    #[case(528385, 522240, 526337)]
    #[crate::test]
    async fn test_empty_and_full_tiles_across_prefix_groups(
        #[case] len: i32,
        #[case] start: i32,
        #[case] end: i32,
        #[values(SelectionImpl::Shared, SelectionImpl::SharedTyped)] implementation: SelectionImpl,
    ) -> VortexResult<()> {
        let rows = PrimitiveArray::from_iter(0..len).into_array();
        let batch = StructArray::from_fields(&[("x", rows)])?.into_array();
        let expected = batch.slice(usize::try_from(start)?..usize::try_from(end)?)?;
        let selected = select_with_impl(
            batch,
            and(gt_eq(col("x"), lit(start)), lt(col("x"), lit(end))),
            implementation,
        )
        .await?;
        let mut ctx = VortexSession::empty()
            .with::<ArraySession>()
            .create_execution_ctx();
        assert_arrays_eq!(expected, selected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[crate::test]
    async fn test_wide_projection_falls_back_to_cub(
        #[values(SelectionImpl::Shared, SelectionImpl::SharedTyped)] implementation: SelectionImpl,
    ) -> VortexResult<()> {
        let rows = PrimitiveArray::from_iter(0i32..100).into_array();
        let fields: Vec<_> = (0..=FILTER_MAX_COLUMNS)
            .map(|index| (format!("c{index}"), rows.clone()))
            .collect();
        let batch = StructArray::from_fields(&fields)?.into_array();
        let expected = batch.slice(0..5)?;
        let selected = select_with_impl(batch, lt(col("c0"), lit(5i32)), implementation).await?;
        let mut ctx = VortexSession::empty()
            .with::<ArraySession>()
            .create_execution_ctx();
        assert_arrays_eq!(expected, selected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[crate::test]
    async fn test_missing_and_unsupported_conjuncts_preserve_superset(
        #[values(SelectionImpl::Shared, SelectionImpl::SharedTyped)] implementation: SelectionImpl,
    ) -> VortexResult<()> {
        let batch = batch()?;
        let filter = and(
            lt(col("x"), lit(100i32)),
            or(lt(col("x"), lit(1i32)), gt(col("x"), lit(4998i32))),
        );
        let mut comparisons = field_comparisons(&filter, batch.dtype())?;
        assert_eq!(comparisons.len(), 1);
        let mut missing = comparisons[0].clone();
        missing.field = "not_projected".into();
        comparisons.insert(0, missing);
        let expected = batch.slice(0..100)?;
        let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
        let selected = select_batch(batch, &comparisons, implementation, &mut ctx).await?;
        let selected = selected
            .execute_cuda(&mut ctx)
            .await?
            .into_host()
            .await?
            .into_array();
        let mut ctx = VortexSession::empty()
            .with::<ArraySession>()
            .create_execution_ctx();
        assert_arrays_eq!(expected, selected, &mut ctx);
        Ok(())
    }
}
