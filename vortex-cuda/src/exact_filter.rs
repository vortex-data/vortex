// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exact row selection for CUDA scans whose filter otherwise only prunes zones.
//!
//! [`scan_cuda_pruned`](crate::scan_cuda_pruned) skips zones a filter rules out, but returns kept
//! zones whole, so a consumer that filters afterwards copies and evaluates every row of a kept
//! zone. [`select_rows`] additionally evaluates the filter's comparisons of a field with a
//! literal on the GPU, and drops the failing rows from each decoded batch before export.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::DeviceRepr;
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
    let comparisons: Arc<[FieldComparison]> = comparisons.into();
    let batches = stream
        .map(move |batch| {
            let comparisons = Arc::clone(&comparisons);
            let session = session.clone();
            async move {
                let mut ctx = CudaSession::create_execution_ctx(&session)?;
                select_batch(batch?, &comparisons, &mut ctx).await
            }
        })
        .buffered(concurrency.max(1));
    Box::pin(ArrayStreamAdapter::new(dtype, batches))
}

async fn select_batch(
    batch: ArrayRef,
    comparisons: &[FieldComparison],
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
        let combine = words.is_some();
        let target = match &mut words {
            Some(words) => words,
            None => words.insert(ctx.device_alloc::<u32>(len.div_ceil(32))?),
        };
        launch_compare(values, len, comparison, target, combine, ctx).await?;
    }
    let Some(words) = words else {
        return whole(fields);
    };
    let selected = count_selected(&words, ctx).await?;
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
    let mut selected_fields = Vec::with_capacity(decoded.len());
    for field in decoded {
        selected_fields.push(select_field(field, &words, len, selected, ctx).await?);
    }
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
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::TemporalArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::expr::and;
    use vortex::array::expr::col;
    use vortex::array::expr::gt;
    use vortex::array::expr::gt_eq;
    use vortex::array::expr::lit;
    use vortex::array::expr::lt;
    use vortex::array::expr::or;
    use vortex::array::extension::datetime::TimeUnit;
    use vortex::array::session::ArraySession;
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
        let comparisons = field_comparisons(&filter, batch.dtype())?;
        let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
        let selected = select_batch(batch, &comparisons, &mut ctx).await?;
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
    #[crate::test]
    async fn dates_compare_on_storage() -> VortexResult<()> {
        let days = PrimitiveArray::from_iter(0i32..5000).into_array();
        let date = TemporalArray::new_date(days, TimeUnit::Days).into_array();
        let batch = StructArray::from_fields(&[("d", date)])?.into_array();
        let selected = select(batch, gt_eq(col("d"), lit(4500i32))).await?;
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
}
