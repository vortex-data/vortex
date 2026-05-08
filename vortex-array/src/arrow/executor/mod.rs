// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod bool;
mod byte;
pub mod byte_view;
mod decimal;
mod dictionary;
mod fixed_size_list;
mod list;
mod list_view;
pub mod null;
pub mod primitive;
mod run_end;
mod struct_;
mod temporal;
mod validity;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::Schema;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::arrays::List;
use crate::arrays::VarBin;
use crate::arrays::list::ListArrayExt;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrow::executor::bool::to_arrow_bool;
use crate::arrow::executor::byte::to_arrow_byte_array;
use crate::arrow::executor::byte_view::to_arrow_byte_view;
use crate::arrow::executor::decimal::to_arrow_decimal;
use crate::arrow::executor::dictionary::to_arrow_dictionary;
use crate::arrow::executor::fixed_size_list::to_arrow_fixed_list;
use crate::arrow::executor::list::to_arrow_list;
use crate::arrow::executor::list_view::to_arrow_list_view;
use crate::arrow::executor::null::to_arrow_null;
use crate::arrow::executor::primitive::to_arrow_primitive;
use crate::arrow::executor::run_end::to_arrow_run_end;
use crate::arrow::executor::struct_::to_arrow_struct;
use crate::arrow::executor::temporal::to_arrow_date;
use crate::arrow::executor::temporal::to_arrow_time;
use crate::arrow::executor::temporal::to_arrow_timestamp;
use crate::arrow::session::ArrowSessionExt;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;

/// Trait for executing a Vortex array to produce an Arrow array.
pub trait ArrowArrayExecutor: Sized {
    /// Execute the array to produce an Arrow array.
    ///
    /// If a [`DataType`] is given, the array will be converted to the desired Arrow type.
    /// If `None`, the array's preferred (cheapest) Arrow type will be used.
    fn execute_arrow(
        self,
        data_type: Option<&DataType>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef>;

    /// Execute the array to produce an Arrow `RecordBatch` with the given schema.
    fn execute_record_batch(
        self,
        schema: &Schema,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RecordBatch> {
        let array = self.execute_arrow(Some(&DataType::Struct(schema.fields.clone())), ctx)?;
        Ok(RecordBatch::from(array.as_struct()))
    }

    /// Execute the array to produce Arrow `RecordBatch`'s with the given schema.
    fn execute_record_batches(
        self,
        schema: &Schema,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<RecordBatch>>;
}

impl ArrowArrayExecutor for ArrayRef {
    fn execute_arrow(
        self,
        data_type: Option<&DataType>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        // Clone the session out of `ctx` to break the immutable borrow chain that prevents
        // `ctx` from being passed back through to the session method.
        let session = ctx.session().clone();
        let target = match data_type {
            Some(dt) => Some(Field::new("", dt.clone(), self.dtype().is_nullable())),
            // No target supplied: if the source dtype tree contains any Vortex extension,
            // synthesize a Field via session-aware inference so registered plugins run and
            // ARROW:extension:name metadata is preserved end-to-end. For non-extension
            // trees we leave target as None so canonical preferred-type logic (e.g.
            // VarBin → Utf8 instead of Utf8View) keeps running.
            None if dtype_has_extension(self.dtype()) => {
                Some(session.arrow().to_arrow_field("", self.dtype())?)
            }
            None => None,
        };
        session.arrow().execute_arrow(self, target.as_ref(), ctx)
    }

    fn execute_record_batches(
        self,
        schema: &Schema,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<RecordBatch>> {
        self.to_array_iterator()
            .map(|a| a?.execute_record_batch(schema, ctx))
            .try_collect()
    }
}

/// Canonical Vortex → Arrow conversion, dispatched by Arrow [`DataType`].
///
/// This is the fallback path used by [`crate::arrow::ArrowSession::execute_arrow`] when no
/// extension plugin matches. Callers normally go through the session; this is `pub(crate)`
/// purely so the session can hand off after its own dispatch.
pub(crate) fn canonical_execute_arrow(
    array: ArrayRef,
    data_type: Option<&DataType>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();

    let resolved_type: DataType = match data_type {
        Some(dt) => dt.clone(),
        None => preferred_arrow_type(&array)?,
    };

    let arrow = match &resolved_type {
        DataType::Null => to_arrow_null(array, ctx),
        DataType::Boolean => to_arrow_bool(array, ctx),
        DataType::Int8 => to_arrow_primitive::<Int8Type>(array, ctx),
        DataType::Int16 => to_arrow_primitive::<Int16Type>(array, ctx),
        DataType::Int32 => to_arrow_primitive::<Int32Type>(array, ctx),
        DataType::Int64 => to_arrow_primitive::<Int64Type>(array, ctx),
        DataType::UInt8 => to_arrow_primitive::<UInt8Type>(array, ctx),
        DataType::UInt16 => to_arrow_primitive::<UInt16Type>(array, ctx),
        DataType::UInt32 => to_arrow_primitive::<UInt32Type>(array, ctx),
        DataType::UInt64 => to_arrow_primitive::<UInt64Type>(array, ctx),
        DataType::Float16 => to_arrow_primitive::<Float16Type>(array, ctx),
        DataType::Float32 => to_arrow_primitive::<Float32Type>(array, ctx),
        DataType::Float64 => to_arrow_primitive::<Float64Type>(array, ctx),
        DataType::Binary => to_arrow_byte_array::<BinaryType>(array, ctx),
        DataType::LargeBinary => to_arrow_byte_array::<LargeBinaryType>(array, ctx),
        DataType::Utf8 => to_arrow_byte_array::<Utf8Type>(array, ctx),
        DataType::LargeUtf8 => to_arrow_byte_array::<LargeUtf8Type>(array, ctx),
        DataType::BinaryView => to_arrow_byte_view::<BinaryViewType>(array, ctx),
        DataType::Utf8View => to_arrow_byte_view::<StringViewType>(array, ctx),
        // TODO(joe): pass down preferred
        DataType::List(elements_field) => to_arrow_list::<i32>(array, elements_field, ctx),
        // TODO(joe): pass down preferred
        DataType::LargeList(elements_field) => to_arrow_list::<i64>(array, elements_field, ctx),
        // TODO(joe): pass down preferred
        DataType::FixedSizeList(elements_field, list_size) => {
            to_arrow_fixed_list(array, *list_size, elements_field, ctx)
        }
        // TODO(joe): pass down preferred
        DataType::ListView(elements_field) => to_arrow_list_view::<i32>(array, elements_field, ctx),
        // TODO(joe): pass down preferred
        DataType::LargeListView(elements_field) => {
            to_arrow_list_view::<i64>(array, elements_field, ctx)
        }
        DataType::Struct(fields) => {
            let fields = if data_type.is_none() {
                None
            } else {
                Some(fields)
            };
            to_arrow_struct(array, fields, ctx)
        }
        // TODO(joe): pass down preferred
        DataType::Dictionary(codes_type, values_type) => {
            to_arrow_dictionary(array, codes_type, values_type, ctx)
        }
        dt @ DataType::Decimal32(..) => to_arrow_decimal(array, dt, ctx),
        dt @ DataType::Decimal64(..) => to_arrow_decimal(array, dt, ctx),
        dt @ DataType::Decimal128(..) => to_arrow_decimal(array, dt, ctx),
        dt @ DataType::Decimal256(..) => to_arrow_decimal(array, dt, ctx),
        // TODO(joe): pass down preferred
        DataType::RunEndEncoded(ends_type, values_type) => {
            to_arrow_run_end(array, ends_type.data_type(), values_type, ctx)
        }
        dt @ (DataType::Date32 | DataType::Date64) => to_arrow_date(array, dt, ctx),
        dt @ (DataType::Time32(_) | DataType::Time64(_)) => to_arrow_time(array, dt, ctx),
        dt @ DataType::Timestamp(..) => to_arrow_timestamp(array, dt, ctx),
        DataType::FixedSizeBinary(_)
        | DataType::Map(..)
        | DataType::Duration(_)
        | DataType::Interval(_)
        | DataType::Union(..) => {
            vortex_bail!("Conversion to Arrow type {resolved_type} is not supported");
        }
    }?;

    vortex_ensure!(
        arrow.len() == len,
        "Arrow array length does not match Vortex array length after conversion to {:?}",
        arrow
    );

    Ok(arrow)
}

/// Determine the preferred (cheapest) Arrow type for an array.
///
/// For most arrays, this returns the canonical Arrow type from `dtype.to_arrow_dtype()`.
/// However, some encodings have cheaper Arrow representations:
/// - `VarBinArray`: Uses `Utf8`/`Binary` (offset-based) instead of `Utf8View`/`BinaryView`
/// - `ListArray`: Uses `List` instead of `ListView`
fn preferred_arrow_type(array: &ArrayRef) -> VortexResult<DataType> {
    // VarBinArray: use offset-based Binary/Utf8 instead of View types
    if let Some(varbin) = array.as_opt::<VarBin>() {
        let offsets_ptype = PType::try_from(varbin.offsets().dtype())?;
        let use_large = matches!(offsets_ptype, PType::I64 | PType::U64);

        return Ok(match (varbin.dtype(), use_large) {
            (DType::Utf8(_), false) => DataType::Utf8,
            (DType::Utf8(_), true) => DataType::LargeUtf8,
            (DType::Binary(_), false) => DataType::Binary,
            (DType::Binary(_), true) => DataType::LargeBinary,
            _ => unreachable!("VarBinArray must have Utf8 or Binary dtype"),
        });
    }

    // ListArray: use List with appropriate offset size
    if let Some(list) = array.as_opt::<List>() {
        let offsets_ptype = PType::try_from(list.offsets().dtype())?;
        let use_large = matches!(offsets_ptype, PType::I64 | PType::U64);
        // Recursively get the preferred type for elements
        let elem_dtype = preferred_arrow_type(list.elements())?;
        let field = FieldRef::new(Field::new_list_field(
            elem_dtype,
            list.elements().dtype().is_nullable(),
        ));

        return Ok(if use_large {
            DataType::LargeList(field)
        } else {
            DataType::List(field)
        });
    }

    // Everything else: use canonical dtype conversion
    array.dtype().to_arrow_dtype()
}

/// Recursively check whether a dtype tree contains a [`DType::Extension`] node.
///
/// Used by the executor entry to decide whether to synthesize a session-aware target
/// [`Field`] (so plugins run + extension metadata survives) or to fall through to the
/// canonical `preferred_arrow_type` path.
fn dtype_has_extension(dtype: &DType) -> bool {
    match dtype {
        DType::Extension(_) => true,
        DType::List(elem, _) | DType::FixedSizeList(elem, ..) => dtype_has_extension(elem),
        DType::Struct(fields, _) => fields.fields().any(|f| dtype_has_extension(&f)),
        _ => false,
    }
}
