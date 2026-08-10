// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(
    deprecated,
    reason = "This module defines and implements a deprecated trait `ArrowArrayExecutor`"
)]

pub mod bool;
mod byte;
pub mod byte_view;
mod decimal;
mod dictionary;
mod fixed_size_list;
mod list;
mod list_view;
mod map;
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
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::List;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::varbin::VarBinArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::executor::bool::to_arrow_bool;
use crate::executor::byte::to_arrow_byte_array;
use crate::executor::byte_view::to_arrow_byte_view;
use crate::executor::decimal::to_arrow_decimal;
use crate::executor::dictionary::to_arrow_dictionary;
use crate::executor::fixed_size_list::to_arrow_fixed_list;
use crate::executor::list::to_arrow_list;
use crate::executor::list_view::to_arrow_list_view;
use crate::executor::map::to_arrow_map;
use crate::executor::null::to_arrow_null;
use crate::executor::primitive::to_arrow_primitive;
use crate::executor::run_end::to_arrow_run_end;
use crate::executor::struct_::to_arrow_struct;
use crate::executor::temporal::to_arrow_date;
use crate::executor::temporal::to_arrow_time;
use crate::executor::temporal::to_arrow_timestamp;
use crate::session::ArrowSessionExt;

/// Trait for executing a Vortex array to produce an Arrow array.
#[deprecated(note = "Use an `ArrowSession` to perform conversions to/from Arrow arrays")]
pub trait ArrowArrayExecutor: Sized {
    /// Execute the array to produce an Arrow array.
    ///
    /// If a [`DataType`] is given, the array will be converted to the desired Arrow type.
    /// If `None`, the array's preferred (cheapest) Arrow type will be used.
    #[deprecated(note = "Use an `ArrowSession` to perform conversions to/from Arrow arrays")]
    fn execute_arrow(
        self,
        data_type: Option<&DataType>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef>;

    /// Execute the array to produce an Arrow `RecordBatch` with the given schema.
    #[deprecated(note = "Use an `ArrowSession` to perform conversions to/from Arrow arrays")]
    fn execute_record_batch(
        self,
        schema: &Schema,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RecordBatch> {
        let array = self.execute_arrow(Some(&DataType::Struct(schema.fields.clone())), ctx)?;
        Ok(RecordBatch::from(array.as_struct()))
    }

    /// Execute the array to produce Arrow `RecordBatch`'s with the given schema.
    #[deprecated(note = "Use an `ArrowSession` to perform conversions to/from Arrow arrays")]
    fn execute_record_batches(
        self,
        schema: &Schema,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<RecordBatch>>;
}

#[expect(deprecated, reason = "backward compatibility")]
impl ArrowArrayExecutor for ArrayRef {
    fn execute_arrow(
        self,
        data_type: Option<&DataType>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        let target = data_type.map(|dt| Field::new("", dt.clone(), self.dtype().is_nullable()));
        let session = ctx.session().clone();
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

/// Execute an arbitrary Vortex array into an Arrow array, dispatched by [`DataType`]. This pathway
/// is naive to any extension type information, and only seeks to map canonical Vortex arrays to
/// some target Arrow physical encoding.
///
/// Public callers should go through the `ArrowSession` instead.
pub(crate) fn execute_arrow_naive(
    array: ArrayRef,
    data_type: Option<&DataType>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();

    let resolved_type: DataType = match data_type {
        Some(dt) => dt.clone(),
        None => infer_nearest_arrow_type(&array, ctx)?,
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
        DataType::Map(entries_field, keys_sorted) => {
            to_arrow_map(array, entries_field, *keys_sorted, ctx)
        }
        DataType::FixedSizeBinary(_)
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
/// For most arrays, this returns the canonical Arrow type for the array's dtype.
/// However, some encodings have cheaper Arrow representations:
/// - `VarBinArray`: Uses `Utf8`/`Binary` (offset-based) instead of `Utf8View`/`BinaryView`
/// - `ListArray`: Uses `List` instead of `ListView`
fn infer_nearest_arrow_type(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DataType> {
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
        // Recursively get the preferred field for elements, so extension elements keep the
        // `ARROW:extension:name` metadata their export plugin assigns.
        let field = FieldRef::new(infer_nearest_arrow_field(
            list.elements(),
            Field::LIST_FIELD_DEFAULT_NAME,
            ctx,
        )?);

        return Ok(if use_large {
            DataType::LargeList(field)
        } else {
            DataType::List(field)
        });
    }

    // Everything else: defer to the session's canonical conversion, which additionally resolves
    // extension dtypes (including ones nested inside containers) through their export plugins.
    Ok(infer_nearest_arrow_field(array, "", ctx)?
        .data_type()
        .clone())
}

/// Determine the preferred (cheapest) Arrow [`Field`] for an array.
///
/// Unlike [`infer_nearest_arrow_type`] this preserves the Field-level `ARROW:extension:name`
/// metadata that export plugins assign, which a bare [`DataType`] cannot carry.
pub(crate) fn infer_nearest_arrow_field(
    array: &ArrayRef,
    name: &str,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Field> {
    // Only the encodings with a cheaper-than-canonical Arrow type need the inference; every other
    // dtype goes through the session, which is the only thing that knows how to map extensions.
    if array.is::<VarBin>() || array.is::<List>() {
        let data_type = infer_nearest_arrow_type(array, ctx)?;
        return Ok(Field::new(name, data_type, array.dtype().is_nullable()));
    }
    ctx.session()
        .clone()
        .arrow()
        .to_arrow_field(name, array.dtype())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::TimeUnit;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;

    use crate::ArrowSessionExt;

    fn utf8() -> ArrayRef {
        VarBinViewArray::from_iter_str(["a", "bb"]).into_array()
    }

    fn primitive() -> ArrayRef {
        PrimitiveArray::from_iter([1i32, 2]).into_array()
    }

    fn boolean() -> ArrayRef {
        BoolArray::from_iter([true, false]).into_array()
    }

    fn list_target() -> DataType {
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true)))
    }

    /// Must error rather than panic inside `execute::<T>`.
    #[rstest]
    #[case::bool_from_utf8(utf8(), DataType::Boolean)]
    #[case::bool_from_primitive(primitive(), DataType::Boolean)]
    #[case::null_from_utf8(utf8(), DataType::Null)]
    #[case::null_from_primitive(primitive(), DataType::Null)]
    #[case::decimal_from_utf8(utf8(), DataType::Decimal128(10, 2))]
    #[case::decimal_from_primitive(primitive(), DataType::Decimal128(10, 2))]
    #[case::list_from_utf8(utf8(), list_target())]
    #[case::list_from_primitive(primitive(), list_target())]
    #[case::list_view_from_primitive(
        primitive(),
        DataType::ListView(Arc::new(Field::new("item", DataType::Int32, true)))
    )]
    #[case::fixed_size_list_from_utf8(
        utf8(),
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2)
    )]
    #[case::byte_from_primitive(primitive(), DataType::Utf8)]
    #[case::byte_from_bool(boolean(), DataType::Binary)]
    fn incompatible_target_returns_error(#[case] array: ArrayRef, #[case] target: DataType) {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("f", target.clone(), array.dtype().is_nullable());

        let result = session.arrow().execute_arrow(array, Some(&field), &mut ctx);

        assert!(
            result.is_err(),
            "expected an error exporting to {target:?}, got Ok"
        );
    }

    /// Cross-class conversions that are genuinely supported must keep working.
    #[rstest]
    #[case::bool_to_int32(boolean(), DataType::Int32)]
    #[case::bool_to_float64(boolean(), DataType::Float64)]
    #[case::primitive_to_int64(primitive(), DataType::Int64)]
    #[case::primitive_to_date32(primitive(), DataType::Date32)]
    #[case::primitive_to_timestamp(primitive(), DataType::Timestamp(TimeUnit::Microsecond, None))]
    #[case::utf8_to_binary(utf8(), DataType::Binary)]
    #[case::utf8_to_large_utf8(utf8(), DataType::LargeUtf8)]
    fn supported_cross_class_target_still_works(#[case] array: ArrayRef, #[case] target: DataType) {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("f", target.clone(), array.dtype().is_nullable());

        let result = session.arrow().execute_arrow(array, Some(&field), &mut ctx);

        assert!(
            result.is_ok(),
            "expected {target:?} export to succeed, got {:?}",
            result.err()
        );
    }
}
