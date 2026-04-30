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
use crate::arrays::ExtensionArray;
use crate::arrays::List;
use crate::arrays::VarBin;
use crate::arrays::extension::ExtensionArrayExt;
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
use crate::arrow::executor::temporal::to_arrow_temporal;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::extension::datetime::AnyTemporal;

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
        let len = self.len();

        // Resolve the DataType if it is a leaf type
        // we should likely make this extensible.
        let resolved_type: DataType = match data_type {
            Some(dt) => dt.clone(),
            None => preferred_arrow_type(&self)?,
        };

        // Extensions with a native Arrow mapping (temporal) keep their wrapper so
        // `to_arrow_temporal` can read the metadata. Other extensions carry identity in
        // Field metadata, so dispatch on the storage array. Mirror the discriminator used
        // by `field_from_dtype` / `native_arrow_dtype_for_extension` in `dtype/arrow.rs`.
        if let DType::Extension(ext) = self.dtype()
            && ext.metadata_opt::<AnyTemporal>().is_none()
        {
            let ext = self.execute::<ExtensionArray>(ctx)?;
            return ext.storage_array().clone().execute_arrow(data_type, ctx);
        }

        let arrow = match &resolved_type {
            DataType::Null => to_arrow_null(self, ctx),
            DataType::Boolean => to_arrow_bool(self, ctx),
            DataType::Int8 => to_arrow_primitive::<Int8Type>(self, ctx),
            DataType::Int16 => to_arrow_primitive::<Int16Type>(self, ctx),
            DataType::Int32 => to_arrow_primitive::<Int32Type>(self, ctx),
            DataType::Int64 => to_arrow_primitive::<Int64Type>(self, ctx),
            DataType::UInt8 => to_arrow_primitive::<UInt8Type>(self, ctx),
            DataType::UInt16 => to_arrow_primitive::<UInt16Type>(self, ctx),
            DataType::UInt32 => to_arrow_primitive::<UInt32Type>(self, ctx),
            DataType::UInt64 => to_arrow_primitive::<UInt64Type>(self, ctx),
            DataType::Float16 => to_arrow_primitive::<Float16Type>(self, ctx),
            DataType::Float32 => to_arrow_primitive::<Float32Type>(self, ctx),
            DataType::Float64 => to_arrow_primitive::<Float64Type>(self, ctx),
            DataType::Timestamp(..)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_) => to_arrow_temporal(self, &resolved_type, ctx),
            DataType::Binary => to_arrow_byte_array::<BinaryType>(self, ctx),
            DataType::LargeBinary => to_arrow_byte_array::<LargeBinaryType>(self, ctx),
            DataType::Utf8 => to_arrow_byte_array::<Utf8Type>(self, ctx),
            DataType::LargeUtf8 => to_arrow_byte_array::<LargeUtf8Type>(self, ctx),
            DataType::BinaryView => to_arrow_byte_view::<BinaryViewType>(self, ctx),
            DataType::Utf8View => to_arrow_byte_view::<StringViewType>(self, ctx),
            // TODO(joe): pass down preferred
            DataType::List(elements_field) => to_arrow_list::<i32>(self, elements_field, ctx),
            // TODO(joe): pass down preferred
            DataType::LargeList(elements_field) => to_arrow_list::<i64>(self, elements_field, ctx),
            // TODO(joe): pass down preferred
            DataType::FixedSizeList(elements_field, list_size) => {
                to_arrow_fixed_list(self, *list_size, elements_field, ctx)
            }
            // TODO(joe): pass down preferred
            DataType::ListView(elements_field) => {
                to_arrow_list_view::<i32>(self, elements_field, ctx)
            }
            // TODO(joe): pass down preferred
            DataType::LargeListView(elements_field) => {
                to_arrow_list_view::<i64>(self, elements_field, ctx)
            }
            DataType::Struct(fields) => {
                let fields = if data_type.is_none() {
                    None
                } else {
                    Some(fields)
                };
                to_arrow_struct(self, fields, ctx)
            }
            // TODO(joe): pass down preferred
            DataType::Dictionary(codes_type, values_type) => {
                to_arrow_dictionary(self, codes_type, values_type, ctx)
            }
            dt @ DataType::Decimal32(..) => to_arrow_decimal(self, dt, ctx),
            dt @ DataType::Decimal64(..) => to_arrow_decimal(self, dt, ctx),
            dt @ DataType::Decimal128(..) => to_arrow_decimal(self, dt, ctx),
            dt @ DataType::Decimal256(..) => to_arrow_decimal(self, dt, ctx),
            // TODO(joe): pass down preferred
            DataType::RunEndEncoded(ends_type, values_type) => {
                to_arrow_run_end(self, ends_type.data_type(), values_type, ctx)
            }
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

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_array::types::UInt64Type;
    use arrow_schema::DataType;
    use arrow_schema::TimeUnit as ArrowTimeUnit;

    use super::*;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::array::IntoArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::Nullability;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::extension::tests::divisible_int::DivisibleInt;
    use crate::extension::tests::divisible_int::Divisor;

    #[test]
    fn execute_arrow_unwraps_extension_to_storage() {
        let storage = PrimitiveArray::from_iter(0u64..6).into_array();
        let ext = ExtensionArray::try_new_from_vtable(DivisibleInt, Divisor(1), storage)
            .unwrap()
            .into_array();

        let arrow = ext
            .execute_arrow(
                Some(&DataType::UInt64),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap();

        let primitives = arrow.as_primitive::<UInt64Type>();
        assert_eq!(primitives.values(), &[0, 1, 2, 3, 4, 5]);
    }

    /// Temporal extensions have native Arrow mappings. The executor must keep the extension
    /// array intact so `to_arrow_temporal` can read `TemporalMetadata` from its dtype.
    #[test]
    fn execute_arrow_keeps_temporal_extension_for_native_arrow_type() {
        use crate::dtype::DType;
        use crate::dtype::PType;

        let storage = PrimitiveArray::from_iter(0i64..3).into_array();
        let ts_ref =
            Timestamp::new_with_tz(TimeUnit::Microseconds, None, Nullability::NonNullable).erased();
        let storage_dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        assert_eq!(ts_ref.storage_dtype(), &storage_dtype);
        let ext = ExtensionArray::try_new(ts_ref, storage)
            .unwrap()
            .into_array();

        let arrow = ext
            .execute_arrow(
                Some(&DataType::Timestamp(ArrowTimeUnit::Microsecond, None)),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap();

        assert!(matches!(
            arrow.data_type(),
            DataType::Timestamp(ArrowTimeUnit::Microsecond, None)
        ));
    }
}
