// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod byte;
mod byte_view;
mod decimal;
mod dictionary;
mod fixed_size_list;
mod list;
mod list_view;
mod null;
mod primitive;
mod run_end;
mod struct_;
mod temporal;
mod validity;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_schema::DataType;
use arrow_schema::Schema;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
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

/// Trait for executing a Vortex array to produce an Arrow array.
pub trait ArrowArrayExecutor: Sized {
    /// Execute the array to produce an Arrow array.
    ///
    /// If a [`DataType`] is given, the array will be converted to the desired Arrow type.
    fn execute_arrow(
        self,
        data_type: &DataType,
        session: &VortexSession,
    ) -> VortexResult<ArrowArrayRef>;

    /// Execute the array to produce an Arrow `RecordBatch` with the given schema.
    fn execute_record_batch(
        self,
        schema: &Schema,
        session: &VortexSession,
    ) -> VortexResult<RecordBatch> {
        let array = self.execute_arrow(&DataType::Struct(schema.fields.clone()), session)?;
        Ok(RecordBatch::from(array.as_struct()))
    }
}

impl ArrowArrayExecutor for ArrayRef {
    fn execute_arrow(
        self,
        data_type: &DataType,
        session: &VortexSession,
    ) -> VortexResult<ArrowArrayRef> {
        match data_type {
            DataType::Null => to_arrow_null(self, session),
            DataType::Boolean => to_arrow_bool(self, session),
            DataType::Int8 => to_arrow_primitive::<Int8Type>(self, session),
            DataType::Int16 => to_arrow_primitive::<Int16Type>(self, session),
            DataType::Int32 => to_arrow_primitive::<Int32Type>(self, session),
            DataType::Int64 => to_arrow_primitive::<Int64Type>(self, session),
            DataType::UInt8 => to_arrow_primitive::<UInt8Type>(self, session),
            DataType::UInt16 => to_arrow_primitive::<UInt16Type>(self, session),
            DataType::UInt32 => to_arrow_primitive::<UInt32Type>(self, session),
            DataType::UInt64 => to_arrow_primitive::<UInt64Type>(self, session),
            DataType::Float16 => to_arrow_primitive::<Float16Type>(self, session),
            DataType::Float32 => to_arrow_primitive::<Float32Type>(self, session),
            DataType::Float64 => to_arrow_primitive::<Float64Type>(self, session),
            DataType::Timestamp(..)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_) => to_arrow_temporal(self, data_type, session),
            DataType::Binary => to_arrow_byte_array::<BinaryType>(self, session),
            DataType::LargeBinary => to_arrow_byte_array::<LargeBinaryType>(self, session),
            DataType::Utf8 => to_arrow_byte_array::<Utf8Type>(self, session),
            DataType::LargeUtf8 => to_arrow_byte_array::<LargeUtf8Type>(self, session),
            DataType::BinaryView => to_arrow_byte_view::<BinaryViewType>(self, session),
            DataType::Utf8View => to_arrow_byte_view::<StringViewType>(self, session),
            DataType::List(elements_field) => to_arrow_list::<i32>(self, elements_field, session),
            DataType::LargeList(elements_field) => {
                to_arrow_list::<i64>(self, elements_field, session)
            }
            DataType::FixedSizeList(elements_field, list_size) => {
                to_arrow_fixed_list(self, *list_size, elements_field, session)
            }
            DataType::ListView(elements_field) => {
                to_arrow_list_view::<i32>(self, elements_field, session)
            }
            DataType::LargeListView(elements_field) => {
                to_arrow_list_view::<i64>(self, elements_field, session)
            }
            DataType::Struct(fields) => to_arrow_struct(self, fields, session),
            DataType::Dictionary(codes_type, values_type) => {
                to_arrow_dictionary(self, codes_type, values_type, session)
            }
            DataType::Decimal32(p, s) => {
                to_arrow_decimal::<Decimal32Type, i32>(self, *p, *s, session)
            }
            DataType::Decimal64(p, s) => {
                to_arrow_decimal::<Decimal64Type, i64>(self, *p, *s, session)
            }
            DataType::Decimal128(p, s) => {
                to_arrow_decimal::<Decimal128Type, i128>(self, *p, *s, session)
            }
            DataType::Decimal256(p, s) => {
                to_arrow_decimal::<Decimal256Type, vortex_dtype::i256>(self, *p, *s, session)
            }
            DataType::RunEndEncoded(ends_type, values_type) => {
                to_arrow_run_end(self, ends_type.data_type(), values_type, session)
            }
            DataType::FixedSizeBinary(_)
            | DataType::Map(..)
            | DataType::Duration(_)
            | DataType::Interval(_)
            | DataType::Union(..) => {
                vortex_bail!("Conversion to Arrow type {data_type} is not supported");
            }
        }
    }
}
