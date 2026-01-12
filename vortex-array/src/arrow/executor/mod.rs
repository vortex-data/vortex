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
use arrow_schema::Schema;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::Array;
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
use crate::executor::ExecutionCtx;

/// Trait for executing a Vortex array to produce an Arrow array.
pub trait ArrowArrayExecutor: Sized {
    /// Execute the array to produce an Arrow array.
    ///
    /// If a [`DataType`] is given, the array will be converted to the desired Arrow type.
    fn execute_arrow(
        self,
        data_type: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef>;

    /// Execute the array to produce an Arrow `RecordBatch` with the given schema.
    fn execute_record_batch(
        self,
        schema: &Schema,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RecordBatch> {
        let array = self.execute_arrow(&DataType::Struct(schema.fields.clone()), ctx)?;
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
        data_type: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        let len = self.len();

        let arrow = match data_type {
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
            | DataType::Time64(_) => to_arrow_temporal(self, data_type, ctx),
            DataType::Binary => to_arrow_byte_array::<BinaryType>(self, ctx),
            DataType::LargeBinary => to_arrow_byte_array::<LargeBinaryType>(self, ctx),
            DataType::Utf8 => to_arrow_byte_array::<Utf8Type>(self, ctx),
            DataType::LargeUtf8 => to_arrow_byte_array::<LargeUtf8Type>(self, ctx),
            DataType::BinaryView => to_arrow_byte_view::<BinaryViewType>(self, ctx),
            DataType::Utf8View => to_arrow_byte_view::<StringViewType>(self, ctx),
            DataType::List(elements_field) => to_arrow_list::<i32>(self, elements_field, ctx),
            DataType::LargeList(elements_field) => to_arrow_list::<i64>(self, elements_field, ctx),
            DataType::FixedSizeList(elements_field, list_size) => {
                to_arrow_fixed_list(self, *list_size, elements_field, ctx)
            }
            DataType::ListView(elements_field) => {
                to_arrow_list_view::<i32>(self, elements_field, ctx)
            }
            DataType::LargeListView(elements_field) => {
                to_arrow_list_view::<i64>(self, elements_field, ctx)
            }
            DataType::Struct(fields) => to_arrow_struct(self, fields, ctx),
            DataType::Dictionary(codes_type, values_type) => {
                to_arrow_dictionary(self, codes_type, values_type, ctx)
            }
            DataType::Decimal32(p, s) => to_arrow_decimal::<Decimal32Type, i32>(self, *p, *s, ctx),
            DataType::Decimal64(p, s) => to_arrow_decimal::<Decimal64Type, i64>(self, *p, *s, ctx),
            DataType::Decimal128(p, s) => {
                to_arrow_decimal::<Decimal128Type, i128>(self, *p, *s, ctx)
            }
            DataType::Decimal256(p, s) => {
                to_arrow_decimal::<Decimal256Type, vortex_dtype::i256>(self, *p, *s, ctx)
            }
            DataType::RunEndEncoded(ends_type, values_type) => {
                to_arrow_run_end(self, ends_type.data_type(), values_type, ctx)
            }
            DataType::FixedSizeBinary(_)
            | DataType::Map(..)
            | DataType::Duration(_)
            | DataType::Interval(_)
            | DataType::Union(..) => {
                vortex_bail!("Conversion to Arrow type {data_type} is not supported");
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
