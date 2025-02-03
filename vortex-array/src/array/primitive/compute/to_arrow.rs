use std::sync::Arc;

use arrow_array::types::{
    Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
    UInt32Type, UInt64Type, UInt8Type,
};
use arrow_array::{ArrayRef, ArrowPrimitiveType, PrimitiveArray as ArrowPrimitiveArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::DataType;
use vortex_dtype::PType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::ToArrowFn;
use crate::variants::PrimitiveArrayTrait;

impl ToArrowFn<PrimitiveArray> for PrimitiveEncoding {
    fn to_arrow(
        &self,
        primitive_array: &PrimitiveArray,
        data_type: &DataType,
    ) -> VortexResult<Option<ArrayRef>> {
        fn as_arrow_array_primitive<T: ArrowPrimitiveType>(
            array: &PrimitiveArray,
            data_type: &DataType,
        ) -> VortexResult<Option<ArrayRef>> {
            if data_type != &T::DATA_TYPE {
                vortex_bail!("Unsupported data type: {data_type}");
            }

            Ok(Some(Arc::new(ArrowPrimitiveArray::<T>::new(
                ScalarBuffer::<T::Native>::new(
                    array.byte_buffer().clone().into_arrow_buffer(),
                    0,
                    array.len(),
                ),
                array.validity_mask()?.to_null_buffer(),
            ))))
        }

        match primitive_array.ptype() {
            PType::U8 => as_arrow_array_primitive::<UInt8Type>(primitive_array, data_type),
            PType::U16 => as_arrow_array_primitive::<UInt16Type>(primitive_array, data_type),
            PType::U32 => as_arrow_array_primitive::<UInt32Type>(primitive_array, data_type),
            PType::U64 => as_arrow_array_primitive::<UInt64Type>(primitive_array, data_type),
            PType::I8 => as_arrow_array_primitive::<Int8Type>(primitive_array, data_type),
            PType::I16 => as_arrow_array_primitive::<Int16Type>(primitive_array, data_type),
            PType::I32 => as_arrow_array_primitive::<Int32Type>(primitive_array, data_type),
            PType::I64 => as_arrow_array_primitive::<Int64Type>(primitive_array, data_type),
            PType::F16 => as_arrow_array_primitive::<Float16Type>(primitive_array, data_type),
            PType::F32 => as_arrow_array_primitive::<Float32Type>(primitive_array, data_type),
            PType::F64 => as_arrow_array_primitive::<Float64Type>(primitive_array, data_type),
        }
    }
}
