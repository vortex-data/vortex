use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray as ArrowBoolArray};
use arrow_schema::DataType;
use vortex_error::{vortex_bail, VortexResult};

use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::ToArrowFn;

impl ToArrowFn<BoolArray> for BoolEncoding {
    fn to_arrow(&self, array: &BoolArray, data_type: &DataType) -> VortexResult<Option<ArrayRef>> {
        if data_type != &DataType::Boolean {
            vortex_bail!("Unsupported data type: {data_type}");
        }
        Ok(Some(Arc::new(ArrowBoolArray::new(
            array.boolean_buffer(),
            array.validity_mask()?.to_null_buffer(),
        ))))
    }
}
