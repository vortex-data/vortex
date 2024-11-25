use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::SliceFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, IntoArrayData};

impl SliceFn<PrimitiveArray> for PrimitiveEncoding {
    fn slice(&self, array: &PrimitiveArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let byte_width = array.ptype().byte_width();
        let buffer = array.buffer().slice(start * byte_width..stop * byte_width);
        Ok(
            PrimitiveArray::new(buffer, array.ptype(), array.validity().slice(start, stop)?)
                .into_array(),
        )
    }
}
