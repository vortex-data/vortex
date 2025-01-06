use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::{FilterMask, MaskFn};
use crate::variants::PrimitiveArrayTrait as _;
use crate::{ArrayData, IntoArrayData};

impl MaskFn<PrimitiveArray> for PrimitiveEncoding {
    fn mask(&self, array: &PrimitiveArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let validity = array.validity().mask(&mask)?;
        Ok(
            PrimitiveArray::from_byte_buffer(array.byte_buffer().clone(), array.ptype(), validity)
                .into_array(),
        )
    }
}
