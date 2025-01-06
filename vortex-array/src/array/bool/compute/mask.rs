use vortex_error::VortexResult;

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::{FilterMask, MaskFn};
use crate::{ArrayData, IntoArrayData};

impl MaskFn<BoolArray> for BoolEncoding {
    fn mask(&self, array: &BoolArray, mask: FilterMask) -> VortexResult<ArrayData> {
        BoolArray::try_new(array.boolean_buffer(), array.validity().mask(&mask)?)
            .map(IntoArrayData::into_array)
    }
}
