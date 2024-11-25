use vortex_error::VortexResult;

use crate::array::varbin::VarBinArray;
use crate::array::VarBinEncoding;
use crate::compute::{slice, SliceFn};
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl SliceFn<VarBinArray> for VarBinEncoding {
    fn slice(&self, array: &VarBinArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        VarBinArray::try_new(
            slice(array.offsets(), start, stop + 1)?,
            array.bytes(),
            array.dtype().clone(),
            array.validity().slice(start, stop)?,
        )
        .map(|a| a.into_array())
    }
}
