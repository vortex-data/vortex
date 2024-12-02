use vortex_array::compute::{like, LikeFn};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl LikeFn<DictArray> for DictEncoding {
    fn like(&self, array: &DictArray, pattern: &ArrayData) -> VortexResult<ArrayData> {
        let values = like(&array.values(), pattern)?;
        Ok(DictArray::try_new(array.codes(), values)?.into_array())
    }
}
