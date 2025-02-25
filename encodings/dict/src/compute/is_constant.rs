use vortex_array::compute::IsConstantFn;
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl IsConstantFn<&DictArray> for DictEncoding {
    fn is_constant(&self, array: &DictArray) -> VortexResult<Option<bool>> {
        Ok(Some(array.codes().is_constant()))
    }
}
