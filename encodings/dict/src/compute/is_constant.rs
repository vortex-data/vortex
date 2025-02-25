use vortex_array::compute::{IsConstantFn, is_constant};
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl IsConstantFn<&DictArray> for DictEncoding {
    fn is_constant(&self, array: &DictArray) -> VortexResult<Option<bool>> {
        is_constant(array.codes()).map(Some)
    }
}
