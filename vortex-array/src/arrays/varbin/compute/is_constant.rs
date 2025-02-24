use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::varbin::stats::compute_is_constant;
use crate::arrays::{VarBinArray, VarBinEncoding};
use crate::compute::IsConstantFn;

impl IsConstantFn<&VarBinArray> for VarBinEncoding {
    fn is_constant(&self, array: &VarBinArray) -> VortexResult<Option<bool>> {
        array.with_iterator(compute_is_constant).map(Some)
    }
}
