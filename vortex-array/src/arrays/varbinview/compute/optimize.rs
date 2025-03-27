use vortex_error::VortexResult;

use crate::arrays::{VarBinViewArray, VarBinViewEncoding};
use crate::compute::OptimizeFn;
use crate::{Array, ArrayRef};

impl OptimizeFn<&VarBinViewArray> for VarBinViewEncoding {
    fn optimize(&self, array: &VarBinViewArray) -> VortexResult<ArrayRef> {
        array.compact().map(|v| v.into_array())
    }
}
