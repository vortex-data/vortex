use vortex_error::VortexResult;

use crate::Array;
use crate::arrays::{VarBinViewArray, VarBinViewEncoding, compute_min_max};
use crate::compute::{MinMaxFn, MinMaxResult};

impl MinMaxFn<&VarBinViewArray> for VarBinViewEncoding {
    fn min_max(&self, array: &VarBinViewArray) -> VortexResult<Option<MinMaxResult>> {
        compute_min_max(array, array.dtype())
    }
}
