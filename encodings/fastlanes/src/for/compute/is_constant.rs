use vortex_array::compute::{IsConstantFn, IsConstantOpts, is_constant};
use vortex_error::VortexResult;

use crate::{FoRArray, FoREncoding};

impl IsConstantFn<&FoRArray> for FoREncoding {
    fn is_constant(&self, array: &FoRArray, _opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        is_constant(array.encoded()).map(Some)
    }
}
