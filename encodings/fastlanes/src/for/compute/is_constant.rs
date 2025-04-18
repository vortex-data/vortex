use vortex_array::compute::{IsConstantFn, IsConstantOpts, is_constant_opts};
use vortex_error::VortexResult;

use crate::{FoRArray, FoREncoding};

impl IsConstantFn<&FoRArray> for FoREncoding {
    fn is_constant(&self, array: &FoRArray, opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        is_constant_opts(array.encoded(), opts).map(Some)
    }
}
