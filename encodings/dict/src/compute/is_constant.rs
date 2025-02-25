use vortex_array::compute::{IsConstantFn, IsConstantOpts, is_constant_opts};
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl IsConstantFn<&DictArray> for DictEncoding {
    fn is_constant(&self, array: &DictArray, opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        is_constant_opts(array.codes(), opts).map(Some)
    }
}
