use vortex_array::compute::{is_constant_opts, IsConstantFn, IsConstantOpts};
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl IsConstantFn<&DictArray> for DictEncoding {
    fn is_constant(&self, array: &DictArray, opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        is_constant_opts(array.codes(), opts).map(Some)
    }
}
