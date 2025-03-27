use vortex_array::compute::OptimizeFn;
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl OptimizeFn<&DictArray> for DictEncoding {
    fn optimize(&self, array: &DictArray) -> VortexResult<ArrayRef> {
        array.compact().map(|a| a.into_array())
    }
}
