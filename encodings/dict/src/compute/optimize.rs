use vortex_array::compute::OptimizeFn;
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexResult;

use crate::builders::dict_encode;
use crate::{DictArray, DictEncoding};

impl OptimizeFn<&DictArray> for DictEncoding {
    fn optimize(&self, array: &DictArray) -> VortexResult<ArrayRef> {
        Ok(dict_encode(array.to_canonical()?.as_ref())?.into_array())
    }
}
