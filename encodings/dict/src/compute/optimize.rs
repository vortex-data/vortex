use vortex_array::compute::{OptimizeFn, take};
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexResult;

use crate::builders::dict_encode;
use crate::{DictArray, DictEncoding};

impl OptimizeFn<&DictArray> for DictEncoding {
    fn optimize(&self, array: &DictArray) -> VortexResult<ArrayRef> {
        let visible_values = take(array.values(), array.codes())?.to_canonical()?;
        Ok(dict_encode(visible_values.as_ref())?.into_array())
    }
}
