use vortex_array::compute::{MinMaxFn, MinMaxResult, min_max, take};
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl MinMaxFn<&DictArray> for DictEncoding {
    fn min_max(&self, array: &DictArray) -> VortexResult<Option<MinMaxResult>> {
        min_max(&take(array.values(), array.codes())?)
    }
}
