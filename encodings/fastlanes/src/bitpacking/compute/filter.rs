use vortex_array::compute::{FilterFn, FilterMask};
use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::BitPackedArray;

impl FilterFn for BitPackedArray {
    fn filter(&self, _mask: &FilterMask) -> VortexResult<ArrayData> {
        todo!()
    }
}
