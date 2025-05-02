use vortex_array::compute::{IsSortedKernel, is_sorted, is_strict_sorted};
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl IsSortedKernel<&DictArray> for DictEncoding {
    fn is_sorted(&self, array: &DictArray) -> VortexResult<bool> {
        let is_sorted = is_sorted(array.values())? && is_sorted(array.codes())?;
        Ok(is_sorted)
    }

    fn is_strict_sorted(&self, array: &DictArray) -> VortexResult<bool> {
        let is_sorted = is_strict_sorted(array.values())? && is_strict_sorted(array.codes())?;
        Ok(is_sorted)
    }
}
