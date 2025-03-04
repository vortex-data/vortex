use vortex_array::compute::{IsSortedFn, is_sorted_opts};
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl IsSortedFn<&DictArray> for DictEncoding {
    fn is_sorted(&self, array: &DictArray, strict: bool) -> VortexResult<bool> {
        let is_sorted =
            is_sorted_opts(array.values(), strict)? && is_sorted_opts(array.codes(), strict)?;
        Ok(is_sorted)
    }
}
