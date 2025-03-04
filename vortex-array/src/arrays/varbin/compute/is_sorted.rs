use crate::accessor::ArrayAccessor;
use crate::arrays::{VarBinArray, VarBinEncoding};
use crate::compute::{IsSortedFn, IteratorExt};

impl IsSortedFn<&VarBinArray> for VarBinEncoding {
    fn is_sorted(&self, array: &VarBinArray, strict: bool) -> vortex_error::VortexResult<bool> {
        array.with_iterator(|bytes_iter| bytes_iter.is_sorted_with_strictness(strict))
    }
}
