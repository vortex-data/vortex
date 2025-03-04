use crate::accessor::ArrayAccessor;
use crate::arrays::{VarBinViewArray, VarBinViewEncoding};
use crate::compute::{IsSortedFn, IteratorExt};

impl IsSortedFn<&VarBinViewArray> for VarBinViewEncoding {
    fn is_sorted(&self, array: &VarBinViewArray, strict: bool) -> vortex_error::VortexResult<bool> {
        array.with_iterator(|bytes_iter| bytes_iter.is_sorted_with_strictness(strict))
    }
}
