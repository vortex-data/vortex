use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::{VarBinViewArray, VarBinViewEncoding};
use crate::compute::{IsSortedFn, IsSortedIteratorExt};

impl IsSortedFn<&VarBinViewArray> for VarBinViewEncoding {
    fn is_sorted(&self, array: &VarBinViewArray) -> VortexResult<bool> {
        array.with_iterator(|bytes_iter| bytes_iter.is_sorted())
    }

    fn is_strict_sorted(&self, array: &VarBinViewArray) -> VortexResult<bool> {
        array.with_iterator(|bytes_iter| bytes_iter.is_strict_sorted())
    }
}
