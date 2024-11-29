use vortex_error::VortexResult;

use crate::array::{SparseArray, SparseEncoding};
use crate::compute::{invert, InvertFn};
use crate::{ArrayData, ArrayLen, IntoArrayData};

impl InvertFn<SparseArray> for SparseEncoding {
    fn invert(&self, array: &SparseArray) -> VortexResult<ArrayData> {
        let inverted_fill = array.fill_scalar().as_bool().invert().into_scalar();
        SparseArray::try_new(
            array.indices(),
            invert(&array.values())?,
            array.len(),
            inverted_fill,
        )
        .map(|a| a.into_array())
    }
}
