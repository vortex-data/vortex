use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::{fill_null, FillNullFn};
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl FillNullFn<ChunkedArray> for ChunkedEncoding {
    fn fill_null(&self, array: &ChunkedArray, fill_value: &Scalar) -> VortexResult<ArrayData> {
        ChunkedArray::try_new(
            array
                .chunks()
                .map(|c| fill_null(c, fill_value))
                .collect::<VortexResult<Vec<_>>>()?,
            array.dtype().clone(),
        )
        .map(|a| a.into_array())
    }
}
