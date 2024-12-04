use vortex_error::VortexResult;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::SumFn;
use crate::ArrayData;

impl SumFn<ChunkedArray> for ChunkedEncoding {
    fn sum(&self, _array: &ChunkedArray, _ends: &[u64]) -> VortexResult<ArrayData> {
        todo!()
    }
    // fn sum(&self, array: &ChunkedArray) -> VortexResult<Scalar> {
    //     if !(array.dtype().is_float() || array.dtype().is_int()) {
    //         vortex_bail!("cannot sum non-numeric array")
    //     }
    //     if array.len() != 1 {
    //         vortex_bail!("length must be one");
    //     }
    //     let inner = array.chunk(0)?;
    //     inner
    //         .encoding()
    //         .sum_fn()
    //         .ok_or_else(|| vortex_err!("chunked children must have sum"))?
    //         .sum(&inner)
    // }
}
