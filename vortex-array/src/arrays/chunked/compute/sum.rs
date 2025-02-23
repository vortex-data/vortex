use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::SumFn;

impl SumFn<&ChunkedArray> for ChunkedEncoding {
    fn sum(&self, array: &ChunkedArray) -> VortexResult<Scalar> {}
}
