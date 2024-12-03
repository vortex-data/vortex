use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::Scalar;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::SumFn;
use crate::{ArrayDType, ArrayLen};

impl SumFn<ChunkedArray> for ChunkedEncoding {
    fn sum(&self, array: &ChunkedArray) -> VortexResult<Scalar> {
        if !(array.dtype().is_float() || array.dtype().is_int()) {
            vortex_bail!("cannot sum non-numeric array")
        }
        assert_eq!(array.len(), 1);
        let inner = array.chunk(0)?;
        inner.encoding().sum_fn().expect("sum_fn").sum(&inner)
    }

    fn sum_sq(&self, array: &ChunkedArray) -> VortexResult<Scalar> {
        if !(array.dtype().is_float() || array.dtype().is_int()) {
            vortex_bail!("cannot sum non-numeric array")
        }
        assert_eq!(array.len(), 1);
        let inner = array.chunk(0)?;
        inner.encoding().sum_fn().expect("sum_fn").sum_sq(&inner)
    }
}
