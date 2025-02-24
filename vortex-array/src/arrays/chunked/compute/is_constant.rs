use vortex_error::VortexResult;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{is_constant, IsConstantFn};

impl IsConstantFn<&ChunkedArray> for ChunkedEncoding {
    fn is_constant(&self, array: &ChunkedArray) -> VortexResult<bool> {
        Ok(array
            .chunks()
            .iter()
            .map(|a| is_constant(a.as_ref()))
            .collect::<VortexResult<Vec<_>>>()?
            .iter()
            .all(|v| *v))
    }
}
