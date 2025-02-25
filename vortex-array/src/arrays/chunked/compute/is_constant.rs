use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{scalar_at, IsConstantFn};

impl IsConstantFn<&ChunkedArray> for ChunkedEncoding {
    fn is_constant(&self, array: &ChunkedArray) -> VortexResult<Option<bool>> {
        let mut chunks = array.chunks().iter();

        let first_chunk = chunks.next().vortex_expect("Must have at least one value");

        if !first_chunk.is_constant() {
            return Ok(Some(false));
        }

        let first_value = scalar_at(first_chunk, 0)?.into_nullable();

        for chunk in chunks {
            if !chunk.is_constant() {
                return Ok(Some(false));
            }

            if first_value != scalar_at(chunk, 0)?.into_nullable() {
                return Ok(Some(false));
            }
        }

        Ok(Some(true))
    }
}
