use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{IsConstantFn, IsConstantOpts, is_constant_opts, scalar_at};

impl IsConstantFn<&ChunkedArray> for ChunkedEncoding {
    fn is_constant(
        &self,
        array: &ChunkedArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        let mut chunks = array.chunks().iter();

        let first_chunk = chunks.next().vortex_expect("Must have at least one value");

        if !is_constant_opts(first_chunk, opts)? {
            return Ok(Some(false));
        }

        let first_value = scalar_at(first_chunk, 0)?.into_nullable();

        for chunk in chunks {
            if !is_constant_opts(chunk, opts)? {
                return Ok(Some(false));
            }

            if first_value != scalar_at(chunk, 0)?.into_nullable() {
                return Ok(Some(false));
            }
        }

        Ok(Some(true))
    }
}
