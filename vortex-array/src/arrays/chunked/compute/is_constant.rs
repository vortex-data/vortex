use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{
    IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, is_constant_opts, scalar_at,
};
use crate::{Array, register_kernel};

impl IsConstantKernel for ChunkedEncoding {
    fn is_constant(
        &self,
        array: &ChunkedArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        let mut chunks = array.chunks().iter().skip_while(|c| c.is_empty());

        let first_chunk = chunks.next().vortex_expect("Must have at least one value");

        match is_constant_opts(first_chunk, opts)? {
            // Un-determined
            None => return Ok(None),
            Some(false) => return Ok(Some(false)),
            Some(true) => {}
        }

        let first_value = scalar_at(first_chunk, 0)?.into_nullable();

        for chunk in chunks {
            if chunk.is_empty() {
                continue;
            }

            match is_constant_opts(chunk, opts)? {
                // Un-determined
                None => return Ok(None),
                Some(false) => return Ok(Some(false)),
                Some(true) => {}
            }

            if first_value != scalar_at(chunk, 0)?.into_nullable() {
                return Ok(Some(false));
            }
        }

        Ok(Some(true))
    }
}

register_kernel!(IsConstantKernelAdapter(ChunkedEncoding).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::{Buffer, buffer};
    use vortex_dtype::{DType, Nullability, PType};

    use crate::arrays::ChunkedArray;
    use crate::{Array, IntoArray};

    #[test]
    fn empty_chunk_is_constant() {
        let chunked = ChunkedArray::try_new(
            vec![
                Buffer::<u8>::empty().into_array(),
                Buffer::<u8>::empty().into_array(),
                buffer![255u8, 255].into_array(),
                Buffer::<u8>::empty().into_array(),
                buffer![255u8, 255].into_array(),
            ],
            DType::Primitive(PType::U8, Nullability::NonNullable),
        )
        .unwrap()
        .into_array();

        assert!(chunked.statistics().compute_is_constant().unwrap());
    }
}
