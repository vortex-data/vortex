// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{ChunkedArray, ChunkedVTable};
use crate::compute::{IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, is_constant_opts};
use crate::{Array, register_kernel};

impl IsConstantKernel for ChunkedVTable {
    fn is_constant(
        &self,
        array: &ChunkedArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        let mut chunks = array.non_empty_chunks();

        let first_chunk = chunks
            .next()
            .vortex_expect("Must have at least one non-empty chunk");

        match is_constant_opts(first_chunk, opts)? {
            // Un-determined
            None => return Ok(None),
            Some(false) => return Ok(Some(false)),
            Some(true) => {}
        }

        let first_value = first_chunk.scalar_at(0).into_nullable();

        for chunk in chunks {
            match is_constant_opts(chunk, opts)? {
                // Un-determined
                None => return Ok(None),
                Some(false) => return Ok(Some(false)),
                Some(true) => {}
            }

            if first_value != chunk.scalar_at(0).into_nullable() {
                return Ok(Some(false));
            }
        }

        Ok(Some(true))
    }
}

register_kernel!(IsConstantKernelAdapter(ChunkedVTable).lift());

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
