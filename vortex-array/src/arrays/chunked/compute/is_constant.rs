// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DynArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::compute::IsConstantKernel;
use crate::compute::IsConstantKernelAdapter;
use crate::compute::IsConstantOpts;
use crate::compute::is_constant_opts;
use crate::register_kernel;

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

        let first_value = first_chunk.scalar_at(0)?.into_nullable();

        for chunk in chunks {
            match is_constant_opts(chunk, opts)? {
                // Un-determined
                None => return Ok(None),
                Some(false) => return Ok(Some(false)),
                Some(true) => {}
            }

            if first_value != chunk.scalar_at(0)?.into_nullable() {
                return Ok(Some(false));
            }
        }

        Ok(Some(true))
    }
}

register_kernel!(IsConstantKernelAdapter(ChunkedVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use crate::DynArray;
    use crate::IntoArray;
    use crate::arrays::ChunkedArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

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
