// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::compute::ZipKernel;
use crate::compute::ZipKernelAdapter;
use crate::compute::zip;
use crate::register_kernel;

// Push down the zip call to the chunks. Without this kernel
// the default implementation canonicalises the chunked array
// then zips once.
impl ZipKernel for ChunkedVTable {
    fn zip(
        &self,
        if_true: &ChunkedArray,
        if_false: &dyn Array,
        mask: &Mask,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<ChunkedVTable>() else {
            return Ok(None);
        };
        let dtype = if_true
            .dtype()
            .union_nullability(if_false.dtype().nullability());
        let mut out_chunks = Vec::with_capacity(if_true.nchunks() + if_false.nchunks());

        let mut lhs_idx = 0;
        let mut rhs_idx = 0;
        let mut lhs_offset = 0;
        let mut rhs_offset = 0;
        let mut pos = 0;
        let total_len = if_true.len();

        while pos < total_len {
            let lhs_chunk = if_true.chunk(lhs_idx);
            let rhs_chunk = if_false.chunk(rhs_idx);

            let lhs_rem = lhs_chunk.len() - lhs_offset;
            let rhs_rem = rhs_chunk.len() - rhs_offset;
            let take_until = lhs_rem.min(rhs_rem);

            let mask_slice = mask.slice(pos..pos + take_until);
            let lhs_slice = lhs_chunk.slice(lhs_offset..lhs_offset + take_until)?;
            let rhs_slice = rhs_chunk.slice(rhs_offset..rhs_offset + take_until)?;

            out_chunks.push(zip(lhs_slice.as_ref(), rhs_slice.as_ref(), &mask_slice)?);

            pos += take_until;
            lhs_offset += take_until;
            rhs_offset += take_until;

            if lhs_offset == lhs_chunk.len() {
                lhs_idx += 1;
                lhs_offset = 0;
            }
            if rhs_offset == rhs_chunk.len() {
                rhs_idx += 1;
                rhs_offset = 0;
            }
        }

        // SAFETY: chunks originate from zipping slices of inputs that share dtype/nullability.
        let chunked = unsafe { ChunkedArray::new_unchecked(out_chunks, dtype) };
        Ok(Some(chunked.to_array()))
    }
}

register_kernel!(ZipKernelAdapter(ChunkedVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ChunkedVTable;
    use crate::compute::zip;

    #[test]
    fn test_chunked_zip_aligns_across_boundaries() {
        let if_true = ChunkedArray::try_new(
            vec![
                buffer![1i32, 2].into_array(),
                buffer![3i32].into_array(),
                buffer![4i32, 5].into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .unwrap();

        let if_false = ChunkedArray::try_new(
            vec![
                buffer![10i32].into_array(),
                buffer![11i32, 12].into_array(),
                buffer![13i32, 14].into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .unwrap();

        let mask = Mask::from_iter([true, false, true, false, true]);

        let zipped = zip(if_true.as_ref(), if_false.as_ref(), &mask).unwrap();
        let zipped = zipped
            .as_opt::<ChunkedVTable>()
            .expect("zip should keep chunked encoding");

        assert_eq!(zipped.nchunks(), 4);
        let mut values: Vec<i32> = Vec::new();
        for chunk in zipped.chunks() {
            let primitive = chunk.to_primitive();
            values.extend_from_slice(primitive.as_slice::<i32>());
        }
        assert_eq!(values, vec![1, 11, 3, 13, 5]);
    }
}
