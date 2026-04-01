// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::zip::ZipKernel;

// Push down the zip call to the chunks. Without this rule
// the default implementation canonicalises the chunked array
// then zips once.
impl ZipKernel for Chunked {
    fn zip(
        if_true: ArrayView<'_, Chunked>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<Chunked>() else {
            return Ok(None);
        };
        let dtype = if_true
            .dtype()
            .union_nullability(if_false.dtype().nullability());
        let mut out_chunks = Vec::with_capacity(if_true.nchunks() + if_false.nchunks());

        for pair in if_true.paired_chunks(&if_false) {
            let pair = pair?;
            let mask_slice = mask.slice(pair.pos)?;
            out_chunks.push(mask_slice.zip(pair.left, pair.right)?);
        }

        // SAFETY: chunks originate from zipping slices of inputs that share dtype/nullability.
        let chunked = unsafe { ChunkedArray::new_unchecked(out_chunks, dtype) };
        Ok(Some(chunked.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::ToCanonical;
    use crate::VortexSessionExecute;
    use crate::arrays::Chunked;
    use crate::arrays::ChunkedArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

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

        let zipped = &mask
            .into_array()
            .zip(if_true.into_array(), if_false.into_array())
            .unwrap();
        // One step of execution will push down the zip.
        let zipped = zipped
            .clone()
            .execute::<ArrayRef>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        let zipped = zipped
            .as_opt::<Chunked>()
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
