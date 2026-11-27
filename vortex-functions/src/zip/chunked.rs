// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_error::VortexResult;

use crate::builtins::ArrayFunctions;
use crate::zip::ZipKernel;
use crate::zip::paired_chunks::PairedChunksExt;
use crate::zip::zip_return_dtype;

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
        let dtype = zip_return_dtype(if_true.dtype(), if_false.dtype())?;
        let chunks = if_true.paired_chunks(&if_false).map(|pair| {
            let pair = pair?;
            mask.slice(pair.pos)?.zip(pair.left, pair.right)
        });
        chunks.process_results(|chunks| {
            // SAFETY: each zipped chunk has dtype
            // `zip_return_dtype(if_true.dtype(), if_false.dtype())`.
            Some(unsafe {
                ChunkedArray::new_unchecked_sized(
                    chunks,
                    dtype,
                    if_true.nchunks() + if_false.nchunks(),
                )
                .into_array()
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Chunked;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::chunked::ChunkedArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::builtins::ArrayFunctions;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();

        vortex_array::initialize(&session);

        crate::initialize(&session);

        session
    });

    #[test]
    fn test_chunked_zip_aligns_across_boundaries() {
        let mut ctx = SESSION.create_execution_ctx();
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
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
            .unwrap();
        let zipped = zipped
            .as_opt::<Chunked>()
            .expect("zip should keep chunked encoding");

        assert_eq!(zipped.nchunks(), 4);
        let mut values: Vec<i32> = Vec::new();
        for chunk in zipped.chunks() {
            let primitive = chunk.execute::<PrimitiveArray>(&mut ctx).unwrap();
            values.extend_from_slice(primitive.as_slice::<i32>());
        }
        assert_eq!(values, vec![1, 11, 3, 13, 5]);
    }

    #[test]
    fn test_chunked_zip_unions_nested_list_nullability() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mask = Mask::from_iter([true, false, true]);

        let non_null_list = ChunkedArray::try_new(
            vec![
                ListViewArray::new(
                    buffer![1i32, 2].into_array(),
                    buffer![0u32, 1].into_array(),
                    buffer![1u32, 1].into_array(),
                    Validity::NonNullable,
                    &mut ctx,
                )
                .into_array(),
                ListViewArray::new(
                    buffer![3i32].into_array(),
                    buffer![0u32].into_array(),
                    buffer![1u32].into_array(),
                    Validity::NonNullable,
                    &mut ctx,
                )
                .into_array(),
            ],
            DType::List(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                Nullability::NonNullable,
            ),
        )?
        .into_array();
        let nullable_list = ChunkedArray::try_new(
            vec![
                ListViewArray::new(
                    PrimitiveArray::from_option_iter([Some(10i32)]).into_array(),
                    buffer![0u32].into_array(),
                    buffer![1u32].into_array(),
                    Validity::NonNullable,
                    &mut ctx,
                )
                .into_array(),
                ListViewArray::new(
                    PrimitiveArray::from_option_iter([None::<i32>, Some(30)]).into_array(),
                    buffer![0u32, 1].into_array(),
                    buffer![1u32, 1].into_array(),
                    Validity::NonNullable,
                    &mut ctx,
                )
                .into_array(),
            ],
            DType::List(
                Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
                Nullability::NonNullable,
            ),
        )?
        .into_array();

        let zipped = mask
            .into_array()
            .zip(non_null_list, nullable_list)?
            .execute::<ArrayRef>(&mut ctx)?;
        let chunked = zipped
            .as_opt::<Chunked>()
            .expect("zip should keep chunked encoding");
        let expected_list = ListViewArray::new(
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
            buffer![0u32, 1, 2].into_array(),
            buffer![1u32, 1, 1].into_array(),
            Validity::NonNullable,
            &mut ctx,
        );
        assert!(
            chunked
                .iter_chunks()
                .all(|chunk| chunk.dtype() == expected_list.dtype())
        );
        assert_arrays_eq!(zipped, expected_list, &mut ctx);
        Ok(())
    }
}
