// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::fns::mask::Mask as MaskExpr;
use crate::scalar_fn::fns::mask::MaskKernel;

impl MaskKernel for Chunked {
    fn mask(
        array: &ChunkedArray,
        mask: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let chunk_offsets = array.chunk_offsets();
        let new_chunks: Vec<ArrayRef> = array
            .iter_chunks()
            .enumerate()
            .map(|(i, chunk)| {
                let start: usize = chunk_offsets[i].try_into()?;
                let end: usize = chunk_offsets[i + 1].try_into()?;
                let chunk_mask = mask.slice(start..end)?;
                MaskExpr.try_new_array(chunk.len(), EmptyOptions, [chunk.clone(), chunk_mask])
            })
            .collect::<VortexResult<_>>()?;

        Ok(Some(
            ChunkedArray::try_new(new_chunks, array.dtype().as_nullable())?.into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    #[rstest]
    #[case(ChunkedArray::try_new(
        vec![
            buffer![0u64, 1].into_array(),
            buffer![2_u64].into_array(),
            PrimitiveArray::empty::<u64>(Nullability::NonNullable).into_array(),
            buffer![3_u64, 4].into_array(),
        ],
        DType::Primitive(PType::U64, Nullability::NonNullable),
    ).unwrap())]
    #[case(ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
            PrimitiveArray::from_option_iter([Some(4i32), Some(5)]).into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::Nullable),
    ).unwrap())]
    #[case(ChunkedArray::try_new(
        vec![
            buffer![42u8].into_array(),
        ],
        DType::Primitive(PType::U8, Nullability::NonNullable),
    ).unwrap())]
    #[case(ChunkedArray::try_new(
        (0..20).map(|i| buffer![i as f32, i as f32 + 0.5].into_array()).collect(),
        DType::Primitive(PType::F32, Nullability::NonNullable),
    ).unwrap())]
    fn test_mask_chunked_conformance(#[case] chunked: ChunkedArray) {
        test_mask_conformance(&chunked.into_array());
    }
}
