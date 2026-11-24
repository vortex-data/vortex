// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_error::VortexResult;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPArraySlotsExt;

impl TakeExecute for ALP {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let taken_encoded = array.encoded().take(indices.clone())?;
        let taken_patches = array
            .patches()
            .map(|p| p.take(indices, ctx))
            .transpose()?
            .flatten();
        Ok(Some(
            ALP::try_new(taken_encoded, array.exponents(), taken_patches)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::alp_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case(buffer![1.23f32, 4.56, 7.89, 10.11, 12.13].into_array())]
    #[case(buffer![100.1f64, 200.2, 300.3, 400.4, 500.5].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]).into_array())]
    #[case(buffer![42.42f64].into_array())]
    fn test_take_alp_conformance(#[case] array: vortex_array::ArrayRef) {
        let mut ctx = SESSION.create_execution_ctx();
        let array_primitive = array.execute::<PrimitiveArray>(&mut ctx).unwrap();
        let alp = alp_encode(array_primitive.as_view(), None, &mut ctx).unwrap();
        test_take_conformance(&alp.into_array(), &mut ctx);
    }

    /// A take of rows of a chunked fixed-size list of ALP elements with patches gives the
    /// rows of the decoded array, for long runs that read little of each chunk and for short
    /// runs.
    #[rstest]
    #[case(768, &[3u32, 200, 41, 170, 9])]
    #[case(64, &[1000u32, 3, 2000, 512, 77, 1500, 9, 30])]
    fn test_take_chunked_fsl_of_alp_with_patches(#[case] list_size: u32, #[case] rows: &[u32]) {
        use vortex_array::arrays::ChunkedArray;
        use vortex_array::arrays::FixedSizeListArray;
        use vortex_array::assert_arrays_eq;
        use vortex_array::validity::Validity;
        use vortex_buffer::Buffer;

        use crate::ALPArrayExt;

        let mut ctx = SESSION.create_execution_ctx();
        let chunk_rows = 16_384 / list_size;
        let chunk_len = chunk_rows * list_size;

        let chunks = (0..4u32)
            .map(|chunk| {
                // About one value in eleven cannot be encoded, so it becomes a patch.
                let values: Buffer<f32> = (0..chunk_len)
                    .map(|i| {
                        let i = i + chunk * chunk_len;
                        if i.is_multiple_of(11) {
                            f32::from(u16::try_from(i % 60_000).unwrap()) * 1.0e3 + 0.123_457
                        } else {
                            f32::from(u16::try_from(i % 1000).unwrap()) / 100.0
                        }
                    })
                    .collect();
                let alp = alp_encode(
                    PrimitiveArray::new(values, Validity::NonNullable).as_view(),
                    None,
                    &mut ctx,
                )
                .unwrap();
                assert!(alp.patches().is_some());

                FixedSizeListArray::new(
                    alp.into_array(),
                    list_size,
                    Validity::NonNullable,
                    usize::try_from(chunk_rows).unwrap(),
                )
                .into_array()
            })
            .collect::<Vec<_>>();
        let dtype = chunks[0].dtype().clone();
        let chunked = ChunkedArray::try_new(chunks, dtype).unwrap().into_array();
        let decoded = chunked
            .clone()
            .execute::<FixedSizeListArray>(&mut ctx)
            .unwrap()
            .into_array();

        let rows = rows
            .iter()
            .map(|&row| row % (4 * chunk_rows))
            .collect::<Buffer<u32>>()
            .into_array();
        let expected = decoded.take(rows.clone()).unwrap();

        assert_arrays_eq!(chunked.take(rows).unwrap(), expected, &mut ctx);
    }
}
