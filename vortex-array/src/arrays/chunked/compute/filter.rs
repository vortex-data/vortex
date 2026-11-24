// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::filter::FilterKernel;

impl FilterKernel for Chunked {
    fn filter(
        array: ArrayView<'_, Chunked>,
        mask: &Mask,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mut chunks = Vec::with_capacity(array.nchunks().min(mask.true_count()));
        let mut indices = mask
            .values()
            .vortex_expect("FilterKernel receives a partially selective mask")
            .bit_buffer()
            .set_indices();
        let offsets = array.chunk_offset_values();
        let ends = &offsets[1..];
        let mut next_chunk = 0;
        while let Some(row) = indices.next() {
            let chunk_id = find_next_chunk(ends, next_chunk, row);
            let end = ends[chunk_id];
            let selected = mask.slice(offsets[chunk_id]..end);
            let chunk = array.chunk(chunk_id);
            chunks.push(if selected.all_true() {
                chunk.clone()
            } else {
                chunk.filter(selected)?
            });
            indices.seek_to(end);
            next_chunk = chunk_id + 1;
        }

        // SAFETY: Filtering preserves each chunk's dtype.
        Ok(Some(unsafe {
            ChunkedArray::new_unchecked(chunks, array.dtype().clone()).into_array()
        }))
    }
}

/// Finds the chunk containing `row` using sorted ends.
/// `start` must not pass that chunk, and `row` must precede the final end.
fn find_next_chunk(ends: &[usize], mut start: usize, row: usize) -> usize {
    let stop = (start + 8).min(ends.len());
    while start < stop {
        if ends[start] > row {
            return start;
        }
        start += 1;
    }
    let mut lower = start;
    let mut step = 1;
    while start + step < ends.len() && ends[start + step] <= row {
        lower = start + step + 1;
        step *= 2;
    }
    let upper = (start + step + 1).min(ends.len());
    lower + ends[lower..upper].partition_point(|&end| end <= row)
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::half::f16;

    #[test]
    fn filter_chunked_floats() {
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![f16::from_f32(0.1463623)].into_array(),
                buffer![
                    f16::NAN,
                    f16::from_f32(0.24987793),
                    f16::from_f32(0.22497559),
                    f16::from_f32(0.22497559),
                    f16::from_f32(-36160.0),
                ]
                .into_array(),
                buffer![
                    f16::NAN,
                    f16::NAN,
                    f16::from_f32(0.22497559),
                    f16::from_f32(0.22497559),
                    f16::from_f32(3174.0),
                ]
                .into_array(),
            ],
            DType::Primitive(PType::F16, Nullability::NonNullable),
        )
        .unwrap();
        let mask = Mask::from_iter([
            true, false, false, true, true, true, true, true, true, true, true,
        ]);
        let filtered = chunked.filter(mask).unwrap();
        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter([
                f16::from_f32(0.1463623),
                f16::from_f32(0.22497559),
                f16::from_f32(0.22497559),
                f16::from_f32(-36160.0),
                f16::NAN,
                f16::NAN,
                f16::from_f32(0.22497559),
                f16::from_f32(0.22497559),
                f16::from_f32(3174.0),
            ]),
            &mut SESSION.create_execution_ctx()
        );
    }

    use std::sync::LazyLock;

    use rstest::rstest;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
            PrimitiveArray::from_option_iter([Some(0u64), None]).into_array(),
            PrimitiveArray::from_option_iter([Some(2u64)]).into_array(),
            PrimitiveArray::empty::<u64>(Nullability::Nullable).into_array(),
            PrimitiveArray::from_option_iter([None, Some(4u64)]).into_array(),
        ],
        DType::Primitive(PType::U64, Nullability::Nullable),
    ).unwrap())]
    #[case(ChunkedArray::try_new(
        vec![
            buffer![1i32].into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ).unwrap())]
    #[case(ChunkedArray::try_new(
        (0..10).map(|i| buffer![i as i64, i as i64 + 10, i as i64 + 20].into_array()),
        DType::Primitive(PType::I64, Nullability::NonNullable),
    ).unwrap())]
    fn test_filter_chunked_conformance(#[case] chunked: ChunkedArray) {
        test_filter_conformance(&chunked.into_array(), &mut SESSION.create_execution_ctx());
    }
}
