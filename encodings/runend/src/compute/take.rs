// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::match_each_integer_ptype;
use vortex_array::search_sorted::SearchResult;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::RunEnd;
use crate::array::RunEndArrayExt;

impl TakeExecute for RunEnd {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "index cast to usize inside macro"
    )]
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Fall back to the canonical-then-gather path when there are many more indices
        // than the RunEnd array's logical length. In that regime, canonicalizing the
        // (small) RunEnd values once and doing an AVX2 gather is dramatically cheaper
        // than `indices.len()` independent binary searches into the ends array. The
        // pathological case is `Dict<RunEnd>` with a small RunEnd dictionary and many
        // codes: each code triggers a `search_sorted` here, blowing up to N * log K
        // work plus an O(N) `Vec<u64>` allocation, when a single decode of the RunEnd
        // (size = array.len()) followed by an AVX2 take would do.
        if indices.len() > array.len() {
            return Ok(None);
        }

        let primitive_indices = indices.clone().execute::<PrimitiveArray>(ctx)?;

        let checked_indices = match_each_integer_ptype!(primitive_indices.ptype(), |P| {
            primitive_indices
                .as_slice::<P>()
                .iter()
                .copied()
                .map(|idx| {
                    let usize_idx = idx as usize;
                    if usize_idx >= array.len() {
                        vortex_bail!(OutOfBounds: usize_idx, 0, array.len());
                    }
                    Ok(usize_idx)
                })
                .collect::<VortexResult<Vec<_>>>()?
        });

        let indices_validity = primitive_indices.validity()?;
        take_indices_unchecked(array, &checked_indices, &indices_validity, ctx).map(Some)
    }
}

/// Perform a take operation on a RunEndArray by binary searching for each of the indices.
pub fn take_indices_unchecked<T: AsPrimitive<usize>>(
    array: ArrayView<'_, RunEnd>,
    indices: &[T],
    validity: &Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
    let ends_len = ends.len();

    // TODO(joe): use the validity mask to skip search sorted.
    let physical_indices = match_each_integer_ptype!(ends.ptype(), |I| {
        let end_slices = ends.as_slice::<I>();
        let physical_indices_vec: Vec<u64> = indices
            .iter()
            .map(|idx| idx.as_() + array.offset())
            .map(|idx| {
                match <I as NumCast>::from(idx) {
                    Some(idx) => end_slices.search_sorted(&idx, SearchSortedSide::Right),
                    None => {
                        // The idx is too large for I, therefore it's out of bounds.
                        Ok(SearchResult::NotFound(ends_len))
                    }
                }
            })
            .map(|result| result.map(|r| r.to_ends_index(ends_len) as u64))
            .collect::<VortexResult<Vec<_>>>()?;
        let buffer = Buffer::from(physical_indices_vec);

        PrimitiveArray::new(buffer, validity.clone())
    });

    array.values().take(physical_indices.into_array())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_buffer::buffer;

    use crate::RunEnd;
    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEnd::encode(
            buffer![1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5].into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
    }

    #[test]
    fn ree_take() {
        let taken = ree_array().take(buffer![9, 8, 1, 3].into_array()).unwrap();
        let expected = PrimitiveArray::from_iter(vec![5i32, 5, 1, 4]).into_array();
        assert_arrays_eq!(taken, expected);
    }

    #[test]
    fn ree_take_end() {
        let taken = ree_array().take(buffer![11].into_array()).unwrap();
        let expected = PrimitiveArray::from_iter(vec![5i32]).into_array();
        assert_arrays_eq!(taken, expected);
    }

    #[test]
    #[should_panic]
    fn ree_take_out_of_bounds() {
        let _array = ree_array()
            .take(buffer![12].into_array())
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
    }

    #[test]
    fn sliced_take() {
        let sliced = ree_array().slice(4..9).unwrap();
        let taken = sliced.take(buffer![1, 3, 4].into_array()).unwrap();

        let expected = PrimitiveArray::from_iter(vec![4i32, 2, 5]).into_array();
        assert_arrays_eq!(taken, expected);
    }

    #[test]
    fn ree_take_nullable() {
        let taken = ree_array()
            .take(PrimitiveArray::from_option_iter([Some(1), None]).into_array())
            .unwrap();

        let expected = PrimitiveArray::from_option_iter([Some(1i32), None]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    /// Regression test: when `indices.len()` exceeds `array.len()`, the take impl falls back
    /// (returns `None`) so the canonical executor picks up the small RunEnd canonicalize +
    /// AVX2 gather path. Without the fallback, `Dict<RunEnd>.execute::<PrimitiveArray>` would
    /// pay `N * log K` work per index against a tiny ends array. See `dict_runend_canonical`
    /// in `encodings/runend/benches/chunked_exec.rs`.
    #[test]
    fn ree_dict_take_dense_indices() -> vortex_error::VortexResult<()> {
        use std::sync::LazyLock;

        use vortex_array::IntoArray;
        use vortex_array::VortexSessionExecute;
        use vortex_array::arrays::DictArray;
        use vortex_array::session::ArraySession;
        use vortex_array::validity::Validity;
        use vortex_buffer::Buffer;
        use vortex_session::VortexSession;

        static SESSION: LazyLock<VortexSession> =
            LazyLock::new(|| VortexSession::empty().with::<ArraySession>());
        let mut ctx = SESSION.create_execution_ctx();

        // dict_size=8, inner_run=2 → ends=[2,4,6,8], values=[0,1,2,3]
        let dict_values =
            PrimitiveArray::new(Buffer::<i32>::from_iter([0, 0, 1, 1, 2, 2, 3, 3]), Validity::NonNullable)
                .into_array();
        let dict_re = RunEnd::encode(dict_values, &mut ctx)?.into_array();

        // 32 codes (>> dict.len()=8), each `i % 8` so the result is 0,0,1,1,...,3,3 repeated 4x.
        let codes_buf: Vec<u32> = (0..32u32).map(|i| i % 8).collect();
        let codes =
            PrimitiveArray::new(Buffer::<u32>::from_iter(codes_buf), Validity::NonNullable).into_array();
        let dict = DictArray::try_new(codes, dict_re)?.into_array();

        let taken = dict.execute::<PrimitiveArray>(&mut ctx)?;
        let expected: Vec<i32> = (0..32).map(|i| (i % 8) / 2).collect();
        assert_arrays_eq!(
            taken.into_array(),
            PrimitiveArray::new(Buffer::<i32>::from_iter(expected), Validity::NonNullable)
                .into_array()
        );
        Ok(())
    }

    #[rstest]
    #[case(ree_array())]
    #[case(RunEnd::encode(
        buffer![1u8, 1, 2, 2, 2, 3, 3, 3, 3, 4].into_array(),
        &mut LEGACY_SESSION.create_execution_ctx(),
    ).unwrap())]
    #[case(RunEnd::encode(
        PrimitiveArray::from_option_iter([
            Some(10),
            Some(10),
            None,
            None,
            Some(20),
            Some(20),
            Some(20),
        ])
        .into_array(),
        &mut LEGACY_SESSION.create_execution_ctx(),
    ).unwrap())]
    #[case(RunEnd::encode(buffer![42i32, 42, 42, 42, 42].into_array(),
        &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap())]
    #[case(RunEnd::encode(
        buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array(),
        &mut LEGACY_SESSION.create_execution_ctx(),
    ).unwrap())]
    #[case({
        let mut values = Vec::new();
        for i in 0..20 {
            for _ in 0..=i {
                values.push(i);
            }
        }
        RunEnd::encode(
            PrimitiveArray::from_iter(values).into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
    })]
    fn test_take_runend_conformance(#[case] array: RunEndArray) {
        test_take_conformance(&array.into_array());
    }

    #[rstest]
    #[case(ree_array().slice(3..6).unwrap())]
    #[case({
        let array = RunEnd::encode(
            buffer![1i32, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        array.slice(2..8).unwrap()
    })]
    fn test_take_sliced_runend_conformance(#[case] sliced: ArrayRef) {
        test_take_conformance(&sliced);
    }
}
