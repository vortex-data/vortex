// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
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
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::RunEnd;
use crate::RunEndData;

const SORTED_TAKE_MERGE_MIN_VALID_COUNT: usize = 64;
const UNSORTED_TAKE_SORT_MERGE_MIN_VALID_COUNT: usize = 8_192;
const UNSORTED_TAKE_SORT_MERGE_RUN_RATIO: usize = 16;

impl TakeExecute for RunEnd {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let primitive_indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let indices_validity = primitive_indices.validity()?;

        let checked_indices = match_each_integer_ptype!(primitive_indices.ptype(), |P| {
            check_indices(
                primitive_indices.as_slice::<P>(),
                array.len(),
                &indices_validity,
            )?
        });

        take_indices_unchecked(&array, &checked_indices, &indices_validity).map(Some)
    }
}

/// Perform a take operation on a RunEndArray by binary searching for each of the indices.
pub fn take_indices_unchecked<T: AsPrimitive<usize>>(
    array: &RunEndData,
    indices: &[T],
    validity: &Validity,
) -> VortexResult<ArrayRef> {
    let ends = array.ends().to_primitive();
    let validity_mask = validity.to_mask(indices.len());

    let physical_indices = match_each_integer_ptype!(ends.ptype(), |I| {
        let end_slices = ends.as_slice::<I>();
        let physical_indices_vec =
            collect_physical_indices(end_slices, indices, array.offset(), &validity_mask)?;
        let buffer = Buffer::from(physical_indices_vec);

        Ok::<PrimitiveArray, vortex_error::VortexError>(PrimitiveArray::new(
            buffer,
            validity.clone(),
        ))
    });

    array.values().take(physical_indices?.into_array())
}

fn check_indices<T: AsPrimitive<usize>>(
    indices: &[T],
    len: usize,
    validity: &Validity,
) -> VortexResult<Vec<usize>> {
    match validity.to_mask(indices.len()).bit_buffer() {
        AllOr::All => indices
            .iter()
            .copied()
            .map(|idx| check_index(idx.as_(), len))
            .collect(),
        AllOr::None => Ok(vec![0; indices.len()]),
        AllOr::Some(mask) => indices
            .iter()
            .copied()
            .enumerate()
            .map(|(position, idx)| {
                if mask.value(position) {
                    check_index(idx.as_(), len)
                } else {
                    Ok(0)
                }
            })
            .collect(),
    }
}

fn check_index(index: usize, len: usize) -> VortexResult<usize> {
    if index >= len {
        vortex_bail!(OutOfBounds: index, 0, len);
    }

    Ok(index)
}

fn collect_physical_indices<E: AsPrimitive<usize> + NumCast + PartialOrd, T: AsPrimitive<usize>>(
    end_slices: &[E],
    indices: &[T],
    offset: usize,
    validity_mask: &Mask,
) -> VortexResult<Vec<u64>> {
    let valid_count = validity_mask.true_count();
    if valid_count == 0 {
        return Ok(vec![0; indices.len()]);
    }

    if !should_try_sorted_merge(valid_count) {
        return search_physical_indices(end_slices, indices, offset, validity_mask);
    }

    let mut physical_indices = vec![0; indices.len()];
    if try_fill_physical_indices_sorted(
        end_slices,
        indices,
        offset,
        validity_mask,
        &mut physical_indices,
    ) {
        return Ok(physical_indices);
    }

    if !should_sort_merge(valid_count, end_slices.len()) {
        return search_physical_indices(end_slices, indices, offset, validity_mask);
    }

    let mut indexed_indices = collect_logical_indices(indices, offset, validity_mask);
    indexed_indices.sort_unstable_by_key(|&(logical_index, _)| logical_index);
    fill_physical_indices(end_slices, indexed_indices, &mut physical_indices);

    Ok(physical_indices)
}

fn should_try_sorted_merge(valid_count: usize) -> bool {
    valid_count >= SORTED_TAKE_MERGE_MIN_VALID_COUNT
}

fn should_sort_merge(valid_count: usize, run_count: usize) -> bool {
    valid_count >= UNSORTED_TAKE_SORT_MERGE_MIN_VALID_COUNT
        && valid_count >= run_count.saturating_mul(UNSORTED_TAKE_SORT_MERGE_RUN_RATIO)
}

fn try_fill_physical_indices_sorted<E: AsPrimitive<usize>, T: AsPrimitive<usize>>(
    end_slices: &[E],
    indices: &[T],
    offset: usize,
    validity_mask: &Mask,
    physical_indices: &mut [u64],
) -> bool {
    let mut previous = None;
    let mut run_index = 0usize;

    let mut record_index = |position: usize, logical_index: usize| {
        if previous.is_some_and(|prev| logical_index < prev) {
            return false;
        }
        previous = Some(logical_index);
        physical_indices[position] = advance_to_run(end_slices, &mut run_index, logical_index);
        true
    };

    match validity_mask.bit_buffer() {
        AllOr::All => indices
            .iter()
            .copied()
            .enumerate()
            .all(|(position, idx)| record_index(position, idx.as_() + offset)),
        AllOr::None => true,
        AllOr::Some(mask) => indices
            .iter()
            .copied()
            .enumerate()
            .filter(|(position, _)| mask.value(*position))
            .all(|(position, idx)| record_index(position, idx.as_() + offset)),
    }
}

fn search_physical_indices<E: NumCast + PartialOrd, T: AsPrimitive<usize>>(
    end_slices: &[E],
    indices: &[T],
    offset: usize,
    validity_mask: &Mask,
) -> VortexResult<Vec<u64>> {
    let ends_len = end_slices.len();
    let mut physical_indices = vec![0; indices.len()];

    let mut record_index = |position: usize, logical_index: usize| -> VortexResult<()> {
        physical_indices[position] = match E::from(logical_index) {
            Some(logical_index) => end_slices
                .search_sorted(&logical_index, SearchSortedSide::Right)?
                .to_ends_index(ends_len) as u64,
            None => SearchResult::NotFound(ends_len).to_ends_index(ends_len) as u64,
        };

        Ok(())
    };

    match validity_mask.bit_buffer() {
        AllOr::All => {
            for (position, idx) in indices.iter().copied().enumerate() {
                record_index(position, idx.as_() + offset)?;
            }
        }
        AllOr::None => {}
        AllOr::Some(mask) => {
            for (position, idx) in indices.iter().copied().enumerate() {
                if mask.value(position) {
                    record_index(position, idx.as_() + offset)?;
                }
            }
        }
    }

    Ok(physical_indices)
}

fn collect_logical_indices<T: AsPrimitive<usize>>(
    indices: &[T],
    offset: usize,
    validity_mask: &Mask,
) -> Vec<(usize, usize)> {
    match validity_mask.bit_buffer() {
        AllOr::All => indices
            .iter()
            .copied()
            .enumerate()
            .map(|(position, idx)| (idx.as_() + offset, position))
            .collect(),
        AllOr::None => Vec::new(),
        AllOr::Some(mask) => indices
            .iter()
            .copied()
            .enumerate()
            .filter(|(position, _)| mask.value(*position))
            .map(|(position, idx)| (idx.as_() + offset, position))
            .collect(),
    }
}

fn fill_physical_indices<E: AsPrimitive<usize>>(
    end_slices: &[E],
    logical_indices: impl IntoIterator<Item = (usize, usize)>,
    physical_indices: &mut [u64],
) {
    let mut run_index = 0usize;

    for (logical_index, position) in logical_indices {
        physical_indices[position] = advance_to_run(end_slices, &mut run_index, logical_index);
    }
}

fn advance_to_run<E: AsPrimitive<usize>>(
    end_slices: &[E],
    run_index: &mut usize,
    logical_index: usize,
) -> u64 {
    while *run_index < end_slices.len() && logical_index >= end_slices[*run_index].as_() {
        *run_index += 1;
    }

    debug_assert!(*run_index < end_slices.len());
    *run_index as u64
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_mask::AllOr;
    use vortex_mask::Mask;

    use crate::RunEnd;
    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEnd::encode(buffer![1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5].into_array()).unwrap()
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

    #[test]
    fn ree_take_null_out_of_bounds_is_ignored() -> VortexResult<()> {
        let indices = PrimitiveArray::new(
            buffer![0u32, 100, 7],
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        )
        .into_array();
        let taken = ree_array().take(indices)?;

        let expected = PrimitiveArray::from_option_iter([Some(1i32), None, Some(2)]).into_array();
        assert_arrays_eq!(taken, expected);
        Ok(())
    }

    #[test]
    fn ree_take_duplicate_indices() -> VortexResult<()> {
        let taken = ree_array().take(buffer![0u32, 0, 8, 8].into_array())?;

        let expected = PrimitiveArray::from_iter([1i32, 1, 5, 5]).into_array();
        assert_arrays_eq!(taken, expected);
        Ok(())
    }

    #[test]
    fn ree_take_sorted_filter_indices() -> VortexResult<()> {
        let indices = match Mask::from_indices(ree_array().len(), vec![0, 1, 6, 7, 8]).indices() {
            AllOr::Some(indices) => PrimitiveArray::from_iter(
                indices
                    .iter()
                    .map(|&idx| u32::try_from(idx).vortex_expect("mask index must fit in u32")),
            )
            .into_array(),
            AllOr::All | AllOr::None => unreachable!(),
        };
        let taken = ree_array().take(indices)?;

        let expected = PrimitiveArray::from_iter([1i32, 1, 2, 2, 5]).into_array();
        assert_arrays_eq!(taken, expected);
        Ok(())
    }

    #[test]
    fn sliced_take_sorted_filter_indices() -> VortexResult<()> {
        let sliced = ree_array().slice(2..10)?;
        let indices = match Mask::from_indices(sliced.len(), vec![0, 3, 4, 5, 7]).indices() {
            AllOr::Some(indices) => PrimitiveArray::from_iter(
                indices
                    .iter()
                    .map(|&idx| u32::try_from(idx).vortex_expect("mask index must fit in u32")),
            )
            .into_array(),
            AllOr::All | AllOr::None => unreachable!(),
        };
        let taken = sliced.take(indices)?;

        let expected = PrimitiveArray::from_iter([1i32, 4, 2, 2, 5]).into_array();
        assert_arrays_eq!(taken, expected);
        Ok(())
    }

    #[test]
    fn sorted_merge_threshold_is_conservative_for_small_takes() {
        assert!(!super::should_try_sorted_merge(
            super::SORTED_TAKE_MERGE_MIN_VALID_COUNT.saturating_sub(1)
        ));
        assert!(super::should_try_sorted_merge(
            super::SORTED_TAKE_MERGE_MIN_VALID_COUNT
        ));
    }

    #[test]
    fn unsorted_sort_merge_threshold_scales_with_run_count() {
        assert!(!super::should_sort_merge(
            super::UNSORTED_TAKE_SORT_MERGE_MIN_VALID_COUNT.saturating_sub(1),
            1,
        ));
        assert!(!super::should_sort_merge(32_768, 4_096));
        assert!(super::should_sort_merge(32_768, 256));
    }

    #[rstest]
    #[case(ree_array())]
    #[case(RunEnd::encode(
        buffer![1u8, 1, 2, 2, 2, 3, 3, 3, 3, 4].into_array(),
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
    ).unwrap())]
    #[case(RunEnd::encode(buffer![42i32, 42, 42, 42, 42].into_array())
        .unwrap())]
    #[case(RunEnd::encode(
        buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array(),
    ).unwrap())]
    #[case({
        let mut values = Vec::new();
        for i in 0..20 {
            for _ in 0..=i {
                values.push(i);
            }
        }
        RunEnd::encode(PrimitiveArray::from_iter(values).into_array()).unwrap()
    })]
    fn test_take_runend_conformance(#[case] array: RunEndArray) {
        test_take_conformance(&array.into_array());
    }

    #[rstest]
    #[case(ree_array().slice(3..6).unwrap())]
    #[case({
        let array = RunEnd::encode(
            buffer![1i32, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].into_array(),
        )
        .unwrap();
        array.slice(2..8).unwrap()
    })]
    fn test_take_sliced_runend_conformance(#[case] sliced: ArrayRef) {
        test_take_conformance(&sliced);
    }
}
