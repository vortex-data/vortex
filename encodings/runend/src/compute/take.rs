// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
use crate::iter::trimmed_ends_iter;

const SORTED_LINEAR_RUNS_PER_INDEX_THRESHOLD: usize = 16;
const UNSORTED_LINEAR_RUNS_PER_INDEX_THRESHOLD: usize = 4;
/// Sorting the indices and merging only beats per-index binary search once the run ends are too
/// large to stay cache-resident; below this run count binary search wins.
const UNSORTED_LINEAR_MIN_RUNS: usize = 1 << 19;
/// Use a dense logical-position-to-run-index table when the array length is at most this many
/// times the number of valid indices: building the table is O(array_len) and each index then
/// resolves with a single unconditional gather.
const TABLE_LEN_PER_INDEX_THRESHOLD: usize = 8;

impl TakeExecute for RunEnd {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let primitive_indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let indices_validity = primitive_indices.validity()?;
        let indices_mask = indices_validity.execute_mask(primitive_indices.len(), ctx)?;

        let taken = match_each_integer_ptype!(primitive_indices.ptype(), |P| {
            take_indices(
                array,
                primitive_indices.as_slice::<P>(),
                &indices_validity,
                &indices_mask,
                true,
                ctx,
            )?
        });

        Ok(Some(taken))
    }
}

/// Perform a take operation on a RunEndArray without bounds-checking the indices.
///
/// The caller must guarantee that all valid indices are in bounds for the array.
pub fn take_indices_unchecked<T: AsPrimitive<usize>>(
    array: ArrayView<'_, RunEnd>,
    indices: &[T],
    validity: &Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity_mask = validity.execute_mask(indices.len(), ctx)?;
    take_indices(array, indices, validity, &validity_mask, false, ctx)
}

fn take_indices<T: AsPrimitive<usize>>(
    array: ArrayView<'_, RunEnd>,
    indices: &[T],
    validity: &Validity,
    validity_mask: &Mask,
    check_bounds: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if validity_mask.all_false() {
        return Ok(
            ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                .into_array(),
        );
    }

    let stats = valid_indices_stats(indices, validity_mask, array.len(), check_bounds)?;
    let ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;

    let physical_indices = match_each_unsigned_integer_ptype!(ends.ptype(), |I| {
        let ends = ends.as_slice::<I>();
        // Run indices fit in u32 for any realistic array; the narrower physical indices halve
        // the memory traffic of the downstream take on the values.
        if ends.len() <= u32::MAX as usize {
            PrimitiveArray::new(
                physical_indices_with_stats::<_, _, u32>(
                    ends,
                    array.offset(),
                    array.len(),
                    indices,
                    validity_mask,
                    stats,
                ),
                validity.clone(),
            )
        } else {
            PrimitiveArray::new(
                physical_indices_with_stats::<_, _, u64>(
                    ends,
                    array.offset(),
                    array.len(),
                    indices,
                    validity_mask,
                    stats,
                ),
                validity.clone(),
            )
        }
    });

    array.values().take(physical_indices.into_array())
}

#[derive(Clone, Copy)]
struct ValidIndicesStats {
    count: usize,
    sorted: bool,
}

fn physical_indices_with_stats<I, T, O>(
    ends: &[I],
    offset: usize,
    array_len: usize,
    indices: &[T],
    validity_mask: &Mask,
    stats: ValidIndicesStats,
) -> Buffer<O>
where
    I: UnsignedPType,
    T: AsPrimitive<usize>,
    O: UnsignedPType,
    usize: AsPrimitive<O>,
{
    if stats.count == 0 {
        return Buffer::zeroed(indices.len());
    }

    if stats.sorted
        && prefer_linear_scan(
            ends.len(),
            stats.count,
            SORTED_LINEAR_RUNS_PER_INDEX_THRESHOLD,
        )
    {
        return physical_indices_linear_sorted(ends, offset, indices, validity_mask);
    }

    // A dense take resolves fastest through the position table regardless of index ordering.
    // Sorted indices reach here only when there are too many runs for the sorted linear scan,
    // for example a narrow slice of a heavily run-encoded array where runs far exceed array_len.
    if array_len <= stats.count.saturating_mul(TABLE_LEN_PER_INDEX_THRESHOLD) {
        return physical_indices_table(ends, offset, array_len, indices, validity_mask);
    }

    if ends.len() >= UNSORTED_LINEAR_MIN_RUNS
        && prefer_linear_scan(
            ends.len(),
            stats.count,
            UNSORTED_LINEAR_RUNS_PER_INDEX_THRESHOLD,
        )
    {
        return physical_indices_linear_unsorted(ends, offset, indices, validity_mask, stats.count);
    }

    physical_indices_binary(ends, offset, indices, validity_mask)
}

/// Count the valid indices and determine whether they are sorted, bounds-checking each valid
/// index against `array_len` when `check_bounds` is set.
fn valid_indices_stats<T: AsPrimitive<usize>>(
    indices: &[T],
    validity_mask: &Mask,
    array_len: usize,
    check_bounds: bool,
) -> VortexResult<ValidIndicesStats> {
    debug_assert_eq!(indices.len(), validity_mask.len());

    let count = validity_mask.true_count();
    if count == 0 {
        return Ok(ValidIndicesStats {
            count,
            sorted: true,
        });
    }

    let sorted = match validity_mask.bit_buffer() {
        AllOr::All => valid_indices_sorted_all(indices, array_len, check_bounds)?,
        AllOr::None => true,
        AllOr::Some(validity) => {
            valid_indices_sorted_masked(indices, validity.iter(), array_len, check_bounds)?
        }
    };

    Ok(ValidIndicesStats { count, sorted })
}

fn valid_indices_sorted_all<T: AsPrimitive<usize>>(
    indices: &[T],
    array_len: usize,
    check_bounds: bool,
) -> VortexResult<bool> {
    // Seed the comparison with the first index; an empty or single-element slice is trivially
    // sorted, so the loop below starts from the second element.
    let Some((first, rest)) = indices.split_first() else {
        return Ok(true);
    };

    let mut previous_idx = first.as_();
    if check_bounds {
        check_index(previous_idx, array_len)?;
    }

    let mut sorted = true;
    for idx in rest {
        let idx = idx.as_();
        if check_bounds {
            check_index(idx, array_len)?;
        }
        if previous_idx > idx {
            sorted = false;
            if !check_bounds {
                break;
            }
        }
        previous_idx = idx;
    }

    Ok(sorted)
}

fn valid_indices_sorted_masked<T: AsPrimitive<usize>>(
    indices: &[T],
    is_valid: impl Iterator<Item = bool>,
    array_len: usize,
    check_bounds: bool,
) -> VortexResult<bool> {
    // Invalid positions are skipped without a bounds check, matching the take path that never
    // dereferences them.
    let mut valid = is_valid
        .zip(indices.iter())
        .filter(|(is_valid, _)| *is_valid)
        .map(|(_, idx)| idx.as_());

    // Seed the comparison with the first valid index; zero or one valid index is trivially
    // sorted, so the loop below starts from the second valid index.
    let Some(mut previous_idx) = valid.next() else {
        return Ok(true);
    };
    if check_bounds {
        check_index(previous_idx, array_len)?;
    }

    let mut sorted = true;
    for idx in valid {
        if check_bounds {
            check_index(idx, array_len)?;
        }
        if previous_idx > idx {
            sorted = false;
            if !check_bounds {
                break;
            }
        }
        previous_idx = idx;
    }

    Ok(sorted)
}

fn prefer_linear_scan(
    ends_len: usize,
    valid_count: usize,
    runs_per_index_threshold: usize,
) -> bool {
    ends_len <= valid_count.saturating_mul(runs_per_index_threshold)
}

fn check_index(index: usize, array_len: usize) -> VortexResult<()> {
    if index >= array_len {
        vortex_bail!(OutOfBounds: index, 0, array_len);
    }
    Ok(())
}

fn physical_indices_linear_sorted<I, T, O>(
    ends: &[I],
    offset: usize,
    indices: &[T],
    validity_mask: &Mask,
) -> Buffer<O>
where
    I: UnsignedPType,
    T: AsPrimitive<usize>,
    O: UnsignedPType,
    usize: AsPrimitive<O>,
{
    let mut run_idx = 0;

    match validity_mask.bit_buffer() {
        AllOr::All => Buffer::from_trusted_len_iter(indices.iter().map(|idx| {
            advance_run(ends, &mut run_idx, idx.as_() + offset);
            run_idx.as_()
        })),
        AllOr::None => unreachable!("AllInvalid indices have been handled earlier"),
        AllOr::Some(validity) => {
            // Invalid positions keep physical index zero, which is always in-bounds for the
            // values and masked out by the result validity.
            let mut physical_indices = BufferMut::zeroed(indices.len());
            for (idx_pos, (is_valid, idx)) in validity.iter().zip(indices.iter()).enumerate() {
                if !is_valid {
                    continue;
                }

                advance_run(ends, &mut run_idx, idx.as_() + offset);
                physical_indices[idx_pos] = run_idx.as_();
            }
            physical_indices.freeze()
        }
    }
}

/// Resolve indices through a dense logical-position-to-run-index table.
///
/// Building the table costs O(array_len), but every index then resolves with an unconditional
/// gather, which beats per-index binary search and sort-then-merge for dense takes. Invalid
/// indices may hold arbitrary values (even out of bounds), so they are redirected to position
/// zero instead of branching; the result validity masks whatever they resolve to.
fn physical_indices_table<I, T, O>(
    ends: &[I],
    offset: usize,
    array_len: usize,
    indices: &[T],
    validity_mask: &Mask,
) -> Buffer<O>
where
    I: UnsignedPType,
    T: AsPrimitive<usize>,
    O: UnsignedPType,
    usize: AsPrimitive<O>,
{
    let table = run_index_table::<I, O>(ends, offset, array_len);
    let table = table.as_slice();

    match validity_mask.bit_buffer() {
        AllOr::All => Buffer::from_trusted_len_iter(indices.iter().map(|idx| table[idx.as_()])),
        AllOr::None => unreachable!("AllInvalid indices have been handled earlier"),
        AllOr::Some(validity) => Buffer::from_trusted_len_iter(
            validity
                .iter()
                .zip(indices.iter())
                .map(|(is_valid, idx)| table[if is_valid { idx.as_() } else { 0 }]),
        ),
    }
}

/// Materialize the run index of every logical position in `[0, len)`.
fn run_index_table<I, O>(ends: &[I], offset: usize, len: usize) -> Buffer<O>
where
    I: UnsignedPType,
    O: UnsignedPType,
    usize: AsPrimitive<O>,
{
    let mut table = BufferMut::with_capacity(len);
    let mut run_start = 0;
    for (run_idx, run_end) in trimmed_ends_iter(ends, offset, len).enumerate() {
        table.push_n(run_idx.as_(), run_end - run_start);
        run_start = run_end;
    }
    table.freeze()
}

fn physical_indices_linear_unsorted<I, T, O>(
    ends: &[I],
    offset: usize,
    indices: &[T],
    validity_mask: &Mask,
    valid_count: usize,
) -> Buffer<O>
where
    I: UnsignedPType,
    T: AsPrimitive<usize>,
    O: UnsignedPType,
    usize: AsPrimitive<O>,
{
    let mut pairs = Vec::with_capacity(valid_count);
    match validity_mask.bit_buffer() {
        AllOr::All => {
            pairs.extend(
                indices
                    .iter()
                    .enumerate()
                    .map(|(idx_pos, idx)| (idx.as_(), idx_pos)),
            );
        }
        AllOr::None => unreachable!("AllInvalid indices have been handled earlier"),
        AllOr::Some(validity) => {
            for (idx_pos, (is_valid, idx)) in validity.iter().zip(indices.iter()).enumerate() {
                if is_valid {
                    pairs.push((idx.as_(), idx_pos));
                }
            }
        }
    }
    pairs.sort_unstable();

    let mut physical_indices = BufferMut::zeroed(indices.len());
    let mut run_idx = 0;

    for (idx, idx_pos) in pairs {
        advance_run(ends, &mut run_idx, idx + offset);
        physical_indices[idx_pos] = run_idx.as_();
    }

    physical_indices.freeze()
}

fn physical_indices_binary<I, T, O>(
    ends: &[I],
    offset: usize,
    indices: &[T],
    validity_mask: &Mask,
) -> Buffer<O>
where
    I: UnsignedPType,
    T: AsPrimitive<usize>,
    O: UnsignedPType,
    usize: AsPrimitive<O>,
{
    match validity_mask.bit_buffer() {
        AllOr::All => Buffer::from_trusted_len_iter(
            indices
                .iter()
                .map(|idx| physical_index_binary(ends, idx.as_() + offset).as_()),
        ),
        AllOr::None => Buffer::zeroed(indices.len()),
        AllOr::Some(validity) => {
            let mut physical_indices = BufferMut::zeroed(indices.len());
            for (idx_pos, (is_valid, idx)) in validity.iter().zip(indices.iter()).enumerate() {
                if !is_valid {
                    continue;
                }

                physical_indices[idx_pos] = physical_index_binary(ends, idx.as_() + offset).as_();
            }
            physical_indices.freeze()
        }
    }
}

fn physical_index_binary<I: UnsignedPType>(ends: &[I], logical_idx: usize) -> usize {
    let index = match <I as NumCast>::from(logical_idx) {
        Some(logical_idx) => ends.partition_point(|end| *end <= logical_idx),
        None => ends.len(),
    };
    index.min(ends.len() - 1)
}

fn advance_run<I: UnsignedPType>(ends: &[I], run_idx: &mut usize, logical_idx: usize) {
    // A logical index that overflows the run-end type sits past every run, so it lands in the
    // final run; otherwise advance while the current run ends at or before it.
    let Some(logical_idx) = I::from(logical_idx) else {
        *run_idx = ends.len().saturating_sub(1);
        return;
    };
    while *run_idx + 1 < ends.len() && ends[*run_idx] <= logical_idx {
        *run_idx += 1;
    }
}

#[cfg(test)]
mod tests {
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
    use vortex_mask::Mask;

    use super::physical_indices_binary;
    use super::physical_indices_linear_sorted;
    use super::physical_indices_linear_unsorted;
    use super::physical_indices_table;
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
    fn ree_take_sorted_boundaries() {
        let taken = ree_array()
            .take(buffer![0, 2, 3, 6, 8, 11].into_array())
            .unwrap();
        let expected = PrimitiveArray::from_iter(vec![1i32, 1, 4, 2, 5, 5]).into_array();
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
    fn sliced_take_unsorted_dense() {
        let sliced = ree_array().slice(4..9).unwrap();
        let taken = sliced.take(buffer![4, 0, 2, 1].into_array()).unwrap();

        let expected = PrimitiveArray::from_iter(vec![5i32, 4, 2, 4]).into_array();
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
    fn ree_take_all_null_indices() {
        let taken = ree_array()
            .take(PrimitiveArray::from_option_iter([None::<u64>, None]).into_array())
            .unwrap();

        let expected = PrimitiveArray::from_option_iter([None::<i32>, None]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    #[test]
    fn ree_take_null_index_skips_out_of_bounds_value() {
        let indices = PrimitiveArray::new(
            buffer![1u64, 12],
            Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        );
        let taken = ree_array().take(indices.into_array()).unwrap();

        let expected = PrimitiveArray::from_option_iter([Some(1i32), None]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    #[test]
    fn ree_take_unsorted_null_index_skips_out_of_bounds_value() {
        let indices = PrimitiveArray::new(
            buffer![3u64, 12, 1],
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        );
        let taken = ree_array().take(indices.into_array()).unwrap();

        let expected = PrimitiveArray::from_option_iter([Some(4i32), None, Some(1)]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    #[test]
    fn ree_take_dense_null_index_skips_out_of_bounds_value() {
        let indices = PrimitiveArray::new(
            buffer![0u64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            Validity::Array(
                BoolArray::from_iter([
                    true, true, true, true, true, true, true, true, true, true, true, false,
                ])
                .into_array(),
            ),
        );
        let taken = ree_array().take(indices.into_array()).unwrap();

        let expected = PrimitiveArray::from_option_iter([
            Some(1i32),
            Some(1),
            Some(1),
            Some(4),
            Some(4),
            Some(4),
            Some(2),
            Some(2),
            Some(5),
            Some(5),
            Some(5),
            None,
        ]);
        assert_arrays_eq!(taken, expected.into_array());
    }

    #[rstest]
    #[case(vec![3u32, 6, 8, 12], 0, 12, vec![0u64, 11, 3, 3, 7, 2, 9], Mask::new_true(7))]
    #[case(vec![3u32, 6, 8, 12], 0, 12, vec![5u64, 100, 2, 11, 0], Mask::from_indices(5, [0, 2, 3, 4]))]
    #[case(vec![6u32, 8, 12], 4, 5, vec![4u64, 0, 2, 1, 3], Mask::new_true(5))]
    fn unsorted_strategies_agree(
        #[case] ends: Vec<u32>,
        #[case] offset: usize,
        #[case] len: usize,
        #[case] indices: Vec<u64>,
        #[case] mask: Mask,
    ) {
        let binary = physical_indices_binary::<u32, u64, u64>(&ends, offset, &indices, &mask);
        let table = physical_indices_table::<u32, u64, u64>(&ends, offset, len, &indices, &mask);
        let sort_merge = physical_indices_linear_unsorted::<u32, u64, u64>(
            &ends,
            offset,
            &indices,
            &mask,
            mask.true_count(),
        );

        assert_eq!(binary.as_slice(), table.as_slice());
        assert_eq!(binary.as_slice(), sort_merge.as_slice());
    }

    #[rstest]
    #[case(vec![3u32, 6, 8, 12], 0, 12, vec![0u64, 2, 3, 6, 8, 11], Mask::new_true(6))]
    #[case(vec![3u32, 6, 8, 12], 0, 12, vec![1u64, 100, 5, 9], Mask::from_indices(4, [0, 2, 3]))]
    #[case(vec![6u32, 8, 12], 4, 5, vec![0u64, 1, 3, 4], Mask::new_true(4))]
    fn sorted_strategies_agree(
        #[case] ends: Vec<u32>,
        #[case] offset: usize,
        #[case] len: usize,
        #[case] indices: Vec<u64>,
        #[case] mask: Mask,
    ) {
        let binary = physical_indices_binary::<u32, u64, u64>(&ends, offset, &indices, &mask);
        let table = physical_indices_table::<u32, u64, u64>(&ends, offset, len, &indices, &mask);
        let sorted =
            physical_indices_linear_sorted::<u32, u64, u64>(&ends, offset, &indices, &mask);

        assert_eq!(binary.as_slice(), table.as_slice());
        assert_eq!(binary.as_slice(), sorted.as_slice());
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
