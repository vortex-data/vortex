// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::match_each_native_ptype;
use vortex_array::point_fn::PointDispatch;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::search_sorted::SearchResult;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt;

impl OperationsVTable<RunEnd> for RunEnd {
    fn scalar_at(
        array: ArrayView<'_, RunEnd>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array
            .values()
            .execute_scalar(array.find_physical_index(index)?, ctx)
    }

    /// Recurse via the dispatch: search `ends` (typically small, sorted by
    /// construction), then read `values` at the resulting run index. Both
    /// child calls hit the session's caches.
    fn point_scalar_at(
        array: ArrayView<'_, RunEnd>,
        index: usize,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<Scalar> {
        let logical = index + array.offset();
        let logical_scalar = usize_as_scalar(array.ends(), logical)?;
        let run_search = d.search_sorted(array.ends(), &logical_scalar, SearchSortedSide::Right)?;
        let run_idx = run_search.to_ends_index(array.ends().len());
        d.scalar_at(array.values(), run_idx)
    }

    /// `search_sorted` on a RunEnd whose `values` is sorted is `O(log num_runs)`:
    /// binary-search `values` directly, then translate the resulting run index
    /// back to a logical position via `ends`.
    ///
    /// Precondition: the array's logical values must be sorted, which on a
    /// RunEnd-encoded array means `values` is sorted.
    fn point_search_sorted(
        array: ArrayView<'_, RunEnd>,
        value: &Scalar,
        side: SearchSortedSide,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<SearchResult> {
        let values = array.values();
        let ends = array.ends();
        let offset = array.offset();
        let logical_len = array.as_ref().len();

        let vrun = d.search_sorted(values, value, side)?;

        // Translate run index → logical position. ends are stored without the
        // slice offset subtracted, so we subtract it on lookup and clamp to the
        // [0, logical_len] window.
        let logical = match vrun.to_index() {
            0 => 0,
            i if i >= values.len() => logical_len,
            i => {
                // ends[i - 1] is the (exclusive) end position of the run before
                // the target run, in unsliced coordinates. Subtract the slice
                // offset and clamp.
                let end_scalar = d.scalar_at(ends, i - 1)?;
                let raw: usize = pvalue_to_usize(&end_scalar)?;
                raw.saturating_sub(offset).min(logical_len)
            }
        };

        Ok(match vrun {
            SearchResult::Found(_) => SearchResult::Found(logical),
            SearchResult::NotFound(_) => SearchResult::NotFound(logical),
        })
    }
}

fn pvalue_to_usize(scalar: &Scalar) -> VortexResult<usize> {
    let pv = scalar
        .as_primitive()
        .pvalue()
        .ok_or_else(|| vortex_error::vortex_err!("expected non-null ends scalar"))?;
    usize::try_from(pv)
}

/// Construct a Scalar of `ends_array.dtype()` from a usize, for searching
/// the run-ends array by logical position. Casts the usize to the ends
/// array's native ptype so the resulting scalar matches the dtype.
fn usize_as_scalar(ends_array: &ArrayRef, value: usize) -> VortexResult<Scalar> {
    let ptype = ends_array.dtype().as_ptype();
    let pvalue = match_each_native_ptype!(ptype, |P| {
        let v: P = <P as num_traits::FromPrimitive>::from_usize(value).ok_or_else(|| {
            vortex_error::vortex_err!("usize {} out of range for {:?}", value, ptype)
        })?;
        PValue::from(v)
    });
    Scalar::try_new(
        ends_array.dtype().clone(),
        Some(ScalarValue::Primitive(pvalue)),
    )
}

/// Find the physical offset for and index that would be an end of the slice i.e., one past the last element.
///
/// If the index exists in the array we want to take that position (as we are searching from the right)
/// otherwise we want to take the next one
pub(crate) fn find_slice_end_index(array: &ArrayRef, index: usize) -> VortexResult<usize> {
    let result = array
        .as_primitive_typed()
        .search_sorted(&PValue::from(index), SearchSortedSide::Right)?;
    Ok(match result {
        SearchResult::Found(i) => i,
        SearchResult::NotFound(i) => {
            if i == array.len() {
                i
            } else {
                i + 1
            }
        }
    })
}

#[cfg(test)]
mod tests {

    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::is_constant::is_constant;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::buffer;

    use crate::RunEnd;

    #[test]
    fn slice_array() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let arr = RunEnd::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
            &mut ctx,
        )
        .unwrap()
        .slice(3..8)
        .unwrap();
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 5);

        let expected = PrimitiveArray::from_iter(vec![2i32, 2, 3, 3, 3]).into_array();
        assert_arrays_eq!(arr, expected);
    }

    #[test]
    fn double_slice() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let arr = RunEnd::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
            &mut ctx,
        )
        .unwrap()
        .slice(3..8)
        .unwrap();
        assert_eq!(arr.len(), 5);

        let doubly_sliced = arr.slice(0..3).unwrap();

        let expected = PrimitiveArray::from_iter(vec![2i32, 2, 3]).into_array();
        assert_arrays_eq!(doubly_sliced, expected);
    }

    #[test]
    fn slice_end_inclusive() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let arr = RunEnd::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
            &mut ctx,
        )
        .unwrap()
        .slice(4..10)
        .unwrap();
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 6);

        let expected = PrimitiveArray::from_iter(vec![2i32, 3, 3, 3, 3, 3]).into_array();
        assert_arrays_eq!(arr, expected);
    }

    #[test]
    fn slice_at_end() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let re_array = RunEnd::try_new(
            buffer![7_u64, 10].into_array(),
            buffer![2_u64, 3].into_array(),
            &mut ctx,
        )
        .unwrap();

        assert_eq!(re_array.len(), 10);

        let sliced_array = re_array.slice(re_array.len()..re_array.len()).unwrap();
        assert!(sliced_array.is_empty());
    }

    #[test]
    fn slice_single_end() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let re_array = RunEnd::try_new(
            buffer![7_u64, 10].into_array(),
            buffer![2_u64, 3].into_array(),
            &mut ctx,
        )
        .unwrap();

        assert_eq!(re_array.len(), 10);

        let sliced_array = re_array.slice(2..5).unwrap();

        assert!(is_constant(&sliced_array, &mut ctx).unwrap())
    }

    #[test]
    fn ree_scalar_at_end() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let scalar = RunEnd::encode(
            buffer![1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5].into_array(),
            &mut ctx,
        )
        .unwrap()
        .execute_scalar(11, &mut ctx)
        .unwrap();
        assert_eq!(scalar, 5.into());
    }

    #[test]
    fn point_search_sorted_through_dispatch() {
        use vortex_array::scalar::Scalar;
        use vortex_array::search_sorted::SearchResult;
        use vortex_array::search_sorted::SearchSortedSide;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Run ends [3, 6, 8, 12], values [1, 2, 4, 7] — sorted.
        // Logical: [1, 1, 1, 2, 2, 2, 4, 4, 7, 7, 7, 7]
        let arr = RunEnd::try_new(
            buffer![3u32, 6, 8, 12].into_array(),
            buffer![1i32, 2, 4, 7].into_array(),
            &mut ctx,
        )
        .unwrap()
        .into_array();
        let mut ctx2 = LEGACY_SESSION.create_execution_ctx();
        let mut access = arr.repeated_access(&mut ctx2);

        // Found cases: each value lives in a contiguous run.
        assert_eq!(
            access
                .search_sorted(&Scalar::from(1i32), SearchSortedSide::Left)
                .unwrap(),
            SearchResult::Found(0)
        );
        assert_eq!(
            access
                .search_sorted(&Scalar::from(2i32), SearchSortedSide::Left)
                .unwrap(),
            SearchResult::Found(3)
        );
        assert_eq!(
            access
                .search_sorted(&Scalar::from(4i32), SearchSortedSide::Left)
                .unwrap(),
            SearchResult::Found(6)
        );
        assert_eq!(
            access
                .search_sorted(&Scalar::from(7i32), SearchSortedSide::Left)
                .unwrap(),
            SearchResult::Found(8)
        );

        // Right side returns one past the rightmost match.
        assert_eq!(
            access
                .search_sorted(&Scalar::from(2i32), SearchSortedSide::Right)
                .unwrap(),
            SearchResult::Found(6)
        );
        assert_eq!(
            access
                .search_sorted(&Scalar::from(7i32), SearchSortedSide::Right)
                .unwrap(),
            SearchResult::Found(12)
        );

        // NotFound cases.
        assert_eq!(
            access
                .search_sorted(&Scalar::from(0i32), SearchSortedSide::Left)
                .unwrap(),
            SearchResult::NotFound(0)
        );
        assert_eq!(
            access
                .search_sorted(&Scalar::from(3i32), SearchSortedSide::Left)
                .unwrap(),
            SearchResult::NotFound(6)
        );
        assert_eq!(
            access
                .search_sorted(&Scalar::from(99i32), SearchSortedSide::Left)
                .unwrap(),
            SearchResult::NotFound(12)
        );
    }

    #[test]
    fn point_scalar_at_through_dispatch() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Logical: [1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]
        let arr = RunEnd::try_new(
            buffer![3u32, 6, 8, 12].into_array(),
            buffer![1i32, 4, 2, 5].into_array(),
            &mut ctx,
        )
        .unwrap()
        .into_array();
        let mut ctx2 = LEGACY_SESSION.create_execution_ctx();
        let mut access = arr.repeated_access(&mut ctx2);

        let expected: Vec<i32> = vec![1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5];
        for (i, expected_v) in expected.iter().enumerate() {
            let got = access.scalar_at(i).unwrap();
            assert_eq!(got, (*expected_v).into(), "mismatch at idx {i}");
        }
    }

    #[test]
    fn slice_along_run_boundaries() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Create a runend array with runs: [1, 1, 1] [4, 4, 4] [2, 2] [5, 5, 5, 5]
        // Run ends at indices: 3, 6, 8, 12
        let arr = RunEnd::try_new(
            buffer![3u32, 6, 8, 12].into_array(),
            buffer![1i32, 4, 2, 5].into_array(),
            &mut ctx,
        )
        .unwrap();

        // Slice from start of first run to end of first run (indices 0..3)
        let slice1 = arr.slice(0..3).unwrap();
        assert_eq!(slice1.len(), 3);
        let expected = PrimitiveArray::from_iter(vec![1i32, 1, 1]).into_array();
        assert_arrays_eq!(slice1, expected);

        // Slice from start of second run to end of second run (indices 3..6)
        let slice2 = arr.slice(3..6).unwrap();
        assert_eq!(slice2.len(), 3);
        let expected = PrimitiveArray::from_iter(vec![4i32, 4, 4]).into_array();
        assert_arrays_eq!(slice2, expected);

        // Slice from start of third run to end of third run (indices 6..8)
        let slice3 = arr.slice(6..8).unwrap();
        assert_eq!(slice3.len(), 2);
        let expected = PrimitiveArray::from_iter(vec![2i32, 2]).into_array();
        assert_arrays_eq!(slice3, expected);

        // Slice from start of last run to end of last run (indices 8..12)
        let slice4 = arr.slice(8..12).unwrap();
        assert_eq!(slice4.len(), 4);
        let expected = PrimitiveArray::from_iter(vec![5i32, 5, 5, 5]).into_array();
        assert_arrays_eq!(slice4, expected);

        // Slice spanning exactly two runs (indices 3..8)
        let slice5 = arr.slice(3..8).unwrap();
        assert_eq!(slice5.len(), 5);
        let expected = PrimitiveArray::from_iter(vec![4i32, 4, 4, 2, 2]).into_array();
        assert_arrays_eq!(slice5, expected);

        // Slice from middle of first run to end of second run (indices 1..6)
        let slice6 = arr.slice(1..6).unwrap();
        assert_eq!(slice6.len(), 5);
        let expected = PrimitiveArray::from_iter(vec![1i32, 1, 4, 4, 4]).into_array();
        assert_arrays_eq!(slice6, expected);

        // Slice from start of second run to middle of third run (indices 3..7)
        let slice7 = arr.slice(3..7).unwrap();
        assert_eq!(slice7.len(), 4);
        let expected = PrimitiveArray::from_iter(vec![4i32, 4, 4, 2]).into_array();
        assert_arrays_eq!(slice7, expected);
    }
}
