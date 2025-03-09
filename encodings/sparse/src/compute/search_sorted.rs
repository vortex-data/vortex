use std::cmp::Ordering;

use vortex_array::Array;
use vortex_array::compute::{
    SearchResult, SearchSortedFn, SearchSortedSide, SearchSortedUsizeFn, scalar_at,
};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::{SparseArray, SparseEncoding};

impl SearchSortedFn<&SparseArray> for SparseEncoding {
    fn search_sorted(
        &self,
        array: &SparseArray,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        // first search result in patches
        let patches_result = array.patches().search_sorted(value.clone(), side)?;
        match patches_result {
            SearchResult::Found(i) => {
                if value == array.fill_scalar() {
                    // Find the relevant position of the fill value in the patches
                    let fill_index = fill_position(array, side)?;
                    match side {
                        SearchSortedSide::Left => Ok(SearchResult::Found(i.min(fill_index))),
                        SearchSortedSide::Right => Ok(SearchResult::Found(i.max(fill_index))),
                    }
                } else {
                    Ok(SearchResult::Found(i))
                }
            }
            SearchResult::NotFound(i) => {
                // Find the relevant position of the fill value in the patches
                let fill_index = fill_position(array, side)?;

                // Adjust the position of the search value relative to the position of the fill value
                match value
                    .partial_cmp(array.fill_scalar())
                    .vortex_expect("value and fill scalar must have same dtype")
                {
                    Ordering::Less => Ok(SearchResult::NotFound(i.min(fill_index))),
                    Ordering::Equal => match side {
                        SearchSortedSide::Left => Ok(SearchResult::Found(i.min(fill_index))),
                        SearchSortedSide::Right => Ok(SearchResult::Found(i.max(fill_index))),
                    },
                    Ordering::Greater => Ok(SearchResult::NotFound(i.max(fill_index))),
                }
            }
        }
    }
}

fn fill_position(array: &SparseArray, side: SearchSortedSide) -> VortexResult<usize> {
    // In not found case we need to find the relative position of fill value to the patches
    let fill_result = if array.fill_scalar().is_null() {
        // For null fill the patches can only ever be after the fill
        SearchResult::NotFound(array.patches().min_index()?)
    } else {
        array
            .patches()
            .search_sorted(array.fill_scalar().clone(), side)?
    };
    let fill_result_index = fill_result.to_index();
    // Find the relevant position of the fill value in the patches
    Ok(if fill_result_index <= array.patches().min_index()? {
        // [fill, ..., patch]
        0
    } else if fill_result_index > array.patches().max_index()? {
        // [patch, ..., fill]
        array.len()
    } else {
        // [patch, fill, ..., fill, patch]
        match side {
            SearchSortedSide::Left => fill_result_index,
            SearchSortedSide::Right => {
                // When searching from right we need to find the right most occurrence of our fill value. If fill value
                // is present in patches this would be the index of the next value after the fill value
                let fill_index = array.patches().search_index(fill_result_index)?.to_index();
                if fill_index < array.patches().num_patches() {
                    // Since we are searching from right the fill_index is the index one after the found one
                    let next_index =
                        usize::try_from(&scalar_at(array.patches().indices(), fill_index)?)?;
                    // fill value is dense with a next patch value we want to return the original fill_index,
                    // i.e. the fill value cannot exist between fill_index and next_index
                    if fill_index + 1 == next_index {
                        fill_index
                    } else {
                        next_index
                    }
                } else {
                    fill_index
                }
            }
        }
    })
}

impl SearchSortedUsizeFn<&SparseArray> for SparseEncoding {
    fn search_sorted_usize(
        &self,
        array: &SparseArray,
        value: usize,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        let Ok(target) = Scalar::from(value).cast(array.dtype()) else {
            // If the downcast fails, then the target is too large for the dtype.
            return Ok(SearchResult::NotFound(array.len()));
        };
        SearchSortedFn::search_sorted(self, array, &target, side)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::{SearchResult, SearchSortedSide, search_sorted};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::SparseArray;

    fn sparse_high_null_fill() -> ArrayRef {
        SparseArray::try_new(
            buffer![17u64, 18, 19].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
            20,
            Scalar::null_typed::<i32>(),
        )
        .unwrap()
        .into_array()
    }

    fn sparse_high_non_null_fill() -> ArrayRef {
        SparseArray::try_new(
            buffer![17u64, 18, 19].into_array(),
            buffer![33_i32, 44, 55].into_array(),
            20,
            Scalar::primitive(22, Nullability::NonNullable),
        )
        .unwrap()
        .into_array()
    }

    fn sparse_low() -> ArrayRef {
        SparseArray::try_new(
            buffer![0u64, 1, 2].into_array(),
            buffer![33_i32, 44, 55].into_array(),
            20,
            Scalar::primitive(60, Nullability::NonNullable),
        )
        .unwrap()
        .into_array()
    }

    fn sparse_low_high() -> ArrayRef {
        SparseArray::try_new(
            buffer![0u64, 1, 17, 18, 19].into_array(),
            buffer![11i32, 22, 33, 44, 55].into_array(),
            20,
            Scalar::primitive(30, Nullability::NonNullable),
        )
        .unwrap()
        .into_array()
    }

    fn sparse_high_fill_in_patches() -> ArrayRef {
        SparseArray::try_new(
            buffer![17u64, 18, 19].into_array(),
            buffer![33_i32, 44, 55].into_array(),
            20,
            Scalar::primitive(33, Nullability::NonNullable),
        )
        .unwrap()
        .into_array()
    }

    fn sparse_low_fill_in_patches() -> ArrayRef {
        SparseArray::try_new(
            buffer![0u64, 1, 2].into_array(),
            buffer![33_i32, 44, 55].into_array(),
            20,
            Scalar::primitive(55, Nullability::NonNullable),
        )
        .unwrap()
        .into_array()
    }

    fn sparse_low_high_fill_in_patches_low() -> ArrayRef {
        SparseArray::try_new(
            buffer![0u64, 1, 17, 18, 19].into_array(),
            buffer![11i32, 22, 33, 44, 55].into_array(),
            20,
            Scalar::primitive(22, Nullability::NonNullable),
        )
        .unwrap()
        .into_array()
    }

    fn sparse_low_high_fill_in_patches_high() -> ArrayRef {
        SparseArray::try_new(
            buffer![0u64, 1, 17, 18, 19].into_array(),
            buffer![11i32, 22, 33, 44, 55].into_array(),
            20,
            Scalar::primitive(33, Nullability::NonNullable),
        )
        .unwrap()
        .into_array()
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::NotFound(20))]
    #[case(sparse_high_non_null_fill(), SearchResult::NotFound(20))]
    #[case(sparse_low(), SearchResult::NotFound(20))]
    #[case(sparse_low_high(), SearchResult::NotFound(20))]
    fn search_larger_than_left(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 66, SearchSortedSide::Left).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::NotFound(20))]
    #[case(sparse_high_non_null_fill(), SearchResult::NotFound(20))]
    #[case(sparse_low(), SearchResult::NotFound(20))]
    #[case(sparse_low_high(), SearchResult::NotFound(20))]
    fn search_larger_than_right(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 66, SearchSortedSide::Right).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::NotFound(17))]
    #[case(sparse_high_non_null_fill(), SearchResult::NotFound(0))]
    #[case(sparse_low(), SearchResult::NotFound(0))]
    #[case(sparse_low_high(), SearchResult::NotFound(1))]
    fn search_less_than_left(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 21, SearchSortedSide::Left).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::NotFound(17))]
    #[case(sparse_high_non_null_fill(), SearchResult::NotFound(0))]
    #[case(sparse_low(), SearchResult::NotFound(0))]
    #[case(sparse_low_high(), SearchResult::NotFound(1))]
    fn search_less_than_right(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 21, SearchSortedSide::Right).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::Found(18))]
    #[case(sparse_high_non_null_fill(), SearchResult::Found(18))]
    #[case(sparse_low(), SearchResult::Found(1))]
    #[case(sparse_low_high(), SearchResult::Found(18))]
    fn search_patches_found_left(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 44, SearchSortedSide::Left).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::Found(19))]
    #[case(sparse_high_non_null_fill(), SearchResult::Found(19))]
    #[case(sparse_low(), SearchResult::Found(2))]
    #[case(sparse_low_high(), SearchResult::Found(19))]
    fn search_patches_found_right(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 44, SearchSortedSide::Right).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::NotFound(19))]
    #[case(sparse_high_non_null_fill(), SearchResult::NotFound(19))]
    #[case(sparse_low(), SearchResult::NotFound(2))]
    #[case(sparse_low_high(), SearchResult::NotFound(19))]
    fn search_mid_patches_not_found_left(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 45, SearchSortedSide::Left).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::NotFound(19))]
    #[case(sparse_high_non_null_fill(), SearchResult::NotFound(19))]
    #[case(sparse_low(), SearchResult::NotFound(2))]
    #[case(sparse_low_high(), SearchResult::NotFound(19))]
    fn search_mid_patches_not_found_right(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 45, SearchSortedSide::Right).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[should_panic]
    #[case(sparse_high_null_fill(), Scalar::null_typed::<i32>(), SearchResult::Found(18))]
    #[case(
        sparse_high_non_null_fill(),
        Scalar::primitive(22, Nullability::NonNullable),
        SearchResult::Found(0)
    )]
    #[case(
        sparse_low(),
        Scalar::primitive(60, Nullability::NonNullable),
        SearchResult::Found(3)
    )]
    #[case(
        sparse_low_high(),
        Scalar::primitive(30, Nullability::NonNullable),
        SearchResult::Found(2)
    )]
    #[case(
        sparse_high_fill_in_patches(),
        Scalar::primitive(33, Nullability::NonNullable),
        SearchResult::Found(0)
    )]
    #[case(
        sparse_low_fill_in_patches(),
        Scalar::primitive(55, Nullability::NonNullable),
        SearchResult::Found(2)
    )]
    #[case(
        sparse_low_high_fill_in_patches_low(),
        Scalar::primitive(22, Nullability::NonNullable),
        SearchResult::Found(1)
    )]
    #[case(
        sparse_low_high_fill_in_patches_high(),
        Scalar::primitive(33, Nullability::NonNullable),
        SearchResult::Found(17)
    )]
    fn search_fill_left(
        #[case] array: ArrayRef,
        #[case] search: Scalar,
        #[case] expected: SearchResult,
    ) {
        let res = search_sorted(&array, search, SearchSortedSide::Left).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[should_panic]
    #[case(sparse_high_null_fill(), Scalar::null_typed::<i32>(), SearchResult::Found(18))]
    #[case(
        sparse_high_non_null_fill(),
        Scalar::primitive(22, Nullability::NonNullable),
        SearchResult::Found(17)
    )]
    #[case(
        sparse_low(),
        Scalar::primitive(60, Nullability::NonNullable),
        SearchResult::Found(20)
    )]
    #[case(
        sparse_low_high(),
        Scalar::primitive(30, Nullability::NonNullable),
        SearchResult::Found(17)
    )]
    #[case(
        sparse_high_fill_in_patches(),
        Scalar::primitive(33, Nullability::NonNullable),
        SearchResult::Found(18)
    )]
    #[case(
        sparse_low_fill_in_patches(),
        Scalar::primitive(55, Nullability::NonNullable),
        SearchResult::Found(20)
    )]
    #[case(
        sparse_low_high_fill_in_patches_low(),
        Scalar::primitive(22, Nullability::NonNullable),
        SearchResult::Found(17)
    )]
    #[case(
        sparse_low_high_fill_in_patches_high(),
        Scalar::primitive(33, Nullability::NonNullable),
        SearchResult::Found(18)
    )]
    fn search_fill_right(
        #[case] array: ArrayRef,
        #[case] search: Scalar,
        #[case] expected: SearchResult,
    ) {
        let res = search_sorted(&array, search, SearchSortedSide::Right).unwrap();
        assert_eq!(res, expected);
    }
}
