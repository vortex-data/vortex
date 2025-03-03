use std::cmp::Ordering;

use vortex_array::Array;
use vortex_array::compute::{SearchResult, SearchSortedFn, SearchSortedSide, SearchSortedUsizeFn};
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
        let min_index = array.patches().min_index()?;
        // For a sorted array the patches can be either at the beginning or at the end of the array
        if min_index == 0 {
            match value
                .partial_cmp(array.fill_scalar())
                .vortex_expect("value and fill scalar must have same dtype")
            {
                Ordering::Less => array.patches().search_sorted(value.clone(), side),
                Ordering::Equal => Ok(SearchResult::Found(if side == SearchSortedSide::Left {
                    // In case of patches being at the beginning we want the first index after the end of patches
                    array.patches().indices().len()
                } else {
                    array.len()
                })),
                Ordering::Greater => Ok(SearchResult::NotFound(array.len())),
            }
        } else {
            match value
                .partial_cmp(array.fill_scalar())
                .vortex_expect("value and fill scalar must have same dtype")
            {
                Ordering::Less => Ok(SearchResult::NotFound(0)),
                Ordering::Equal => Ok(SearchResult::Found(if side == SearchSortedSide::Left {
                    0
                } else {
                    // Searching from right the min_index is one value after the last fill value
                    min_index
                })),
                Ordering::Greater => array.patches().search_sorted(value.clone(), side),
            }
        }
    }
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

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::NotFound(20))]
    #[case(sparse_high_non_null_fill(), SearchResult::NotFound(20))]
    #[case(sparse_low(), SearchResult::NotFound(20))]
    fn search_larger_than_left(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 66, SearchSortedSide::Left).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::NotFound(20))]
    #[case(sparse_high_non_null_fill(), SearchResult::NotFound(20))]
    #[case(sparse_low(), SearchResult::NotFound(20))]
    fn search_larger_than_right(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 66, SearchSortedSide::Right).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::NotFound(17))]
    #[case(sparse_high_non_null_fill(), SearchResult::NotFound(0))]
    #[case(sparse_low(), SearchResult::NotFound(0))]
    fn search_less_than_left(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 21, SearchSortedSide::Left).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::NotFound(17))]
    #[case(sparse_high_non_null_fill(), SearchResult::NotFound(0))]
    #[case(sparse_low(), SearchResult::NotFound(0))]
    fn search_less_than_right(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 21, SearchSortedSide::Right).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::Found(18))]
    #[case(sparse_high_non_null_fill(), SearchResult::Found(18))]
    #[case(sparse_low(), SearchResult::Found(1))]
    fn search_patches_found_left(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 44, SearchSortedSide::Left).unwrap();
        assert_eq!(res, expected);
    }

    #[rstest]
    #[case(sparse_high_null_fill(), SearchResult::Found(19))]
    #[case(sparse_high_non_null_fill(), SearchResult::Found(19))]
    #[case(sparse_low(), SearchResult::Found(2))]
    fn search_patches_found_right(#[case] array: ArrayRef, #[case] expected: SearchResult) {
        let res = search_sorted(&array, 44, SearchSortedSide::Right).unwrap();
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
    fn search_fill_right(
        #[case] array: ArrayRef,
        #[case] search: Scalar,
        #[case] expected: SearchResult,
    ) {
        let res = search_sorted(&array, search, SearchSortedSide::Right).unwrap();
        assert_eq!(res, expected);
    }
}
