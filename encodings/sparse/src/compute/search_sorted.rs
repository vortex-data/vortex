use std::cmp::Ordering;

use vortex_array::Array;
use vortex_array::compute::{SearchResult, SearchSortedFn, SearchSortedSide};
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

// Find the fill position relative to patches, in case of fill being in between patches we want to find the right most
// index of the fill relative to patches.
fn fill_position(array: &SparseArray, side: SearchSortedSide) -> VortexResult<usize> {
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
        let fill_index = array.patches().search_index(fill_result_index)?.to_index();
        match fill_result {
            // If fill value is present in patches this would be the index of the next or previous value after the fill value depending on the side
            SearchResult::Found(_) => match side {
                SearchSortedSide::Left => {
                    usize::try_from(&array.patches().indices().scalar_at(fill_index - 1)?)? + 1
                }
                SearchSortedSide::Right => {
                    if fill_index < array.patches().num_patches() {
                        usize::try_from(&array.patches().indices().scalar_at(fill_index)?)?
                    } else {
                        fill_result_index
                    }
                }
            },
            // If the fill value is not in patches but falls in between two patch values we want to take the right most index of that will match the fill value
            // This will then be min/maxed with result of searching for value in patches
            SearchResult::NotFound(_) => {
                if fill_index < array.patches().num_patches() {
                    usize::try_from(&array.patches().indices().scalar_at(fill_index)?)?
                } else {
                    fill_result_index
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::conformance::search_sorted::rstest_reuse::apply;
    use vortex_array::compute::conformance::search_sorted::{search_sorted_conformance, *};
    use vortex_array::compute::{SearchResult, SearchSortedSide, search_sorted};
    use vortex_array::{Array, ArrayRef, IntoArray};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_error::VortexUnwrap;
    use vortex_scalar::Scalar;

    use crate::SparseArray;

    #[apply(search_sorted_conformance)]
    fn sparse_search_sorted(
        #[case] array: ArrayRef,
        #[case] value: i32,
        #[case] side: SearchSortedSide,
        #[case] expected: SearchResult,
    ) {
        let sparse_array = SparseArray::encode(&array, None).vortex_unwrap();
        let res = search_sorted(&sparse_array, value, side).unwrap();
        assert_eq!(res, expected);
    }

    fn high_fill_in_patches() -> ArrayRef {
        SparseArray::try_new(
            buffer![17u64, 18, 19].into_array(),
            buffer![33_i32, 44, 55].into_array(),
            20,
            Scalar::primitive(33, Nullability::NonNullable),
        )
        .unwrap()
        .into_array()
    }

    fn low_fill_in_patches() -> ArrayRef {
        SparseArray::try_new(
            buffer![0u64, 1, 2].into_array(),
            buffer![33_i32, 44, 55].into_array(),
            20,
            Scalar::primitive(55, Nullability::NonNullable),
        )
        .unwrap()
        .into_array()
    }

    fn low_high_fill_in_patches_low() -> ArrayRef {
        SparseArray::try_new(
            buffer![0u64, 1, 17, 18, 19].into_array(),
            buffer![11i32, 22, 33, 44, 55].into_array(),
            20,
            Scalar::primitive(22, Nullability::NonNullable),
        )
        .unwrap()
        .into_array()
    }

    fn low_high_fill_in_patches_high() -> ArrayRef {
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
    #[case(
        high_fill_in_patches(),
        33,
        SearchSortedSide::Left,
        SearchResult::Found(0)
    )]
    #[case(
        low_fill_in_patches(),
        55,
        SearchSortedSide::Left,
        SearchResult::Found(2)
    )]
    #[case(
        low_high_fill_in_patches_low(),
        22,
        SearchSortedSide::Left,
        SearchResult::Found(1)
    )]
    #[case(
        low_high_fill_in_patches_high(),
        33,
        SearchSortedSide::Left,
        SearchResult::Found(2)
    )]
    #[case(
        high_fill_in_patches(),
        33,
        SearchSortedSide::Right,
        SearchResult::Found(18)
    )]
    #[case(
        low_fill_in_patches(),
        55,
        SearchSortedSide::Right,
        SearchResult::Found(20)
    )]
    #[case(
        low_high_fill_in_patches_low(),
        22,
        SearchSortedSide::Right,
        SearchResult::Found(17)
    )]
    #[case(
        low_high_fill_in_patches_high(),
        33,
        SearchSortedSide::Right,
        SearchResult::Found(18)
    )]
    fn search_fill(
        #[case] array: ArrayRef,
        #[case] search: i32,
        #[case] side: SearchSortedSide,
        #[case] expected: SearchResult,
    ) {
        let res = search_sorted(&array, search, side).vortex_unwrap();
        assert_eq!(res, expected);
    }
}
