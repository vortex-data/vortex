use std::cmp::Ordering;

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::{ConstantArray, ConstantEncoding};
use crate::compute::{SearchResult, SearchSortedFn, SearchSortedSide};
use crate::ArrayLen;

impl SearchSortedFn<ConstantArray> for ConstantEncoding {
    fn search_sorted(
        &self,
        array: &ConstantArray,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        match array.scalar().partial_cmp(value).unwrap_or(Ordering::Less) {
            Ordering::Greater => Ok(SearchResult::NotFound(0)),
            Ordering::Less => Ok(SearchResult::NotFound(array.len())),
            Ordering::Equal => match side {
                SearchSortedSide::Left => Ok(SearchResult::Found(0)),
                SearchSortedSide::Right => Ok(SearchResult::Found(array.len())),
            },
        }
    }
}

#[cfg(test)]
mod test {
    use crate::array::constant::ConstantArray;
    use crate::compute::{search_sorted, SearchResult, SearchSortedSide};
    use crate::IntoArrayData;

    #[test]
    pub fn search() {
        let cst = ConstantArray::new(42, 5000).into_array();
        assert_eq!(
            search_sorted(&cst, 33, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(0)
        );
        assert_eq!(
            search_sorted(&cst, 55, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(5000)
        );
    }

    #[test]
    pub fn search_equals() {
        let cst = ConstantArray::new(42, 5000).into_array();
        assert_eq!(
            search_sorted(&cst, 42, SearchSortedSide::Left).unwrap(),
            SearchResult::Found(0)
        );
        assert_eq!(
            search_sorted(&cst, 42, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(5000)
        );
    }
}
