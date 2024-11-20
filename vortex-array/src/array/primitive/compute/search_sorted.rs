use std::cmp::Ordering;
use std::cmp::Ordering::Greater;

use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::{IndexOrd, Len, SearchResult, SearchSorted, SearchSortedFn, SearchSortedSide};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayLen};

impl SearchSortedFn<PrimitiveArray> for PrimitiveEncoding {
    fn search_sorted(
        &self,
        array: &PrimitiveArray,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        match_each_native_ptype!(array.ptype(), |$T| {
            match array.validity() {
                Validity::NonNullable | Validity::AllValid => {
                    let pvalue: $T = value.cast(array.dtype())?.try_into()?;
                    Ok(SearchSortedPrimitive::new(array).search_sorted(&pvalue, side))
                }
                Validity::AllInvalid => Ok(SearchResult::NotFound(0)),
                Validity::Array(_) => {
                    let pvalue: $T = value.cast(array.dtype())?.try_into()?;
                    Ok(SearchSortedNullsLast::new(array).search_sorted(&pvalue, side))
                }
            }
        })
    }

    #[allow(clippy::cognitive_complexity)]
    fn search_sorted_usize(
        &self,
        array: &PrimitiveArray,
        value: usize,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        match_each_native_ptype!(array.ptype(), |$T| {
            if let Some(pvalue) = num_traits::cast::<usize, $T>(value) {
                match array.validity() {
                    Validity::NonNullable | Validity::AllValid => {
                        // null-free search
                        Ok(SearchSortedPrimitive::new(array).search_sorted(&pvalue, side))
                    }
                    Validity::AllInvalid => Ok(SearchResult::NotFound(0)),
                    Validity::Array(_) => {
                        // null-aware search
                        Ok(SearchSortedNullsLast::new(array).search_sorted(&pvalue, side))
                    }
                }
            } else {
                // provided u64 is too large to fit in the provided PType, value must be off
                // the right end of the array.
                Ok(SearchResult::NotFound(array.len()))
            }
        })
    }
}

struct SearchSortedPrimitive<'a, T> {
    values: &'a [T],
}

impl<'a, T: NativePType> SearchSortedPrimitive<'a, T> {
    pub fn new(array: &'a PrimitiveArray) -> Self {
        Self {
            values: array.maybe_null_slice(),
        }
    }
}

impl<T: NativePType> IndexOrd<T> for SearchSortedPrimitive<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &T) -> Option<Ordering> {
        // SAFETY: Used in search_sorted_by same as the standard library. The search_sorted ensures idx is in bounds
        Some(unsafe { self.values.get_unchecked(idx) }.total_compare(*elem))
    }
}

impl<T> Len for SearchSortedPrimitive<'_, T> {
    fn len(&self) -> usize {
        self.values.len()
    }
}

struct SearchSortedNullsLast<'a, T> {
    values: SearchSortedPrimitive<'a, T>,
    validity: Validity,
}

impl<'a, T: NativePType> SearchSortedNullsLast<'a, T> {
    pub fn new(array: &'a PrimitiveArray) -> Self {
        Self {
            values: SearchSortedPrimitive::new(array),
            validity: array.validity(),
        }
    }
}

impl<T: NativePType> IndexOrd<T> for SearchSortedNullsLast<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &T) -> Option<Ordering> {
        if self.validity.is_null(idx) {
            return Some(Greater);
        }

        self.values.index_cmp(idx, elem)
    }
}

impl<T> Len for SearchSortedNullsLast<'_, T> {
    fn len(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compute::search_sorted;
    use crate::IntoArrayData;

    #[test]
    fn test_search_sorted_primitive() {
        let values = vec![1u16, 2, 3].into_array();

        assert_eq!(
            search_sorted(&values, 0, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(0)
        );
        assert_eq!(
            search_sorted(&values, 1, SearchSortedSide::Left).unwrap(),
            SearchResult::Found(0)
        );
        assert_eq!(
            search_sorted(&values, 1, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(1)
        );
        assert_eq!(
            search_sorted(&values, 4, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(3)
        );
    }
}
