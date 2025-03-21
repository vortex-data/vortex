use std::cmp::Ordering;
use std::cmp::Ordering::Less;

use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::Array;
use crate::arrays::PrimitiveEncoding;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::{
    IndexOrd, SearchResult, SearchSorted, SearchSortedFn, SearchSortedSide, SearchSortedUsizeFn,
};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;

impl SearchSortedFn<&PrimitiveArray> for PrimitiveEncoding {
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
                Validity::AllInvalid => Ok(SearchResult::NotFound(array.len())),
                Validity::Array(_) => {
                    let pvalue: $T = value.cast(array.dtype())?.try_into()?;
                    Ok(SearchSortedNullsFirst::try_new(array)?.search_sorted(&pvalue, side))
                }
            }
        })
    }
}

impl SearchSortedUsizeFn<&PrimitiveArray> for PrimitiveEncoding {
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
                    Validity::AllInvalid => Ok(SearchResult::NotFound(array.len())),
                    Validity::Array(_) => {
                        // null-aware search
                        Ok(SearchSortedNullsFirst::try_new(array)?.search_sorted(&pvalue, side))
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
            values: array.as_slice(),
        }
    }
}

impl<T: NativePType> IndexOrd<T> for SearchSortedPrimitive<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &T) -> Option<Ordering> {
        // SAFETY: Used in search_sorted_by same as the standard library. The search_sorted ensures idx is in bounds
        Some(unsafe { self.values.get_unchecked(idx) }.total_compare(*elem))
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

struct SearchSortedNullsFirst<'a, T> {
    values: SearchSortedPrimitive<'a, T>,
    mask: Mask,
}

impl<'a, T: NativePType> SearchSortedNullsFirst<'a, T> {
    pub fn try_new(array: &'a PrimitiveArray) -> VortexResult<Self> {
        Ok(Self {
            values: SearchSortedPrimitive::new(array),
            mask: array.validity_mask()?,
        })
    }
}

impl<T: NativePType> IndexOrd<T> for SearchSortedNullsFirst<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &T) -> Option<Ordering> {
        if !self.mask.value(idx) {
            return Some(Less);
        }

        self.values.index_cmp(idx, elem)
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod test {
    use crate::ArrayRef;
    use crate::compute::conformance::search_sorted::rstest_reuse::apply;
    use crate::compute::conformance::search_sorted::{search_sorted_conformance, *};
    use crate::compute::{SearchResult, SearchSortedSide, search_sorted};

    #[apply(search_sorted_conformance)]
    fn primitive_search_sorted(
        #[case] array: ArrayRef,
        #[case] value: i32,
        #[case] side: SearchSortedSide,
        #[case] expected: SearchResult,
    ) {
        let res = search_sorted(&array, value, side).unwrap();
        assert_eq!(res, expected);
    }
}
