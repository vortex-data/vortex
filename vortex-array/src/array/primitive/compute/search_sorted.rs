use std::cmp::Ordering;
use std::cmp::Ordering::Less;

use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::{
    IndexOrd, Len, SearchResult, SearchSorted, SearchSortedFn, SearchSortedSide,
    SearchSortedUsizeFn,
};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;

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
                Validity::AllInvalid => Ok(SearchResult::NotFound(array.len())),
                Validity::Array(_) => {
                    let pvalue: $T = value.cast(array.dtype())?.try_into()?;
                    Ok(SearchSortedNullsFirst::new(array).search_sorted(&pvalue, side))
                }
            }
        })
    }
}

impl SearchSortedUsizeFn<PrimitiveArray> for PrimitiveEncoding {
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
                        Ok(SearchSortedNullsFirst::new(array).search_sorted(&pvalue, side))
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
}

impl<T> Len for SearchSortedPrimitive<'_, T> {
    fn len(&self) -> usize {
        self.values.len()
    }
}

struct SearchSortedNullsFirst<'a, T> {
    values: SearchSortedPrimitive<'a, T>,
    validity: Validity,
}

impl<'a, T: NativePType> SearchSortedNullsFirst<'a, T> {
    pub fn new(array: &'a PrimitiveArray) -> Self {
        Self {
            values: SearchSortedPrimitive::new(array),
            validity: array.validity(),
        }
    }
}

impl<T: NativePType> IndexOrd<T> for SearchSortedNullsFirst<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &T) -> Option<Ordering> {
        if self
            .validity
            .is_null(idx)
            // TODO(ngates): SearchSortedPrimitive should hold canonical (logical) validity
            .vortex_expect("Failed to check null validity")
        {
            return Some(Less);
        }

        self.values.index_cmp(idx, elem)
    }
}

impl<T> Len for SearchSortedNullsFirst<'_, T> {
    fn len(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;

    use super::*;
    use crate::array::BoolArray;
    use crate::compute::search_sorted;
    use crate::IntoArray;

    #[test]
    fn test_search_sorted_primitive() {
        let values = buffer![1u16, 2, 3].into_array();

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

    #[test]
    fn search_sorted_nulls_first() {
        let values = PrimitiveArray::new(
            buffer![1u16, 2, 3],
            Validity::Array(
                BoolArray::new(
                    BooleanBuffer::collect_bool(3, |idx| idx != 0),
                    Nullability::NonNullable,
                )
                .into_array(),
            ),
        )
        .into_array();

        assert_eq!(
            search_sorted(&values, 0, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(1)
        );
        assert_eq!(
            search_sorted(&values, 1, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(1)
        );
        assert_eq!(
            search_sorted(&values, 2, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(2)
        );
        assert_eq!(
            search_sorted(&values, 4, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(3)
        );
    }

    #[test]
    fn search_sorted_all_nulls() {
        let values = PrimitiveArray::new(buffer![1u16, 2, 3], Validity::AllInvalid).into_array();

        assert_eq!(
            search_sorted(&values, 0, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(3)
        );
    }
}
