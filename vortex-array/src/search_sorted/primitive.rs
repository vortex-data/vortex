// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::RefCell;
use std::cmp::Ordering;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::NativePType;
use crate::search_sorted::IndexOrd;

/// A [`SearchSorted`](crate::search_sorted::SearchSorted) adapter over a sorted primitive-typed
/// array, comparing elements as the native type `T`.
///
/// Values can be searched as `T`, `Option<T>`, or `usize`. Searching as `T` or `usize` treats
/// null elements as `T::zero()`; use `Option<T>` when the array may contain nulls, in which case
/// nulls sort before all non-null values.
pub struct SearchSortedPrimitiveArray<'a, T>(
    &'a ArrayRef,
    RefCell<&'a mut ExecutionCtx>,
    PhantomData<T>,
);

impl<'a, T: NativePType> SearchSortedPrimitiveArray<'a, T> {
    /// Wraps `array` for searching, panicking if the array's [`PType`](crate::dtype::PType) is
    /// not `T::PTYPE`.
    pub fn new(array: &'a ArrayRef, ctx: &'a mut ExecutionCtx) -> Self {
        assert_eq!(
            array.dtype().as_ptype(),
            T::PTYPE,
            "Array PType must match primitive type"
        );
        Self(array, RefCell::new(ctx), PhantomData)
    }

    /// Returns the value at `idx`, with nulls mapped to `T::zero()`.
    fn value(&self, idx: usize) -> VortexResult<T> {
        Ok(self
            .0
            .execute_scalar(idx, &mut self.1.borrow_mut())?
            .as_primitive()
            .typed_value::<T>()
            .unwrap_or_else(|| T::zero()))
    }
}

impl<T: NativePType> IndexOrd<T> for SearchSortedPrimitiveArray<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &T) -> VortexResult<Option<Ordering>> {
        let value = self.value(idx)?;
        Ok(Some(value.total_compare(*elem)))
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

impl<T: NativePType> IndexOrd<Option<T>> for SearchSortedPrimitiveArray<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &Option<T>) -> VortexResult<Option<Ordering>> {
        // The borrow must end before `self.value` re-borrows the ctx.
        let valid = self.0.is_valid(idx, &mut self.1.borrow_mut())?;
        let value = valid.then(|| self.value(idx)).transpose()?;

        Ok(match (value, elem.as_ref()) {
            (Some(l), Some(r)) => Some(l.total_compare(*r)),
            (Some(_), None) => Some(Ordering::Greater),
            (None, Some(_)) => Some(Ordering::Less),
            (None, None) => Some(Ordering::Equal),
        })
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

impl<T: NativePType> IndexOrd<usize> for SearchSortedPrimitiveArray<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &usize) -> VortexResult<Option<Ordering>> {
        let value = self.value(idx)?;

        let Some(elem_t) = T::from_usize(*elem) else {
            return Ok(Some(Ordering::Less));
        };

        Ok(Some(value.total_compare(elem_t)))
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::executor::VortexSessionExecute;
    use crate::search_sorted::SearchResult;
    use crate::search_sorted::SearchSorted;
    use crate::search_sorted::SearchSortedPrimitiveArray;
    use crate::search_sorted::SearchSortedSide;
    use crate::validity::Validity;

    #[test]
    fn search_sorted_optional_value() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::AllValid).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let res = SearchSortedPrimitiveArray::<i32>::new(&array, &mut ctx)
            .search_sorted(&Some(3i32), SearchSortedSide::Left)?;
        assert_eq!(res, SearchResult::Found(2));
        Ok(())
    }

    #[test]
    fn search_sorted_optional_value_with_nulls() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([None, None, Some(2i32), Some(3)]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let searcher = SearchSortedPrimitiveArray::<i32>::new(&array, &mut ctx);
        assert_eq!(
            searcher.search_sorted(&Some(2i32), SearchSortedSide::Left)?,
            SearchResult::Found(2)
        );
        assert_eq!(
            searcher.search_sorted(&None, SearchSortedSide::Right)?,
            SearchResult::Found(2)
        );
        Ok(())
    }
}
