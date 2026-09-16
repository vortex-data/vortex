// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::RefCell;
use std::cmp::Ordering;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::RepeatedArrayProbe;
use crate::scalar::Scalar;
use crate::search_sorted::IndexOrd;

/// A [`SearchSorted`](crate::search_sorted::SearchSorted) adapter over a sorted array of any
/// encoding, comparing elements as [`Scalar`]s.
///
/// Reads go through a [`RepeatedArrayProbe`], so the whole search shares one execution context
/// and whatever the first comparison resolved.
///
/// Prefer [`SearchSortedPrimitiveArray`](crate::search_sorted::SearchSortedPrimitiveArray) when
/// the element type is known: comparing through `Scalar` cannot read the values directly and
/// builds a scalar per probe.
pub struct SearchSortedArray<'a> {
    probe: RefCell<RepeatedArrayProbe>,
    len: usize,
    ctx: RefCell<&'a mut ExecutionCtx>,
}

impl<'a> SearchSortedArray<'a> {
    /// Wraps `array` for searching.
    pub fn new(array: &ArrayRef, ctx: &'a mut ExecutionCtx) -> Self {
        Self {
            probe: RefCell::new(array.repeated_probe()),
            len: array.len(),
            ctx: RefCell::new(ctx),
        }
    }
}

impl IndexOrd<Scalar> for SearchSortedArray<'_> {
    /// The probe and the context are separate cells, so the two borrows never overlap.
    fn index_cmp(&self, idx: usize, elem: &Scalar) -> VortexResult<Option<Ordering>> {
        let scalar = self
            .probe
            .borrow_mut()
            .execute_scalar(idx, &mut self.ctx.borrow_mut())?;
        Ok(scalar.partial_cmp(elem))
    }

    fn index_len(&self) -> usize {
        self.len
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
    use crate::scalar::Scalar;
    use crate::search_sorted::SearchResult;
    use crate::search_sorted::SearchSorted;
    use crate::search_sorted::SearchSortedArray;
    use crate::search_sorted::SearchSortedSide;
    use crate::validity::Validity;

    #[test]
    fn search_sorted_scalar() -> VortexResult<()> {
        let array =
            PrimitiveArray::new(buffer![1i32, 2, 3, 3, 4], Validity::NonNullable).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let searcher = SearchSortedArray::new(&array, &mut ctx);
        assert_eq!(
            searcher.search_sorted(&Scalar::from(3i32), SearchSortedSide::Left)?,
            SearchResult::Found(2)
        );
        assert_eq!(
            searcher.search_sorted(&Scalar::from(3i32), SearchSortedSide::Right)?,
            SearchResult::Found(4)
        );
        assert_eq!(
            searcher.search_sorted(&Scalar::from(5i32), SearchSortedSide::Left)?,
            SearchResult::NotFound(5)
        );
        Ok(())
    }

    #[test]
    fn search_sorted_scalar_with_nulls() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([None, None, Some(2i32), Some(3)]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let searcher = SearchSortedArray::new(&array, &mut ctx);
        assert_eq!(
            searcher.search_sorted(&Scalar::from(Some(2i32)), SearchSortedSide::Left)?,
            SearchResult::Found(2)
        );
        Ok(())
    }
}
