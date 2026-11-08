// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ArrayVTable;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::Slice;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::arrays::runend::compute::take_from::RunEndTakeFrom;
use crate::kernel::ExecuteParentKernel;
use crate::match_each_unsigned_integer_ptype;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;
use crate::search_sorted::IndexOrd;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedPrimitiveArray;
use crate::search_sorted::SearchSortedSide;

pub(super) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Binary.id(), RunEnd, CompareExecuteAdaptor(RunEnd));
    kernels.register_execute_parent_kernel(Slice.id(), RunEnd, RunEndSliceKernel);
    kernels.register_execute_parent_kernel(Filter.id(), RunEnd, FilterExecuteAdaptor(RunEnd));
    kernels.register_execute_parent_kernel(Dict.id(), RunEnd, TakeExecuteAdaptor(RunEnd));
    kernels.register_execute_parent_kernel(Dict.id(), RunEnd, RunEndTakeFrom);
}

/// Kernel to execute slicing on a RunEnd array.
///
/// This directly slices the RunEnd array using OperationsVTable::slice and then
/// executes the result to get the canonical form.
#[derive(Debug)]
struct RunEndSliceKernel;

impl ExecuteParentKernel<RunEnd> for RunEndSliceKernel {
    type Parent = Slice;

    fn execute_parent(
        &self,
        array: ArrayView<'_, RunEnd>,
        parent: ArrayView<'_, Slice>,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        slice(array, parent.slice_range().clone(), ctx).map(Some)
    }
}

/// The most ends a slice decodes up front instead of searching them through probes.
///
/// A probing search costs about 30 scalar reads; decoding ends costs a fraction of a nanosecond
/// each, so up to this many ends decoding wins.
const DECODE_ENDS_MAX_LEN: usize = 64 * 1024;

/// The physical range of runs covering logical `start..end`.
///
/// The end search gallops forward from the start run instead of bisecting all ends: ends strictly
/// increase, so the end run is at most `end - start` runs after the start run, and galloping
/// reads only ends near the slice, each compressed chunk at most once.
fn slice_bounds<S>(ends: &S, start: usize, end: usize) -> VortexResult<(usize, usize)>
where
    S: IndexOrd<usize>,
{
    let n = ends.index_len();
    let begin = ends
        .search_sorted(&start, SearchSortedSide::Right)?
        .to_ends_index(n);

    // Find a window `[lo, hi)` past `begin` whose last element exceeds `end`, doubling the step.
    let limit = n.min(begin + (end - start) + 1);
    let mut lo = begin;
    let mut step = 1;
    while lo + step < limit && !ends.index_gt(lo + step, &end)? {
        lo += step;
        step *= 2;
    }
    let hi = limit.min(lo + step + 1);
    // Every end before `lo` is at most `end` and every end from `hi` on is above it, so a
    // search within the window finds the same insertion point as one over all ends.
    let found = Window {
        inner: ends,
        from: lo,
        to: hi,
    }
    .search_sorted(&end, SearchSortedSide::Right)?;
    let slice_end = match found {
        SearchResult::Found(i) => lo + i,
        SearchResult::NotFound(i) => {
            let i = lo + i;
            if i == n { i } else { i + 1 }
        }
    };
    Ok((begin, slice_end))
}

/// The elements `from..to` of a sorted collection, indexed from 0.
struct Window<'a, S> {
    inner: &'a S,
    from: usize,
    to: usize,
}

impl<S: IndexOrd<usize>> IndexOrd<usize> for Window<'_, S> {
    fn index_cmp(&self, idx: usize, elem: &usize) -> VortexResult<Option<std::cmp::Ordering>> {
        self.inner.index_cmp(self.from + idx, elem)
    }

    fn index_len(&self) -> usize {
        self.to - self.from
    }
}

fn slice(
    array: ArrayView<'_, RunEnd>,
    range: Range<usize>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let new_length = range.len();

    // Searching compressed ends reads them one probe at a time, and each probe pays for a scalar
    // read; decoding a moderate number of ends up front and searching the buffer is cheaper, and
    // the decoded ends then serve as the sliced ends as well.
    let ends = if !array.ends().is::<Primitive>() && array.ends().len() <= DECODE_ENDS_MAX_LEN {
        array
            .ends()
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .into_array()
    } else {
        array.ends().clone()
    };

    let (slice_begin, slice_end) =
        match_each_unsigned_integer_ptype!(ends.dtype().as_ptype(), |T| {
            // One search for both bounds, so a probing reader (a compressed ends child) keeps
            // its state, such as a decoded chunk, from the first search into the second.
            let search = SearchSortedPrimitiveArray::<T>::new(&ends, ctx);
            slice_bounds(
                &search,
                range.start + array.offset(),
                range.end + array.offset(),
            )?
        });

    // If the sliced range contains only a single run, opt to return a ConstantArray.
    if slice_begin + 1 == slice_end {
        let value = array.values().execute_scalar(slice_begin, ctx)?;
        return Ok(ConstantArray::new(value, new_length).into_array());
    }

    // SAFETY: we maintain the ends invariant in our slice implementation
    Ok(unsafe {
        RunEnd::new_unchecked(
            ends.slice(slice_begin..slice_end)?,
            array.values().slice(slice_begin..slice_end)?,
            range.start + array.offset(),
            new_length,
        )
        .into_array()
    })
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::slice_bounds;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::runend::ops::find_physical_index;
    use crate::arrays::runend::ops::find_slice_end_index;
    use crate::arrays::runend::tests::SESSION;
    use crate::search_sorted::SearchSortedPrimitiveArray;

    /// The galloping, shared search finds the same runs as two full searches.
    #[rstest]
    #[case::short_runs(1, 4)]
    #[case::mixed_runs(1, 40)]
    #[case::long_runs(500, 3000)]
    fn slice_bounds_match_full_searches(
        #[case] min_run: u32,
        #[case] max_run: u32,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut rng = StdRng::seed_from_u64(u64::from(max_run));
        let mut end = 0u32;
        let ends: PrimitiveArray = (0..2000)
            .map(|_| {
                end += rng.random_range(min_run..=max_run);
                end
            })
            .collect();
        let len = end as usize;
        let ends = ends.into_array();
        for start in (0..len).step_by(len / 97 + 1).chain([0, len - 1]) {
            for slice_len in [1, 2, 7, 64, 1000, 65_536, len - start] {
                let stop = (start + slice_len).min(len);
                let expected = (
                    find_physical_index(&ends, start, &mut ctx)?,
                    find_slice_end_index(&ends, stop, &mut ctx)?,
                );
                let search = SearchSortedPrimitiveArray::<u32>::new(&ends, &mut ctx);
                assert_eq!(
                    slice_bounds(&search, start, stop)?,
                    expected,
                    "slice {start}..{stop}"
                );
            }
        }
        Ok(())
    }
}
