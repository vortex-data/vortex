use vortex_array::compute::unary::ScalarAtFn;
use vortex_array::compute::{filter, ArrayCompute, FilterFn, SearchSortedFn, SliceFn, TakeFn};
use vortex_array::stats::ArrayStatistics;
use vortex_array::{ArrayData, IntoCanonical};
use vortex_error::{vortex_err, VortexResult};

use crate::BitPackedArray;

mod scalar_at;
mod search_sorted;
mod slice;
mod take;

impl ArrayCompute for BitPackedArray {
    fn filter(&self) -> Option<&dyn FilterFn> {
        Some(self)
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn search_sorted(&self) -> Option<&dyn SearchSortedFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}

impl FilterFn for BitPackedArray {
    fn filter(&self, predicate: &ArrayData) -> VortexResult<ArrayData> {
        let _predicate_true_count = predicate
            .statistics()
            .compute_true_count()
            .ok_or_else(|| vortex_err!("Cannot compute true count of predicate"))?;

        filter(self.clone().into_canonical()?.as_ref(), predicate)
    }
}
