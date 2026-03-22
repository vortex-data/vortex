// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryRunEndArray;
pub use array::*;
pub use iter::trimmed_ends_iter;

mod array;
#[cfg(feature = "arrow")]
mod arrow;
pub mod compress;
mod compute;
pub mod decompress_bool;
mod iter;
mod kernel;
mod ops;
mod rules;

#[doc(hidden)]
pub mod _benchmarking {
    pub use compute::take::take_indices_unchecked;

    use super::*;
}

use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize run-end encoding in the given session.
pub fn initialize(session: &mut VortexSession) {
    session.arrays().register(RunEnd);

    // Register the RunEnd-specific aggregate kernels.
    session.aggregate_fns().register_aggregate_kernel(
        RunEnd::ID,
        Some(MinMax.id()),
        &compute::min_max::RunEndMinMaxKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        RunEnd::ID,
        Some(IsConstant.id()),
        &compute::is_constant::RunEndIsConstantKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        RunEnd::ID,
        Some(IsSorted.id()),
        &compute::is_sorted::RunEndIsSortedKernel,
    );
}

#[cfg(test)]
mod tests {
    use vortex_array::ProstMetadata;
    use vortex_array::dtype::PType;
    use vortex_array::test_harness::check_metadata;

    use crate::RunEndMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_runend_metadata() {
        check_metadata(
            "runend.metadata",
            ProstMetadata(RunEndMetadata {
                ends_ptype: PType::U64 as i32,
                num_runs: u64::MAX,
                offset: u64::MAX,
            }),
        );
    }
}
