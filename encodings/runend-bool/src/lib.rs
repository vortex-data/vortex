// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run-end encoding specialized for boolean arrays.
//!
//! Boolean runs strictly alternate, so a [`RunEndBoolArray`] stores only the run `ends`, the value
//! of the first run (`start`), and an optional validity child, rather than a separate values array.

pub use array::*;

mod array;
pub mod compress;
mod compute;
mod kernel;
mod ops;

use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize run-end bool encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(RunEndBool);

    session.aggregate_fns().register_aggregate_kernel(
        RunEndBool.id(),
        Some(MinMax.id()),
        &compute::min_max::RunEndBoolMinMaxKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        RunEndBool.id(),
        Some(IsConstant.id()),
        &compute::is_constant::RunEndBoolIsConstantKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        RunEndBool.id(),
        Some(IsSorted.id()),
        &compute::is_sorted::RunEndBoolIsSortedKernel,
    );
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::dtype::PType;
    use vortex_array::test_harness::check_metadata;

    use crate::RunEndBoolMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_runend_bool_metadata() {
        check_metadata(
            "runend_bool.metadata",
            &RunEndBoolMetadata {
                ends_ptype: PType::U64 as i32,
                num_runs: u64::MAX,
                offset: u64::MAX,
                start: true,
            }
            .encode_to_vec(),
        );
    }
}
