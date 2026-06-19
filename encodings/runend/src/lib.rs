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

use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::session::ArraySession;
use vortex_session::VortexSessionBuilder;

/// Initialize run-end encoding in the given session.
pub fn initialize(session: &mut VortexSessionBuilder) {
    session.get_mut::<ArraySession>().register(RunEnd);
    kernel::initialize(session);

    // Register the RunEnd-specific aggregate kernels.
    let aggregate_fns = session.get_mut::<AggregateFnSession>();
    aggregate_fns.register_aggregate_kernel(
        RunEnd.id(),
        Some(MinMax.id()),
        &compute::min_max::RunEndMinMaxKernel,
    );
    aggregate_fns.register_aggregate_kernel(
        RunEnd.id(),
        Some(IsConstant.id()),
        &compute::is_constant::RunEndIsConstantKernel,
    );
    aggregate_fns.register_aggregate_kernel(
        RunEnd.id(),
        Some(IsSorted.id()),
        &compute::is_sorted::RunEndIsSortedKernel,
    );
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use prost::Message;
    use vortex_array::dtype::PType;
    use vortex_array::test_harness::check_metadata;
    use vortex_session::VortexSession;

    use crate::RunEndMetadata;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let mut builder = vortex_array::default_session_builder();
        crate::initialize(&mut builder);
        builder.build()
    });

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_runend_metadata() {
        check_metadata(
            "runend.metadata",
            &RunEndMetadata {
                ends_ptype: PType::U64 as i32,
                num_runs: u64::MAX,
                offset: u64::MAX,
            }
            .encode_to_vec(),
        );
    }
}
