// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod compress;
mod compute;
mod kernel;
mod rules;

/// Represents the equation A\[i\] = a * i + b.
/// This can be used for compression, fast comparisons and also for row ids.
pub use array::Sequence;
/// Represents the equation A\[i\] = a * i + b.
/// This can be used for compression, fast comparisons and also for row ids.
pub use array::SequenceArray;
pub use array::SequenceArrayParts;
pub use array::SequenceData;
pub use compress::sequence_encode;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize sequence encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(Sequence);

    // Register the Sequence-specific aggregate kernels.
    session.aggregate_fns().register_aggregate_kernel(
        Sequence::ID,
        Some(MinMax.id()),
        &compute::min_max::SequenceMinMaxKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        Sequence::ID,
        Some(IsSorted.id()),
        &compute::is_sorted::SequenceIsSortedKernel,
    );
}

// TODO(joe): hook up to the compressor
// TODO(joe): support comparisons with other operators
// TODO(joe): support list in expr pushdown
