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
pub use array::SequenceData;
pub use array::SequenceDataParts;
pub use compress::sequence_encode;
use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::session::ArraySession;
use vortex_session::VortexSessionBuilder;

/// Initialize sequence encoding in the given session.
pub fn initialize(session: &mut VortexSessionBuilder) {
    session.get_mut::<ArraySession>().register(Sequence);
    kernel::initialize(session);

    // Register the Sequence-specific aggregate kernels.
    let aggregate_fns = session.get_mut::<AggregateFnSession>();
    aggregate_fns.register_aggregate_kernel(
        Sequence.id(),
        Some(MinMax.id()),
        &compute::min_max::SequenceMinMaxKernel,
    );
    aggregate_fns.register_aggregate_kernel(
        Sequence.id(),
        Some(IsSorted.id()),
        &compute::is_sorted::SequenceIsSortedKernel,
    );
}

// TODO(joe): hook up to the compressor
// TODO(joe): support comparisons with other operators
// TODO(joe): support list in expr pushdown
