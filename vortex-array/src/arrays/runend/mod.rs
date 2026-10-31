// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryRunEndArray;
pub use array::*;
pub use iter::trimmed_ends_iter;

mod array;
pub mod compress;
pub(crate) mod compute;
pub mod decompress_bool;
mod iter;
mod kernel;
pub mod ops;
mod probe;
mod rules;
#[cfg(test)]
#[cfg(not(codspeed))]
mod trace_tests;

#[doc(hidden)]
pub mod _benchmarking {
    pub use compute::filter::filter_run_end_primitive;
    pub use compute::take::take_indices_unchecked;

    use super::*;
}

use vortex_session::VortexSession;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
    rules::register_parent_rules(session);
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use prost::Message;
    use vortex_session::VortexSession;

    use crate::arrays::runend::RunEndMetadata;
    use crate::dtype::PType;
    use crate::test_harness::check_metadata;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
