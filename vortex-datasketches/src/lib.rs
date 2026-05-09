// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Apache DataSketches aggregate functions for Vortex.
//!
//! Aggregates in this crate return serialized Apache DataSketches sketches as non-null
//! `Binary` scalars. The serialized bytes are also the partial aggregate format, so partial
//! aggregate states can be merged by the same Vortex aggregate function.

pub mod hll;
pub mod tdigest;

pub use hll::Hll;
pub use hll::HllOptions;
pub use hll::HllTarget;
pub use hll::hll;
pub use tdigest::TDigest;
pub use tdigest::TDigestOptions;
pub use tdigest::tdigest;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_session::VortexSession;

/// Initialize Apache DataSketches aggregate functions in the given Vortex session.
pub fn initialize(session: &VortexSession) {
    session.aggregate_fns().register(Hll);
    session.aggregate_fns().register(TDigest);
}

#[cfg(test)]
mod tests {
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_session::VortexSession;

    use crate::hll::Hll;
    use crate::tdigest::TDigest;

    #[test]
    fn initialize_registers_aggregate_functions() {
        let session = VortexSession::empty();
        crate::initialize(&session);

        let registry = session.aggregate_fns().registry().clone();
        assert!(registry.find(&Hll.id()).is_some());
        assert!(registry.find(&TDigest.id()).is_some());
    }
}
