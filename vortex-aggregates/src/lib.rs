// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate functions built on the vortex-array aggregate framework that the core array crate
//! does not itself depend on.
//!
//! The aggregates backing array statistics (min/max, sum, null and NaN counts, sortedness,
//! constancy, uncompressed size) live in `vortex_array::aggregate_fn::fns`. This crate holds the
//! remaining built-ins: [`count`], [`first`], [`last`], [`mean`], the byte-bounded
//! [`bounded_min`]/[`bounded_max`] used by zone maps, and the [`combined`] adapter that composes
//! two aggregates into one.
//!
//! Call [`initialize`] to register them with a session.

use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_session::VortexSession;

pub mod bounded_max;
pub mod bounded_min;
pub mod combined;
pub mod count;
pub mod first;
pub mod last;
pub mod mean;

#[cfg(test)]
mod combined_tests;

/// Register this crate's aggregate functions and grouped kernels with the given session.
pub fn initialize(session: &VortexSession) {
    let aggregate_fns = session.aggregate_fns();
    aggregate_fns.register(bounded_max::BoundedMax);
    aggregate_fns.register(bounded_min::BoundedMin);
    aggregate_fns.register(first::First);
    aggregate_fns.register(last::Last);
    aggregate_fns.register_grouped_kernel(count::Count.id(), &count::CountGroupedKernel);
}

#[cfg(test)]
fn test_session() -> VortexSession {
    let session = VortexSession::empty();
    initialize(&session);
    session
}
