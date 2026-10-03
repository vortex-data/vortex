// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A map encoding that shreds frequently occurring keys into dedicated columns.
//!
//! Observability data attaches a label map to every sample. A handful of keys (`__name__`, `job`,
//! `instance`) appear in almost every row, while the long tail varies by metric. [`shred`] moves
//! each frequent key into its own row-aligned column and leaves the rest in a residual map, so
//! reading one label is a column read instead of a per-row key search.

pub use array::*;
pub use shred::*;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

mod array;
mod columnar;
mod decode;
mod flat;
mod gather;
pub mod keyset;
pub mod labels;
pub mod ops;
pub mod point;
mod rowcmp;
mod rules;
pub mod scheme;
mod shred;
pub mod squeeze;

#[cfg(test)]
mod tests;

/// Registers the shredded map encoding in the given session.
pub fn initialize(session: &VortexSession) {
    vortex_sparse::initialize(session);
    session.arrays().register(ShreddedMap);
    session.arrays().register(keyset::KeySetMap);
}
