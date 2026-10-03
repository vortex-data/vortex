// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Finalized results in historical file footer and array node fields.
//!
//! The codec preserves the wire schema and historical scalar types, including types that current
//! aggregate kernels cannot compute. A present nullable null is a result. A missing field is
//! absent.

mod summary;
mod truncate;

pub use summary::legacy_stats_to_results;
pub(crate) use summary::load_node_summary;
pub use summary::read_summary;
pub use summary::validate_summary_aggregates;
pub(crate) use summary::write_node_summary;
pub use summary::write_summary;
pub(crate) use truncate::truncate_summary;

#[cfg(test)]
mod tests;
