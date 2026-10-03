// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Finalized results in historical file footer fields.
//!
//! The codec preserves the wire schema and historical scalar types, including types that current
//! aggregate kernels cannot compute. A present nullable null is a result. A missing field is
//! absent.

mod footer;
pub use footer::legacy_stats_to_results;
pub use footer::read_summary;
pub use footer::write_summary;

#[cfg(test)]
mod tests;
