// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Approximate count of distinct values (NDV).
//!
//! Query engines use the number of distinct values to estimate join and group-by cardinalities.
//! [`HyperLogLog`] is a mergeable sketch of a column's values: sketches of separate chunks or
//! files combine into the sketch of their union without rescanning the data.

mod sketch;

pub use sketch::HyperLogLog;
pub use sketch::MAX_PRECISION;
pub use sketch::MIN_PRECISION;
