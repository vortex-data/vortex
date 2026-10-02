// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Precision and bounds for stored aggregate results.
//!
//! These generic types describe what is known about a value. Aggregate functions supply the meaning
//! of each result and the direction of its bounds.

mod bound;
pub use bound::*;

mod precision;
pub use precision::*;

mod stat_bound;
pub use stat_bound::*;
