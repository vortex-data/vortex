// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Whole-array primitive sums over run-end encoded arrays.
//!
//! Both [`Sum`] and [`SumV2`] reduce valid run values and lengths without expanding the runs. Each
//! aggregate retains its own partial representation and empty-input semantics. Grouped sums and
//! decimal inputs use the fallback.
//!
//! Floating-point multiplication can round differently from repeated addition, as with constant sums.
//!
//! [`Sum`]: vortex_array::aggregate_fn::fns::sum::Sum
//! [`SumV2`]: vortex_array::aggregate_fn::fns::sum_v2::SumV2

mod kernel;
pub(crate) use kernel::RunEndSumKernel;

mod runs;

#[cfg(test)]
mod tests;
