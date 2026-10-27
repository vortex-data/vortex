// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Membership in a constant set. [`PreparedSet`] holds a constant list together with the
//! [`ArrayHaystack`](crate::ArrayHaystack) of its elements, so that the kernels of a needle
//! encoding can look up their own values.

mod data;
mod function;

pub use data::PreparedSetData;
pub use function::PreparedSet;

#[cfg(test)]
mod tests;
