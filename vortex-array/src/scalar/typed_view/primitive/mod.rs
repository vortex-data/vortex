// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PrimitiveScalar`] and [`PValue`].

mod numeric_operator;
mod pvalue;
mod scalar;

pub use numeric_operator::NumericOperator;
pub use pvalue::PValue;
pub use scalar::PrimitiveScalar;

#[cfg(test)]
mod tests;
