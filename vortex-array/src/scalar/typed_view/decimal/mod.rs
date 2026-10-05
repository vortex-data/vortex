// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`DecimalScalar`] and [`DecimalValue`].

mod arithmetic;
mod dvalue;
mod integer_cast;
mod scalar;

pub(crate) use arithmetic::decimal_numeric_result_dtype;
pub(crate) use arithmetic::decimal_numeric_work_dtype;
pub use dvalue::DecimalValue;
pub(crate) use integer_cast::DecimalToIntegerCast;
pub use scalar::DecimalScalar;

#[cfg(test)]
mod tests;
