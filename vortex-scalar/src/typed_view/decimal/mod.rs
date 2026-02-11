// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`DecimalScalar`] and [`DecimalValue`].

mod dvalue;
mod scalar;

pub use dvalue::DecimalValue;
pub use scalar::DecimalScalar;

#[cfg(test)]
mod tests;
