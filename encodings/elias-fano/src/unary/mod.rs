// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Unary`] encoding: the array, its encoder, the decode behind `execute`, and the walk over
//! unset bits that decoding and validation use.

mod array;
mod bits;
mod compress;
#[cfg(test)]
mod tests;

pub use array::Unary;
pub use array::UnaryArray;
pub use array::UnaryData;
pub use compress::unary_encode;
