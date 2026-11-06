// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Unary`] encoding: the array, its encoder, and the decode behind `execute`.

mod array;
pub use array::UnaryData;

mod compress;
pub use compress::unary_encode;

mod vtable;
pub use vtable::Unary;
pub use vtable::UnaryArray;

#[cfg(test)]
mod tests;
