// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`EliasFano`] encoding: the array, its encoder, and the decode behind `execute`.

mod array;
pub use array::EliasFanoData;

mod compress;
pub use compress::elias_fano_encode;

mod vtable;
pub use vtable::EliasFano;
pub use vtable::EliasFanoArray;

#[cfg(test)]
mod tests;
