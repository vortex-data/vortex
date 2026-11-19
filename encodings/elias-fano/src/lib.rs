// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Elias-Fano and unary encodings for integers.
//!
//! [`Unary`] stores a non-nullable unsigned column as one bit array, writing each value in unary.
//! It takes `len + num_ones` bits, where `num_ones` is the sum of the values, so it suits small
//! values in any order. It is also meant to hold the bucket counts of [`EliasFano`].
//!
//! [`EliasFano`] stores a sorted, non-nullable unsigned column as two children: the number of
//! values in each bucket of high bits, and the low bits of each value. [`elias_fano_encode`]
//! leaves both to the compressor, and the EF scheme in `vortex-btrblocks` encodes them as
//! [`Unary`] and `BitPacked`.
//!
//! None of these encodings is in a file edition or registered by `vortex-file`, so the default
//! writers reject them for now. [`initialize`] registers them in a session.
#![deny(missing_docs)]

mod elias_fano;
mod unary;

pub use elias_fano::EliasFano;
pub use elias_fano::EliasFanoArray;
pub use elias_fano::EliasFanoArrayExt;
pub use elias_fano::EliasFanoArraySlotsExt;
pub use elias_fano::EliasFanoData;
pub use elias_fano::elias_fano_encode;
pub use unary::Unary;
pub use unary::UnaryArray;
pub use unary::UnaryData;
pub use unary::unary_encode;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Register this crate's encodings in `session`.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(EliasFano);
    session.arrays().register(Unary);
}
