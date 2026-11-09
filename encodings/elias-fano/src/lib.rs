// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Elias-Fano and unary encodings for integers.
//!
//! [`Unary`] stores a non-nullable unsigned column as one bit array, writing each value in unary.
//! It takes `len + num_ones` bits, where `num_ones` is the sum of the values, so it suits small
//! values in any order. It is also meant to hold the bucket counts of the high bits of the
//! Elias-Fano encoding this crate is being built towards.
//!
//! None of these encodings is in a file edition or registered by `vortex-file`, so the default
//! writers reject them for now. [`initialize`] registers them in a session.
#![deny(missing_docs)]

mod unary;

pub use unary::Unary;
pub use unary::UnaryArray;
pub use unary::UnaryData;
pub use unary::unary_encode;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Register this crate's encodings in `session`.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(Unary);
}
