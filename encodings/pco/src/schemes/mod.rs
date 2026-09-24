// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression schemes owned by this encoding package.

use vortex_array::session::ArraySessionExt;
use vortex_compressor::session::CompressionSessionExt;
pub mod float;
pub mod integer;

/// Register the encoding plugins and their compression schemes.
pub fn initialize(session: &vortex_session::VortexSession) {
    session.arrays().register(crate::Pco);
    session.register_scheme(&integer::PcoScheme);
    session.register_scheme(&float::PcoScheme);
}
