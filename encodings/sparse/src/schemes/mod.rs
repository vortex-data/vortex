// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression schemes owned by this encoding package.

use vortex_compressor::session::CompressionSessionExt;
pub mod float;
pub mod integer;
pub mod string;

/// Register the encoding plugins and their compression schemes.
pub fn initialize(session: &vortex_session::VortexSession) {
    crate::initialize(session);
    session.register_scheme(&integer::SparseScheme);
    session.register_scheme(&float::NullDominatedSparseScheme);
    session.register_scheme(&string::NullDominatedSparseScheme);
}
