// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! First-party package initialization for the default compressor.

use vortex_session::VortexSession;

/// Register the default encoding packages and their compression schemes.
/// Repeated initialization is idempotent.
pub fn initialize(session: &VortexSession) {
    vortex_compressor::builtins::initialize(session);
    vortex_alp::schemes::initialize(session);
    vortex_datetime_parts::schemes::initialize(session);
    vortex_decimal_byte_parts::schemes::initialize(session);
    vortex_fastlanes::schemes::initialize(session);
    vortex_fsst::schemes::initialize(session);
    vortex_onpair::schemes::initialize(session);
    vortex_runend::schemes::initialize(session);
    vortex_sequence::schemes::initialize(session);
    vortex_sparse::schemes::initialize(session);
    vortex_zigzag::schemes::initialize(session);
}

/// Add the compact compression schemes supported by the enabled Cargo features.
pub fn initialize_compact(session: &VortexSession) {
    #[cfg(feature = "pco")]
    vortex_pco::schemes::initialize(session);
    #[cfg(feature = "zstd")]
    vortex_zstd::schemes::initialize(session);
    let _ = session;
}
