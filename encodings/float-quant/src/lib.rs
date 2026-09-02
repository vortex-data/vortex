// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless float quantization as two integer child arrays.

mod array;
mod rules;
mod slice;

pub use array::*;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Register the float quantization encoding in one session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(FloatQuant);
}
