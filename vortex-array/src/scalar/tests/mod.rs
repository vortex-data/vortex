// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test modules for the vortex-scalar crate.

mod casting;
mod consistency;
mod nested;
mod nullability;
mod primitives;
mod round_trip;

use std::sync::LazyLock;

use vortex_dtype::session::DTypeSession;
use vortex_session::VortexSession;

pub(crate) static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<DTypeSession>());
