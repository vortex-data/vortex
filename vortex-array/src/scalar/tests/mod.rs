// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test modules for scalars.

mod casting;
mod consistency;
mod nested;
mod nullability;
mod primitives;
mod round_trip;

use std::sync::LazyLock;

use vortex_session::VortexSession;

use crate::dtype::session::DTypeSession;

pub(crate) static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<DTypeSession>());
