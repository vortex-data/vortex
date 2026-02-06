// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_io::session::RuntimeSession;
use vortex_session::VortexSession;

use crate::session::LayoutSession;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>()
});
