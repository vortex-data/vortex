// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_io::session::RuntimeSession;
use vortex_session::VortexSession;

use crate::session::LayoutSession;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
});
