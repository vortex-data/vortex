// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_io::session::RuntimeSession;
use vortex_metrics::VortexMetrics;
use vortex_session::VortexSession;
use vortex_session::VortexSessionRef;

use crate::session::LayoutSession;

pub static SESSION: LazyLock<VortexSessionRef> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<VortexMetrics>()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>()
        .freeze()
});
