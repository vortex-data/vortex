// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session extension for accessing metrics.

use crate::VortexMetrics;
use vortex_session::SessionExt;

/// Extension trait for accessing session metrics.
pub trait MetricsSessionExt: SessionExt {
    /// Return the global session metrics registry.
    fn metrics(&self) -> VortexMetrics {
        self.get::<VortexMetrics>().clone()
    }
}
impl<S: SessionExt> MetricsSessionExt for S {}
