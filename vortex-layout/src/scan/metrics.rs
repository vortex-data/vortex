// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Resource usage for scans that the caller owns, and its export to a metrics registry.

pub use vortex_scan::metrics::ScanMetrics;
pub(crate) use vortex_scan::metrics::ScanRecorder;
pub use vortex_scan::metrics::ScanUsage;
pub use vortex_scan::metrics::ScanUsageSnapshot;
