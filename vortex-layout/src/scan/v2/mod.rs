// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An alternative scan builder and execution path for incremental scan development.
//!
//! DuckDB and DataFusion select this path when `VORTEX_SCAN_V2=1`. It initially copies the
//! default builder, preparation, and streaming behavior and shares the existing split tasks.
//! Subsequent changes can replace its execution without changing the default scan path.
//!
//! [`ScanBuilder::from_default`] transfers an already configured builder into this path.
//! Callers can also construct [`ScanBuilder`] directly without setting the environment variable.

mod repeated_scan;
mod scan_builder;

use std::env;
use std::sync::LazyLock;

pub use repeated_scan::RepeatedScanV2;
pub use scan_builder::ScanBuilder;

/// The environment variable that selects this executor when set to `1`.
pub const ENV_VAR: &str = "VORTEX_SCAN_V2";

static ENABLED: LazyLock<bool> = LazyLock::new(|| env::var(ENV_VAR).is_ok_and(|v| v == "1"));

/// Whether [`ENV_VAR`] selects this executor. Read once per process.
pub fn enabled() -> bool {
    *ENABLED
}

#[cfg(test)]
mod tests;
