// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An alternative scan executor, selected by callers when [`enabled`] returns true.
//!
//! Callers configure a [`ScanBuilder`] exactly as for the default executor and swap only the final
//! call: [`prepare`] for [`ScanBuilder::prepare`], and [`into_stream`] for
//! [`ScanBuilder::into_stream`]. Both return the same types as the calls they replace.
//!
//! [`ScanBuilder`]: crate::scan::scan_builder::ScanBuilder
//! [`ScanBuilder::prepare`]: crate::scan::scan_builder::ScanBuilder::prepare
//! [`ScanBuilder::into_stream`]: crate::scan::scan_builder::ScanBuilder::into_stream

mod repeated_scan;
mod stream;

use std::env;
use std::sync::LazyLock;

pub use repeated_scan::RepeatedScanV2;
pub use repeated_scan::prepare;
pub use stream::into_stream;

/// The environment variable that selects this executor when set to `1`.
pub const ENV_VAR: &str = "VORTEX_SCAN_V2";

static ENABLED: LazyLock<bool> = LazyLock::new(|| env::var(ENV_VAR).is_ok_and(|v| v == "1"));

/// Whether [`ENV_VAR`] selects this executor. Read once per process.
pub fn enabled() -> bool {
    *ENABLED
}

#[cfg(test)]
mod tests;
