// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An alternative scan executor, selected by callers when [`enabled`] returns true.
//!
//! Callers configure a [`ScanBuilder`] exactly as for the default executor and swap only the final
//! call: [`prepare`] for [`ScanBuilder::prepare`], and [`into_stream`] for
//! [`ScanBuilder::into_stream`]. Both return the same types as the calls they replace, and also
//! take the [`ScanFile`] the builder's reader was opened over.
//!
//! Each split runs as a [`SplitMorsel`](crate::scan::planning::SplitMorsel): the planning protocol
//! decides which segments the split needs, and the file's own segment source fetches them.
//!
//! [`ScanBuilder`]: crate::scan::scan_builder::ScanBuilder
//! [`ScanBuilder::prepare`]: crate::scan::scan_builder::ScanBuilder::prepare
//! [`ScanBuilder::into_stream`]: crate::scan::scan_builder::ScanBuilder::into_stream

mod repeated_scan;
mod split;
mod stream;

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;

pub use repeated_scan::RepeatedScanV2;
pub use repeated_scan::prepare;
pub use stream::into_stream;

use crate::LayoutRef;
use crate::scan::planning::SegmentLocation;
use crate::segments::SegmentSource;

/// The environment variable that selects this executor when set to `1`.
pub const ENV_VAR: &str = "VORTEX_SCAN_V2";

static ENABLED: LazyLock<bool> = LazyLock::new(|| env::var(ENV_VAR).is_ok_and(|v| v == "1"));

/// Whether [`ENV_VAR`] selects this executor. Read once per process.
pub fn enabled() -> bool {
    *ENABLED
}

/// The file a scan reads: its root layout, where each segment lives, and the source that fetches
/// segments by id.
#[derive(Clone)]
pub struct ScanFile {
    /// The root layout the scan builder's reader was opened over.
    pub layout: LayoutRef,
    /// The location of every segment, indexed by segment id.
    pub locations: Arc<[SegmentLocation]>,
    /// Fetches segment bytes by id.
    pub segments: Arc<dyn SegmentSource>,
}

#[cfg(test)]
mod tests;
