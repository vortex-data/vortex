// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An alternative scan executor, selected by callers when [`enabled`] returns true.
//!
//! The executor has its own [`ScanBuilder`], a copy of the default one that also takes the
//! [`ScanFile`] the reader was opened over.
//!
//! Callers that configure a default [`ScanBuilder`](crate::scan::scan_builder::ScanBuilder) can
//! instead swap only the final call: [`prepare`] for its `prepare`, and [`into_stream`] for its
//! `into_stream`. Both copy the default builder into this executor's and return the same types as
//! the calls they replace.
//!
//! Each split runs as a [`FilterPlanner`](crate::scan::planning::FilterPlanner), which hands the
//! rows that survive to a [`ProjectionPlanner`](crate::scan::planning::ProjectionPlanner) and its
//! morsel. Both execute physical plans lowered from the file's layout, on the planning driver, and
//! the file's own segment source serves their reads.
//!
//! The driver runs inside the split's future, on whichever thread polls it, and the future awaits
//! the reads it starts from the file's segment source. Every split announces the segments its
//! plans are likely to read when it starts, so the source can coalesce them.
//!
//! A caller that drives the planning protocol itself takes the splits as [`SplitPlan`]s from
//! [`RepeatedScanV2::split_plans`] and admits them to its own run, reading through
//! [`RepeatedScanV2::io`].

mod conjuncts;
mod file;
pub(crate) mod io;
mod lower;
pub(crate) mod prefetch;
mod pruning;
mod repeated_scan;
mod scan_builder;
mod share;
mod split;
pub(crate) mod splits;
mod stream;

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;

pub use file::FilePlans;
pub use repeated_scan::RepeatedScanV2;
pub use repeated_scan::prepare;
pub use scan_builder::ScanBuilder;
pub use split::SplitPlan;
pub use stream::into_stream;
use vortex_io::request::IoService;

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
    /// Serves the splits' reads, when the file provides a service for them; otherwise they are
    /// served from `segments`.
    pub io: Option<Arc<dyn IoService>>,
    /// Retains layout plans, dictionary values, and zone statistics for an explicitly cached
    /// reader. Ordinary decoded segments and query expressions are not retained here.
    /// Only share this state between scans over the same layout and segment source.
    pub plans: Option<Arc<FilePlans>>,
}

#[cfg(test)]
mod tests;
