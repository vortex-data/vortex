// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Options for the scan sources that read layout readers.

use std::num::NonZeroUsize;

use crate::scan::split_by::SplitBy;

/// Options for a scan source that reads layout readers, such as
/// [`LayoutReaderDataSource`](crate::scan::layout::LayoutReaderDataSource).
///
/// They configure how the source runs its scans, so a host sets them when it builds the source
/// rather than on each [`ScanRequest`](vortex_scan::ScanRequest).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayoutScanOptions {
    concurrency: Option<NonZeroUsize>,
    split_by: Option<SplitBy>,
}

impl LayoutScanOptions {
    /// Decodes at most `concurrency` splits at once for each partition being read, or for each
    /// stream of an opened scan.
    ///
    /// It does not bound the whole scan: a scan read by several workers decodes up to this many
    /// splits for each partition each worker reads. Each split being decoded holds the decoded
    /// chunks of every column it overlaps, so a scan that runs alongside others can set this to
    /// bound its memory, and a host that polls one split at a time on each of its threads sets it
    /// to one. It does not bound I/O: segment reads for later splits may still be coalesced ahead
    /// of them. Without it the source picks a bound from the runtime.
    pub fn with_concurrency(mut self, concurrency: NonZeroUsize) -> Self {
        self.concurrency = Some(concurrency);
        self
    }

    /// The most splits decoded at once for each partition or stream, if bounded.
    pub fn concurrency(&self) -> Option<NonZeroUsize> {
        self.concurrency
    }

    /// Splits each scan by `split_by` instead of [`SplitBy::default`].
    ///
    /// With [`SplitBy::Parts`], the source cuts its scans into the parts the planner-driven
    /// executor cuts: [`split`](vortex_scan::DataSourceScan::split) into `n` pieces cuts at least
    /// `n` parts, as that executor does, and an opened scan cuts its batches at the chunk
    /// boundaries of their columns, as [`ScanExec::with_batch_cuts`] does. Comparing the two
    /// executors with it set compares their execution over the same units of work.
    ///
    /// [`ScanExec::with_batch_cuts`]: vortex_scan::stream::ScanExec::with_batch_cuts
    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = Some(split_by);
        self
    }

    /// How each scan is split, if not by [`SplitBy::default`].
    pub fn split_by(&self) -> Option<SplitBy> {
        self.split_by
    }
}
