// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Options for the scan sources that read layout readers.

use std::num::NonZeroUsize;

/// Options for a scan source that reads layout readers, such as
/// [`LayoutReaderDataSource`](crate::scan::layout::LayoutReaderDataSource).
///
/// They configure how the source runs its scans, so a host sets them when it builds the source
/// rather than on each [`ScanRequest`](vortex_scan::ScanRequest).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayoutScanOptions {
    concurrency: Option<NonZeroUsize>,
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
}
