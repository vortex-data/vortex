// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scanning a table: open its files through an [`Openable`], several at once and each footer in
//! one read, drop the files whose statistics rule the filter out, and run each remaining file's
//! ordinary scan, in table order.
//!
//! Every file is scanned by its own [`ScanDriver`]. TODO: once IO requests carry a source id, a
//! single plan driver can run all the files' planners together.

use std::sync::Arc;

use futures::StreamExt;
use futures::TryStreamExt;
use futures::future;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_expr::Expression;
use vortex_file::VortexFile;
use vortex_file::v2::ScanDriver;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::Manifest;
use crate::ManifestEntry;
use crate::Openable;

/// How many files a scan opens and reads the footers of at once.
pub const OPEN_CONCURRENCY: usize = 16;

/// A file a scan keeps, and the table row its first row is.
pub struct PlannedFile {
    /// The opened file.
    pub file: VortexFile,
    /// The table row of the file's first row.
    pub row_offset: u64,
}

/// A scan over a table's files.
pub struct TableScan {
    session: VortexSession,
    projection: Option<Expression>,
    filter: Option<Expression>,
    concurrency: usize,
}

impl TableScan {
    /// A scan of every column of every row.
    pub fn new(session: VortexSession) -> Self {
        Self {
            session,
            projection: None,
            filter: None,
            concurrency: get_available_parallelism().unwrap_or(1),
        }
    }

    /// Returns only these columns, or this expression of each row.
    pub fn with_projection(mut self, projection: Expression) -> Self {
        self.projection = Some(projection);
        self
    }

    /// Returns only the rows this predicate keeps.
    pub fn with_filter(mut self, filter: Expression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// How many chunks of a file run at once; zero runs them inline. Defaults to the available
    /// parallelism.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Opens the manifest's files, each through the [`Openable`] `open` gives for its entry, up
    /// to [`OPEN_CONCURRENCY`] at once, and keeps those the filter may match on their file
    /// statistics, in table order.
    pub async fn plan(
        &self,
        manifest: &Manifest,
        open: impl Fn(&ManifestEntry) -> VortexResult<Arc<dyn Openable>>,
    ) -> VortexResult<Vec<PlannedFile>> {
        let row_offsets = manifest.entries.iter().scan(0, |next, entry| {
            let row_offset = *next;
            *next += entry.row_count;
            Some(row_offset)
        });
        stream::iter(manifest.entries.iter().zip(row_offsets))
            .map(|(entry, row_offset)| {
                let openable = open(entry);
                async move {
                    let read = openable?.open().await?;
                    let file = entry.open(&self.session, read).await?;
                    let pruned = match &self.filter {
                        Some(filter) => file.can_prune(filter)?,
                        None => false,
                    };
                    VortexResult::Ok((!pruned).then_some(PlannedFile { file, row_offset }))
                }
            })
            .buffered(OPEN_CONCURRENCY)
            .try_filter_map(|planned| future::ready(Ok(planned)))
            .try_collect()
            .await
    }

    /// Scans the planned files one after another, in table order.
    pub fn run(
        &self,
        files: Vec<PlannedFile>,
    ) -> VortexResult<BoxStream<'static, VortexResult<ArrayRef>>> {
        let scans = files
            .into_iter()
            .map(|PlannedFile { file, row_offset }| {
                let dtype = file.dtype();
                let mut scan = ScanBuilder::new(self.session.clone(), file.scan_file())
                    .with_some_filter(self.filter.as_ref().map(|f| f.bind(dtype)).transpose()?)
                    .with_row_offset(row_offset);
                if let Some(projection) = &self.projection {
                    scan = scan.with_projection(projection.bind(dtype)?);
                }
                Ok(ScanDriver::new(self.session.handle(), self.concurrency, true).run(scan))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(stream::iter(scans).flatten().boxed())
    }
}
