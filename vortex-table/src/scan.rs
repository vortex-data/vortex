// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scanning a table: open its files, reading each footer in one read, drop the files whose statistics
//! rule the filter out, and run each remaining file's ordinary scan, in table order.
//!
//! Every file is scanned by its own [`ScanDriver`]. TODO: once IO requests carry a source id, a
//! single plan driver can run all the files' planners together.

use std::sync::Arc;

use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_file::VortexFile;
use vortex_file::v2::ScanDriver;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::Manifest;

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

    /// How many splits of a file to work on at once.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Opens the manifest's files, through `open` for their segments, and keeps those the
    /// filter may match on their file statistics.
    pub async fn plan(
        &self,
        manifest: &Manifest,
        open: impl Fn(&str) -> VortexResult<Arc<dyn VortexReadAt>>,
    ) -> VortexResult<Vec<PlannedFile>> {
        let mut planned = Vec::new();
        let mut row_offset = 0;
        for entry in &manifest.entries {
            let file = entry.open(&self.session, open(&entry.path)?).await?;
            let pruned = match &self.filter {
                Some(filter) => file.can_prune(filter)?,
                None => false,
            };
            if !pruned {
                planned.push(PlannedFile { file, row_offset });
            }
            row_offset += entry.row_count;
        }
        Ok(planned)
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
