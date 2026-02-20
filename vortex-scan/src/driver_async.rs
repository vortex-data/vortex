// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;

use crate::api::DataSource;
use crate::api::DataSourceRef;
use crate::api::DataSourceScan;
use crate::api::DataSourceScanRef;
use crate::api::ScanRequest;
use crate::api::SplitStream;

/// A data source that drives multiple child data sources.
///
/// How many files do we want to be scanning at once? Does it matter? Should we just keep launching
/// splits? Yes... but splits can be large. We have one split per file. So how many do we drive
/// at once? Maybe splits are too small?
///
/// We kind of need to know, at the single file scan level, when all splits have been launched
/// but not yet resolved. Do we need another level in the hierarchy?!
///
///
///
struct MultiDataSource {
    dtype: DType,
    children: Vec<DataSourceRef>,
    /// How many child data sources to open ahead-of-time during the scan.
    prefetch: usize,
    handle: Handle,
}

#[async_trait]
impl DataSource for MultiDataSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count_estimate(&self) -> Option<Precision<u64>> {
        todo!()
    }

    fn byte_size_estimate(&self) -> Option<Precision<u64>> {
        todo!()
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        // We create a stream to open each child scan, and we buffer it by the prefetch count.
        let scans = stream::iter(self.children.clone().into_iter())
            .map(|ds| ds.scan(scan_request.clone()))
            .buffer_unordered(self.prefetch)
            .boxed();

        // Questions:
        // * How many scans do we try to open concurrently? If we want to run 40 scans concurrently,
        //   and keep 8 in reserve that are pre-opened but not yet scanning, then surely from the
        //   beginning we should spawn 40 + 8 scans immediately. If we just do a naive buffered
        //   stream, then we'll only open 8 at a time until we ramp up to the 40 concurrent ones.
    }

    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet> {
        todo!()
    }
}

struct MultiScan {
    dtype: DType,
    prefetch_task: Task<()>,
}

impl MultiScan {
    fn try_new(handle: &Handle) -> VortexResult<Self> {}
}

impl DataSourceScan for MultiScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn split_count_estimate(&self) -> Option<Precision<usize>> {
        todo!()
    }

    fn splits(self: Box<Self>) -> SplitStream {
        todo!()
    }
}
