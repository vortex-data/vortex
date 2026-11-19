// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_error::VortexResult;
use vortex_scan::DataSource;
use vortex_scan::DataSourceScan;
use vortex_scan::DataSourceScanRef;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::ScanRequest;
use vortex_session::VortexSession;

use crate::FileStatistics;
use crate::pruning::is_falsified_by_file_stats;

/// A file's scans with the statistics of its footer.
pub(crate) struct FileDataSource<S> {
    source: S,
    statistics: Option<FileStatistics>,
    /// Set when the inner source does not prune with the file's statistics itself.
    pruning: Option<FilePruning>,
}

/// What [`FileDataSource`] needs to prove a scan's filter false over the whole file.
struct FilePruning {
    row_count: u64,
    session: VortexSession,
}

impl<S: DataSource> FileDataSource<S> {
    pub(crate) fn new(source: S, statistics: Option<FileStatistics>) -> Self {
        Self {
            source,
            statistics,
            pruning: None,
        }
    }

    /// Skip scans whose filter the file's statistics prove false over all `row_count` rows of
    /// the file, before the inner source reads anything.
    pub(crate) fn with_file_pruning(mut self, row_count: u64, session: VortexSession) -> Self {
        self.pruning = Some(FilePruning { row_count, session });
        self
    }

    /// Whether the file's statistics prove `filter` false for every row of the file, and so for
    /// every row range of it.
    ///
    /// A filter that does not bind is not pruned: the inner scan reports the error.
    fn is_pruned(&self, filter: Option<&Expression>) -> VortexResult<bool> {
        let (Some(pruning), Some(statistics), Some(filter)) =
            (&self.pruning, &self.statistics, filter)
        else {
            return Ok(false);
        };
        let Ok(filter) = filter
            .bind(self.source.dtype())
            .and_then(|filter| filter.optimize_recursive())
        else {
            return Ok(false);
        };
        is_falsified_by_file_stats(&filter, pruning.row_count, statistics, &pruning.session)
    }
}

#[async_trait]
impl<S: DataSource> DataSource for FileDataSource<S> {
    fn dtype(&self) -> &DType {
        self.source.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        self.source.row_count()
    }

    fn byte_size(&self) -> Precision<u64> {
        self.source.byte_size()
    }

    fn deserialize_partition(
        &self,
        data: &[u8],
        session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        self.source.deserialize_partition(data, session)
    }

    async fn scan(&self, request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let filter = request.filter.clone();
        let scan = self.source.scan(request).await?;
        Ok(if self.is_pruned(filter.as_ref())? {
            Box::new(PrunedScan {
                dtype: scan.dtype().clone(),
            })
        } else {
            scan
        })
    }

    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(self
            .statistics
            .as_ref()
            .and_then(|statistics| statistics.get_by_path(field_path))
            .map(|(statistics, _)| statistics.to_stats_set())
            .unwrap_or_default())
    }

    async fn field_statistic(
        &self,
        field_path: &FieldPath,
        aggregate_fn: &AggregateFnRef,
    ) -> VortexResult<Precision<Scalar>> {
        let Some((statistics, _)) = self
            .statistics
            .as_ref()
            .and_then(|statistics| statistics.get_by_path(field_path))
        else {
            return Ok(Precision::Absent);
        };
        Ok(statistics
            .iter()
            .find(|statistic| statistic.aggregate_fn() == aggregate_fn)
            .and_then(|statistic| statistic.partial())
            .cloned()
            .map_or(Precision::Absent, Precision::Exact))
    }
}

/// A scan of a file whose statistics rule out every row: it has no partitions.
struct PrunedScan {
    dtype: DType,
}

impl DataSourceScan for PrunedScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(0usize)
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::empty().boxed()
    }
}
