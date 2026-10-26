// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_error::VortexResult;
use vortex_layout::scan::layout::LayoutReaderDataSource;
use vortex_scan::DataSource;
use vortex_scan::DataSourceScanRef;
use vortex_scan::PartitionRef;
use vortex_scan::ScanRequest;
use vortex_session::VortexSession;

use crate::FileStatistics;

pub(crate) struct FileDataSource {
    source: LayoutReaderDataSource,
    statistics: Option<FileStatistics>,
}

impl FileDataSource {
    pub(crate) fn new(source: LayoutReaderDataSource, statistics: Option<FileStatistics>) -> Self {
        Self { source, statistics }
    }
}

#[async_trait]
impl DataSource for FileDataSource {
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
        self.source.scan(request).await
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
        // AggregateStats::get omits nulls for pruning; the Scan API retains exact null results.
        if let Some(statistic) = statistics
            .iter()
            .find(|statistic| statistic.aggregate_fn() == aggregate_fn)
        {
            return Ok(statistic.value().clone());
        }
        Ok(statistics.get(aggregate_fn))
    }
}
