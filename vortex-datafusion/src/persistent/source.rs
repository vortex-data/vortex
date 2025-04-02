use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileOpener, FileScanConfig, FileSource};
use datafusion_common::{Result as DFResult, Statistics};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use itertools::Itertools as _;
use object_store::ObjectStore;
use vortex_error::VortexExpect as _;
use vortex_expr::{Identity, VortexExpr};
use vortex_file::VORTEX_FILE_EXTENSION;
use vortex_metrics::VortexMetrics;

use super::cache::VortexFileCache;
use super::config::{ConfigProjection, FileScanConfigExt};
use super::metrics::PARTITION_LABEL;
use super::opener::VortexFileOpener;

/// A config for [`VortexFileOpener`]. Used to create [`DataSourceExec`] based physical plans.
///
/// [`DataSourceExec`]: datafusion_physical_plan::source::DataSourceExec
#[derive(Clone)]
pub struct VortexSource {
    pub(crate) file_cache: VortexFileCache,
    pub(crate) predicate: Option<Arc<dyn VortexExpr>>,
    pub(crate) projection: Option<Arc<dyn VortexExpr>>,
    pub(crate) batch_size: Option<usize>,
    pub(crate) projected_statistics: Option<Statistics>,
    pub(crate) arrow_schema: Option<SchemaRef>,
    pub(crate) metrics: VortexMetrics,
    _unused_df_metrics: ExecutionPlanMetricsSet,
}

impl VortexSource {
    pub(crate) fn new(file_cache: VortexFileCache, metrics: VortexMetrics) -> Self {
        Self {
            file_cache,
            metrics,
            projection: None,
            batch_size: None,
            projected_statistics: None,
            arrow_schema: None,
            predicate: None,
            _unused_df_metrics: Default::default(),
        }
    }

    /// Sets a [`VortexExpr`] as a predicate
    pub fn with_predicate(&self, predicate: Arc<dyn VortexExpr>) -> Self {
        let mut source = self.clone();
        source.predicate = Some(predicate);
        source
    }
}

impl FileSource for VortexSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        _base_config: &FileScanConfig,
        partition: usize,
    ) -> Arc<dyn FileOpener> {
        let partition_metrics = self
            .metrics
            .child_with_tags([(PARTITION_LABEL, partition.to_string())].into_iter());

        let batch_size = self
            .batch_size
            .vortex_expect("batch_size must be supplied to VortexSource");

        let opener = VortexFileOpener::new(
            object_store,
            self.projection.clone().unwrap_or_else(Identity::new_expr),
            self.predicate.clone(),
            self.file_cache.clone(),
            self.arrow_schema
                .clone()
                .vortex_expect("We should have a schema here"),
            batch_size,
            partition_metrics,
        );

        Arc::new(opener)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut source = self.clone();
        source.batch_size = Some(batch_size);
        Arc::new(source)
    }

    fn with_schema(&self, schema: SchemaRef) -> Arc<dyn FileSource> {
        // todo(adam): does this need to the same as `with_projection`?
        let mut source = self.clone();
        source.arrow_schema = Some(schema);
        Arc::new(source)
    }

    fn with_projection(&self, config: &FileScanConfig) -> Arc<dyn FileSource> {
        let ConfigProjection {
            arrow_schema,
            constraints: _constraints,
            statistics,
            projection_expr,
        } = config.project_for_vortex();

        let statistics = if self.predicate.is_some() {
            statistics.to_inexact()
        } else {
            statistics
        };

        let mut source = self.clone();
        source.projection = Some(projection_expr);
        source.arrow_schema = Some(arrow_schema);
        source.projected_statistics = Some(statistics);

        Arc::new(source)
    }

    fn with_statistics(&self, statistics: Statistics) -> Arc<dyn FileSource> {
        let mut source = self.clone();
        source.projected_statistics = Some(statistics);
        Arc::new(source)
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self._unused_df_metrics
    }

    fn statistics(&self) -> DFResult<Statistics> {
        let statistics = self
            .projected_statistics
            .clone()
            .vortex_expect("projected_statistics must be set");

        if self.predicate.is_some() {
            Ok(statistics.to_inexact())
        } else {
            Ok(statistics)
        }
    }

    fn file_type(&self) -> &str {
        VORTEX_FILE_EXTENSION
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        _repartition_file_min_size: usize,
        _output_ordering: Option<datafusion_physical_expr::LexOrdering>,
        config: &FileScanConfig,
    ) -> DFResult<Option<FileScanConfig>> {
        let mut new_config = config.clone();
        let file_groups = std::mem::take(&mut new_config.file_groups);
        new_config.file_groups = repartition_by_size(file_groups, target_partitions);

        Ok(Some(new_config))
    }
}

pub(crate) fn repartition_by_size(
    file_groups: Vec<Vec<PartitionedFile>>,
    desired_partitions: usize,
) -> Vec<Vec<PartitionedFile>> {
    let all_files = file_groups.iter().flatten().collect::<Vec<_>>();
    let total_file_count = all_files.len();
    let total_size = all_files.iter().map(|f| f.object_meta.size).sum::<usize>();
    let target_partition_size = total_size / desired_partitions;

    let mut partitions = Vec::with_capacity(desired_partitions);

    let mut curr_partition_size = 0;
    let mut curr_partition = Vec::default();

    let mut all_files = VecDeque::from_iter(
        all_files
            .into_iter()
            .sorted_unstable_by_key(|f| f.object_meta.size),
    );

    while !all_files.is_empty() && partitions.len() < desired_partitions {
        // If the current partition is empty, we want to bootstrap it with the biggest file we have leftover.
        let file = if curr_partition.is_empty() {
            all_files.pop_back()
        // If we already have files in the partition, we try and fill it up.
        } else {
            // Peak at the biggest file left
            let biggest_file_size = all_files
                .back()
                .vortex_expect("We must have at least one item")
                .object_meta
                .size;

            let smallest_file_size = all_files
                .front()
                .vortex_expect("We must have at least one item")
                .object_meta
                .size;

            // We try and find a file on either end that fits in the partition
            if curr_partition_size + biggest_file_size >= target_partition_size {
                all_files.pop_front()
            } else if curr_partition_size + smallest_file_size >= target_partition_size {
                all_files.pop_back()
            } else {
                None
            }
        };

        // Add a file to the partition
        if let Some(file) = file {
            curr_partition_size += file.object_meta.size;
            curr_partition.push(file.clone());
        }

        // If the partition is full, move on to the next one
        if curr_partition_size >= target_partition_size || file.is_none() {
            curr_partition_size = 0;
            partitions.push(std::mem::take(&mut curr_partition));
        }
    }

    // If we we're still missing the last partition
    if !curr_partition.is_empty() && partitions.len() != desired_partitions {
        partitions.push(std::mem::take(&mut curr_partition));
    } else if !curr_partition.is_empty() {
        for (idx, file) in curr_partition.into_iter().enumerate() {
            let new_part_idx = idx % partitions.len();
            partitions[new_part_idx].push(file.clone());
        }
    }

    for (idx, file) in all_files.into_iter().enumerate() {
        let new_part_idx = idx % partitions.len();
        partitions[new_part_idx].push(file.clone());
    }

    // Assert that we have the correct number of partitions and that the total number of files is right
    assert_eq!(total_file_count, partitions.iter().flatten().count());

    partitions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_repartition_test() {
        let file_groups = vec![vec![
            PartitionedFile::new("a", 100),
            PartitionedFile::new("b", 25),
            PartitionedFile::new("c", 25),
            PartitionedFile::new("d", 25),
            PartitionedFile::new("e", 50),
        ]];

        repartition_by_size(file_groups, 2);

        let file_groups = vec![
            (0..100)
                .map(|idx| PartitionedFile::new(format!("{idx}"), idx))
                .collect(),
        ];

        repartition_by_size(file_groups, 16);
    }
}
