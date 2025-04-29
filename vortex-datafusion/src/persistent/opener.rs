use std::ops::Range;
use std::sync::{Arc, Weak};

use arrow_schema::{ArrowError, SchemaRef};
use dashmap::{DashMap, Entry};
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use futures::{FutureExt as _, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use object_store::path::Path;
use tokio::runtime::Handle;
use vortex_array::ArrayRef;
use vortex_error::VortexError;
use vortex_expr::{ExprRef, VortexExpr};
use vortex_file::scan::ScanBuilder;
use vortex_layout::LayoutReader;
use vortex_layout::scan::SplitBy;
use vortex_metrics::VortexMetrics;

use super::cache::VortexFileCache;

#[derive(Clone)]
pub(crate) struct VortexFileOpener {
    pub object_store: Arc<dyn ObjectStore>,
    pub projection: ExprRef,
    pub filter: Option<ExprRef>,
    pub(crate) file_cache: VortexFileCache,
    pub projected_arrow_schema: SchemaRef,
    pub batch_size: usize,
    metrics: VortexMetrics,
    layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
}

impl VortexFileOpener {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        object_store: Arc<dyn ObjectStore>,
        projection: Arc<dyn VortexExpr>,
        filter: Option<Arc<dyn VortexExpr>>,
        file_cache: VortexFileCache,
        projected_arrow_schema: SchemaRef,
        batch_size: usize,
        metrics: VortexMetrics,
        layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
    ) -> Self {
        Self {
            object_store,
            projection,
            filter,
            file_cache,
            projected_arrow_schema,
            batch_size,
            metrics,
            layout_readers,
        }
    }
}

impl FileOpener for VortexFileOpener {
    fn open(&self, file_meta: FileMeta) -> DFResult<FileOpenFuture> {
        let filter = self.filter.clone();
        let projection = self.projection.clone();
        let file_cache = self.file_cache.clone();
        let object_store = self.object_store.clone();
        let projected_arrow_schema = self.projected_arrow_schema.clone();
        let metrics = self.metrics.clone();
        let batch_size = self.batch_size;
        let layout_reader = self.layout_readers.clone();

        Ok(async move {
            let vxf = file_cache
                .try_get(&file_meta.object_meta, object_store)
                .await?;

            // We share out layout readers with others partitions in the scan, so we can only need to read each layout in each file once.
            let layout_reader = match layout_reader.entry(file_meta.object_meta.location.clone()) {
                Entry::Occupied(mut occupied_entry) => {
                    if let Some(reader) = occupied_entry.get().upgrade() {
                        log::trace!("reusing layout reader for {}", occupied_entry.key());
                        reader
                    } else {
                        log::trace!("creating layout reader for {}", occupied_entry.key());
                        let reader = vxf.layout_reader()?;
                        occupied_entry.insert(Arc::downgrade(&reader));
                        reader
                    }
                }
                Entry::Vacant(vacant_entry) => {
                    log::trace!("creating layout reader for {}", vacant_entry.key());
                    let reader = vxf.layout_reader()?;
                    vacant_entry.insert(Arc::downgrade(&reader));

                    reader
                }
            };

            let scan_builder = ScanBuilder::new(layout_reader);
            let scan_builder = apply_byte_range(file_meta, vxf.row_count(), scan_builder);

            Ok(scan_builder
                .with_tokio_executor(Handle::current())
                .with_metrics(metrics)
                .with_projection(projection)
                .with_some_filter(filter)
                // DataFusion likes ~8k row batches. Ideally we would respect the config,
                // but at the moment our scanner has too much overhead to process small
                // batches efficiently.
                .with_split_by(SplitBy::RowCount(8 * batch_size))
                .map_to_record_batch(projected_arrow_schema.clone())
                .into_stream()?
                .map_err(|e: VortexError| ArrowError::from(e))
                .boxed())
        }
        .boxed())
    }
}

/// If the file has a [`FileRange`](datafusion::datasource::listing::FileRange), we translate it into a row range in the file for the scan.
fn apply_byte_range(
    file_meta: FileMeta,
    row_count: u64,
    scan_builder: ScanBuilder<ArrayRef>,
) -> ScanBuilder<ArrayRef> {
    if let Some(byte_range) = file_meta.range {
        let row_range = byte_range_to_row_range(
            byte_range.start as u64..byte_range.end as u64,
            row_count,
            file_meta.object_meta.size as u64,
        );

        scan_builder.with_row_range(row_range)
    } else {
        scan_builder
    }
}

fn byte_range_to_row_range(byte_range: Range<u64>, row_count: u64, total_size: u64) -> Range<u64> {
    let average_row = total_size / row_count;
    assert!(average_row > 0, "A row must always have at least one byte");

    let start_row = byte_range.start / average_row;
    let end_row = byte_range.end / average_row;

    // We take the min here as `end_row` might overshoot
    start_row..u64::min(row_count, end_row)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0..100, 100, 100, 0..100)]
    #[case(0..105, 100, 105, 0..100)]
    #[case(0..50, 100, 105, 0..50)]
    #[case(50..105, 100, 105, 50..100)]
    #[case(0..1, 4, 8, 0..0)]
    #[case(1..8, 4, 8, 0..4)]
    fn test_range_translation(
        #[case] byte_range: Range<u64>,
        #[case] row_count: u64,
        #[case] total_size: u64,
        #[case] expected: Range<u64>,
    ) {
        assert_eq!(
            byte_range_to_row_range(byte_range, row_count, total_size),
            expected
        );
    }

    #[test]
    fn test_consecutive_ranges() {
        let row_count = 100;
        let total_size = 429;
        let bytes_a = 0..143;
        let bytes_b = 143..286;
        let bytes_c = 286..429;

        let rows_a = byte_range_to_row_range(bytes_a, row_count, total_size);
        let rows_b = byte_range_to_row_range(bytes_b, row_count, total_size);
        let rows_c = byte_range_to_row_range(bytes_c, row_count, total_size);

        assert_eq!(rows_a.end - rows_a.start, 35);
        assert_eq!(rows_b.end - rows_b.start, 36);
        assert_eq!(rows_c.end - rows_c.start, 29);

        assert_eq!(rows_a.start, 0);
        assert_eq!(rows_c.end, 100);
        for (left, right) in [rows_a, rows_b, rows_c].iter().tuple_windows() {
            assert_eq!(left.end, right.start);
        }
    }
}
