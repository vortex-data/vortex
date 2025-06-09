//! This module defines the [`VortexFile`] struct, which represents a Vortex file on disk or in memory.
//!
//! The `VortexFile` provides methods for accessing file metadata, creating segment sources for reading
//! data from the file, and initiating scans to read the file's contents into memory as Vortex arrays.
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::arrays::{ConstantArray, StructArray};
use vortex_array::stats::{Stat, StatsSet};
use vortex_dtype::{DType, Field, FieldPath, FieldPathSet};
use vortex_error::VortexResult;
use vortex_expr::pruning::checked_pruning_expr;
use vortex_expr::{ExprRef, Identifier, Scope, ScopeFieldPathSet};
use vortex_layout::LayoutReader;
use vortex_layout::scan::ScanBuilder;
use vortex_layout::segments::SegmentSource;
use vortex_metrics::VortexMetrics;

use crate::footer::Footer;
use crate::pruning::extract_relevant_file_stat_as_struct_row;

/// Represents a Vortex file, providing access to its metadata and content.
///
/// A `VortexFile` is created by opening a Vortex file using [`VortexOpenOptions`](crate::VortexOpenOptions).
/// It provides methods for accessing file metadata (such as row count, data type, and statistics)
/// and for initiating scans to read the file's contents.
#[derive(Clone)]
pub struct VortexFile {
    /// The footer of the Vortex file, containing metadata and layout information.
    pub(crate) footer: Footer,
    /// A factory for creating segment sources that read data from the file.
    pub(crate) segment_source_factory: Arc<dyn SegmentSourceFactory>,
    /// Metrics tied to the file.
    pub(crate) metrics: VortexMetrics,
}

impl VortexFile {
    /// Returns a reference to the file's footer, which contains metadata and layout information.
    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.footer.row_count()
    }

    /// Returns the data type of the file's contents.
    pub fn dtype(&self) -> &DType {
        self.footer.dtype()
    }

    /// Returns the file's statistics, if available.
    ///
    /// Statistics can be used for query optimization and data exploration.
    pub fn file_stats(&self) -> Option<&Arc<[StatsSet]>> {
        self.footer.statistics()
    }

    /// Returns a reference to the file's metrics.
    pub fn metrics(&self) -> &VortexMetrics {
        &self.metrics
    }

    /// Create a new segment source for reading from the file.
    ///
    /// This may spawn a background I/O driver that will exit when the returned segment source
    /// is dropped.
    pub fn segment_source(&self) -> Arc<dyn SegmentSource> {
        self.segment_source_factory
            .segment_source(self.metrics.clone())
    }

    /// Create a new layout reader for the file.
    pub fn layout_reader(&self) -> VortexResult<Arc<dyn LayoutReader>> {
        let segment_source = self.segment_source();
        self.footer
            .layout()
            // TODO(ngates): we may want to allow the user pass in a name here?
            .new_reader("".into(), segment_source, self.footer().ctx().clone())
    }

    /// Initiate a scan of the file, returning a builder for configuring the scan.
    pub fn scan(&self) -> VortexResult<ScanBuilder<ArrayRef>> {
        Ok(ScanBuilder::new(self.layout_reader()?).with_metrics(self.metrics.clone()))
    }

    /// Returns true if the expression will never match any rows in the file.
    pub fn can_prune(&self, filter: &ExprRef, file_idx: u64) -> VortexResult<bool> {
        let Some((stats, fields)) = self
            .footer
            .statistics()
            .zip(self.footer.dtype().as_struct())
        else {
            return Ok(false);
        };

        let set = FieldPathSet::from_iter(fields.names().iter().zip(stats.iter()).flat_map(
            |(name, stats)| {
                stats.iter().map(|(stat, _)| {
                    FieldPath::from_iter([
                        Field::Name(name.clone()),
                        Field::Name(stat.name().into()),
                    ])
                })
            },
        ));

        let mut scope_set = ScopeFieldPathSet::new(set);
        let row_id = Identifier::Other(Arc::from("row_id"));
        scope_set = scope_set.with_set(
            row_id.clone(),
            FieldPathSet::from_iter([
                FieldPath::from_iter([
                    Field::Name("file_row_number".into()),
                    Field::Name(Stat::Max.name().into()),
                ]),
                FieldPath::from_iter([
                    Field::Name("file_row_number".into()),
                    Field::Name(Stat::Min.name().into()),
                ]),
                FieldPath::from_iter([
                    Field::Name("file_index".into()),
                    Field::Name(Stat::Max.name().into()),
                ]),
                FieldPath::from_iter([
                    Field::Name("file_index".into()),
                    Field::Name(Stat::Min.name().into()),
                ]),
            ]),
        );

        let Some((predicate, required_stats)) = checked_pruning_expr(filter, &scope_set) else {
            return Ok(false);
        };

        let required_file_stats =
            HashMap::from_iter(required_stats.map().iter().filter_map(|(path, stats)| {
                if path.identifier().is_identity() {
                    Some((path.field_path().clone(), stats.clone()))
                } else {
                    None
                }
            }));

        let Some(file_stats) =
            extract_relevant_file_stat_as_struct_row(&required_file_stats, stats, fields)?
        else {
            return Ok(false);
        };

        let mut scope = Scope::new(file_stats);

        let file_idx = file_idx;
        let file_len = self.row_count();
        scope = scope.with_array(
            row_id,
            StructArray::from_fields(&[
                (
                    "file_row_number_max",
                    ConstantArray::new(file_len, 1).to_array(),
                ),
                (
                    "file_row_number_min",
                    ConstantArray::new(0u64, 1).to_array(),
                ),
                ("file_index_max", ConstantArray::new(file_idx, 1).to_array()),
                ("file_index_min", ConstantArray::new(file_idx, 1).to_array()),
            ])
            .unwrap()
            .to_array(),
        );

        // println!(
        //     "--\nprune filter {}\n expr pred {}\n, res {}\n--",
        //     filter,
        //     predicate,
        //     predicate.evaluate(&scope).unwrap().as_constant().unwrap()
        // );

        Ok(predicate
            .evaluate(&scope)?
            .as_constant()
            .is_some_and(|result| result.as_bool().value() == Some(true)))
    }
}

/// A factory for creating segment sources that read data from a Vortex file.
///
/// This trait abstracts over different implementations of segment sources, allowing
/// for different I/O strategies (e.g., synchronous, asynchronous, memory-mapped)
/// to be used with the same file interface.
pub trait SegmentSourceFactory: 'static + Send + Sync {
    /// Create a segment source for reading segments from the file.
    ///
    /// # Arguments
    ///
    /// * `metrics` - Metrics for monitoring the performance of the segment source.
    ///
    /// # Returns
    ///
    /// A new segment source that can be used to read data from the file.
    fn segment_source(&self, metrics: VortexMetrics) -> Arc<dyn SegmentSource>;
}
