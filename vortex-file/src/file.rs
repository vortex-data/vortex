// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the [`VortexFile`] struct, which represents a Vortex file on disk or in memory.
//!
//! The `VortexFile` provides methods for accessing file metadata, creating segment sources for reading
//! data from the file, and initiating scans to read the file's contents into memory as Vortex arrays.

use std::sync::Arc;

use vortex_array::stats::StatsSet;
use vortex_array::ArrayRef;
use vortex_dtype::{DType, Field, FieldPath, FieldPathSet};
use vortex_error::VortexResult;
use vortex_expr::pruning::checked_pruning_expr;
use vortex_expr::{ExprRef, Scope};
use vortex_io::runtime::{FileIo, Handle};
use vortex_layout::segments::SegmentSourceRef;
use vortex_layout::LayoutReaderRef;
use vortex_metrics::VortexMetrics;
use vortex_scan::ScanBuilder;
use vortex_utils::aliases::hash_map::HashMap;

use crate::footer::Footer;
use crate::pruning::extract_relevant_file_stats_as_struct_row;
use crate::segments::FileSegmentSource;

/// Represents a Vortex file, providing access to its metadata and content.
///
/// A `VortexFile` is created by opening a Vortex file using [`VortexOpenOptions`](crate::VortexOpenOptions).
/// It provides methods for accessing file metadata (such as row count, data type, and statistics)
/// and for initiating scans to read the file's contents.
#[derive(Clone)]
pub struct VortexFile<'rt> {
    /// The footer of the Vortex file, containing metadata and layout information.
    pub(crate) footer: Footer,
    /// The file containing the segments.
    pub(crate) file: FileIo<'rt>,
    /// Metrics tied to the file.
    pub(crate) metrics: VortexMetrics,
    /// The handle to use for I/O operations.
    /// FIXME(ngates): this shoud have a lifetime? Then the user should be encouraged to stash
    ///  the footer if they care about cheap re-opening of a VortexFile.
    pub(crate) handle: Handle<'rt>,
}

impl<'rt> VortexFile<'rt> {
    /// Returns a reference to the file's footer, which contains metadata and layout information.
    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    /// Consumes the `VortexFile`, returning its footer.
    pub fn into_footer(self) -> Footer {
        self.footer
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
    pub fn segment_source(&self) -> SegmentSourceRef<'rt> {
        Arc::new(FileSegmentSource::new(
            self.footer.segment_map().clone(),
            self.file.clone(),
        ))
    }

    /// Create a new layout reader for the file.
    pub fn layout_reader(&self) -> VortexResult<LayoutReaderRef<'rt>> {
        self.footer
            .layout()
            // TODO(ngates): we may want to allow the user pass in a name here?
            .new_reader("".into(), self.segment_source(), self.handle.clone())
    }

    /// Initiate a scan of the file, returning a builder for configuring the scan.
    pub fn scan(&self) -> VortexResult<ScanBuilder<'rt, ArrayRef>> {
        Ok(ScanBuilder::new(self.layout_reader()?, self.handle.clone())
            .with_metrics(self.metrics.clone()))
    }

    /// Returns true if the expression will never match any rows in the file.
    pub fn can_prune(&self, filter: &ExprRef) -> VortexResult<bool> {
        let Some((stats, fields)) = self
            .footer
            .statistics()
            .zip(self.footer.dtype().as_struct_fields_opt())
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

        let Some((predicate, required_stats)) = checked_pruning_expr(filter, &set) else {
            return Ok(false);
        };

        let required_file_stats = HashMap::from_iter(
            required_stats
                .map()
                .iter()
                .map(|(path, stats)| (path.clone(), stats.clone())),
        );

        let Some(file_stats) =
            extract_relevant_file_stats_as_struct_row(&required_file_stats, stats, fields)?
        else {
            return Ok(false);
        };

        let scope = Scope::new(file_stats);

        Ok(predicate
            .evaluate(&scope)?
            .as_constant()
            .is_some_and(|result| result.as_bool().value() == Some(true)))
    }
}
