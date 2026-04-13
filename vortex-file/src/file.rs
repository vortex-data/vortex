// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the [`VortexFile`] struct, which represents a Vortex file on disk or in memory.
//!
//! The `VortexFile` provides methods for accessing file metadata, creating segment sources for reading
//! data from the file, and initiating scans to read the file's contents into memory as Vortex arrays.

use std::ops::Range;
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::FieldPathSet;
use vortex_array::expr::Expression;
use vortex_array::expr::pruning::checked_pruning_expr;
use vortex_error::VortexResult;
use vortex_layout::LayoutReader;
use vortex_layout::scan::layout::LayoutReaderDataSource;
use vortex_layout::scan::scan_builder::ScanBuilder;
use vortex_layout::scan::split_by::SplitBy;
use vortex_layout::segments::SegmentSource;
use vortex_scan::DataSourceRef;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::FileStatistics;
use crate::footer::Footer;
use crate::pruning::extract_relevant_file_stats_as_struct_row;
use crate::v2::FileStatsLayoutReader;

/// Represents a Vortex file, providing access to its metadata and content.
///
/// A `VortexFile` is created by opening a Vortex file using [`VortexOpenOptions`](crate::VortexOpenOptions).
/// It provides methods for accessing file metadata (such as row count, data type, and statistics)
/// and for initiating scans to read the file's contents.
#[derive(Clone)]
pub struct VortexFile {
    /// The footer of the Vortex file, containing metadata and layout information.
    pub(crate) footer: Footer,
    /// The segment source used to read segments from this file.
    pub(crate) segment_source: Arc<dyn SegmentSource>,
    /// The Vortex session used to open this file
    pub(crate) session: VortexSession,
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
    pub fn file_stats(&self) -> Option<&FileStatistics> {
        self.footer.statistics()
    }

    /// Create a new segment source for reading from the file.
    ///
    /// This may spawn a background I/O driver that will exit when the returned segment source
    /// is dropped.
    pub fn segment_source(&self) -> Arc<dyn SegmentSource> {
        Arc::clone(&self.segment_source)
    }

    /// Create a new layout reader for the file.
    pub fn layout_reader(&self) -> VortexResult<Arc<dyn LayoutReader>> {
        let segment_source = self.segment_source();
        self.footer
            .layout()
            // TODO(ngates): we may want to allow the user pass in a name here?
            .new_reader("".into(), segment_source, &self.session)
    }

    /// Create a [`DataSource`](vortex_scan::DataSource) from this file for scanning.
    ///
    /// Wraps the file's layout reader with [`FileStatsLayoutReader`] (when file-level
    /// statistics are available) and [`LayoutReaderDataSource`].
    pub fn data_source(&self) -> VortexResult<DataSourceRef> {
        let mut reader = self.layout_reader()?;
        if let Some(stats) = self.file_stats().cloned() {
            reader = Arc::new(FileStatsLayoutReader::new(
                reader,
                stats,
                self.session.clone(),
            ));
        }
        Ok(Arc::new(LayoutReaderDataSource::new(
            reader,
            self.session.clone(),
        )))
    }

    /// Initiate a scan of the file, returning a builder for configuring the scan.
    pub fn scan(&self) -> VortexResult<ScanBuilder<ArrayRef>> {
        Ok(ScanBuilder::new(
            self.session.clone(),
            self.layout_reader()?,
        ))
    }

    /// Returns true if the expression will never match any rows in the file.
    pub fn can_prune(&self, filter: &Expression) -> VortexResult<bool> {
        let Some((stats, fields)) = self
            .footer
            .statistics()
            .zip(self.footer.dtype().as_struct_fields_opt())
        else {
            return Ok(false);
        };

        let set = FieldPathSet::from_iter(
            fields
                .names()
                .iter()
                .zip(stats.stats_sets().iter())
                .flat_map(|(name, stats)| {
                    stats.iter().map(|(stat, _)| {
                        FieldPath::from_iter([
                            Field::Name(name.clone()),
                            Field::Name(stat.name().into()),
                        ])
                    })
                }),
        );

        let Some((predicate, required_stats)) = checked_pruning_expr(filter, &set) else {
            return Ok(false);
        };

        let required_file_stats = HashMap::from_iter(
            required_stats
                .map()
                .iter()
                .map(|(path, stats)| (path.clone(), stats.clone())),
        );

        let Some(file_stats) = extract_relevant_file_stats_as_struct_row(
            &required_file_stats,
            stats.stats_sets(),
            fields,
        )?
        else {
            return Ok(false);
        };

        let mut ctx = self.session.create_execution_ctx();
        Ok(
            match file_stats
                .apply(&predicate)?
                .execute::<Columnar>(&mut ctx)?
            {
                Columnar::Constant(s) => s.scalar().as_bool().value() == Some(true),
                Columnar::Canonical(_) => false,
            },
        )
    }

    pub fn splits(&self) -> VortexResult<Vec<Range<u64>>> {
        let reader = self.layout_reader()?;
        Ok(SplitBy::Layout
            .splits(reader.as_ref(), &(0..reader.row_count()), &[FieldMask::All])?
            .into_iter()
            .tuple_windows()
            .map(|(start, end)| start..end)
            .collect())
    }
}
