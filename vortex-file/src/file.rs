// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the [`VortexFile`] struct, which represents a Vortex file on disk or in memory.
//!
//! The `VortexFile` provides methods for accessing file metadata, creating segment sources for reading
//! data from the file, and initiating scans to read the file's contents into memory as Vortex arrays.

use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_layout::LayoutReader;
use vortex_layout::scan::layout::LayoutReaderDataSource;
use vortex_layout::scan::scan_builder::ScanBuilder;
use vortex_layout::scan::split_by::SplitBy;
use vortex_layout::segments::ScheduledSegmentSource;
use vortex_layout::segments::ScheduledSegmentSourceAdapter;
use vortex_layout::segments::SegmentInfo;
use vortex_layout::segments::SegmentSource;
use vortex_scan::DataSourceRef;
use vortex_scan::ScanRequest;
use vortex_scan::plan::PreparedStateCache;
use vortex_scan::plan::PreparedStateCacheRef;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::segments::SegmentFutureCache;
use vortex_session::VortexSession;

use crate::FileStatistics;
use crate::footer::Footer;
use crate::multi::scan_v2;
use crate::pruning::can_prune_file_stats;
use crate::v2::FileStatsLayoutReader;

/// Represents a Vortex file, providing access to its metadata and content.
///
/// A `VortexFile` is created by opening a Vortex file using [`VortexOpenOptions`](crate::VortexOpenOptions).
/// It provides methods for accessing file metadata (such as row count, data type, and statistics)
/// and for initiating scans to read the file's contents.
#[derive(Clone)]
pub struct VortexFile {
    /// The footer of the Vortex file, containing metadata and layout information.
    footer: Footer,
    /// The segment source used to read segments from this file.
    segment_source: Arc<dyn SegmentSource>,
    /// Scheduled view of the same segment source.
    scheduled_segment_source: Arc<dyn ScheduledSegmentSource>,
    /// The Vortex session used to open this file.
    session: VortexSession,
    /// None id LayoutReader caching is turned off
    layout_reader_cache: Option<OnceLock<Arc<dyn LayoutReader>>>,
    /// Shared cache for the v2 physical scan plan root.
    scan_plan_root_cache: Arc<OnceLock<ScanPlanRef>>,
    /// Shared cache for v2 prepared state across row-range scans of this file.
    scan_plan_state_cache: PreparedStateCacheRef,
    /// Shared cache for v2 in-flight segment futures across row-range scans of this file.
    scan_plan_segment_future_cache: Arc<SegmentFutureCache>,
}

fn layout_reader(
    segment_source: Arc<dyn SegmentSource>,
    footer: &Footer,
    session: &VortexSession,
) -> VortexResult<Arc<dyn LayoutReader>> {
    let root_reader = footer
        .layout()
        // TODO(ngates): we may want to allow the user pass in a name here?
        .new_reader("".into(), segment_source, session, &Default::default())?;

    Ok(if let Some(stats) = footer.statistics().cloned() {
        Arc::new(FileStatsLayoutReader::new(
            root_reader,
            stats,
            session.clone(),
        ))
    } else {
        root_reader
    })
}

impl VortexFile {
    /// Creates a new `VortexFile` from the given footer, segment source, and session.
    pub fn new(
        footer: Footer,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> Self {
        let segment_infos: Arc<[SegmentInfo]> = footer
            .segment_map()
            .iter()
            .map(|segment| SegmentInfo::cacheable(u64::from(segment.length)))
            .collect::<Vec<_>>()
            .into();
        let scheduled_segment_source: Arc<dyn ScheduledSegmentSource> = Arc::new(
            ScheduledSegmentSourceAdapter::new(Arc::clone(&segment_source), segment_infos),
        );
        Self {
            footer,
            segment_source,
            scheduled_segment_source,
            session,
            layout_reader_cache: None,
            scan_plan_root_cache: Arc::new(OnceLock::new()),
            scan_plan_state_cache: Arc::new(PreparedStateCache::default()),
            scan_plan_segment_future_cache: Arc::new(SegmentFutureCache::new()),
        }
    }

    /// Enable layout reader caching
    pub fn with_caching(self) -> Self {
        Self {
            footer: self.footer,
            segment_source: self.segment_source,
            scheduled_segment_source: self.scheduled_segment_source,
            session: self.session,
            layout_reader_cache: Some(OnceLock::new()),
            scan_plan_root_cache: self.scan_plan_root_cache,
            scan_plan_state_cache: self.scan_plan_state_cache,
            scan_plan_segment_future_cache: self.scan_plan_segment_future_cache,
        }
    }

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

    /// Return the scheduler-aware segment source for this file.
    pub fn scheduled_segment_source(&self) -> Arc<dyn ScheduledSegmentSource> {
        Arc::clone(&self.scheduled_segment_source)
    }

    /// Returns a reference to the Vortex session used to open this file.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Create a new layout reader for the file.
    ///
    /// Wraps the root layout in a [`FileStatsLayoutReader`] if file stats are available.
    pub fn layout_reader(&self) -> VortexResult<Arc<dyn LayoutReader>> {
        match &self.layout_reader_cache {
            None => layout_reader(
                Arc::clone(&self.segment_source),
                &self.footer,
                &self.session,
            ),
            Some(reader) => {
                // get_or_try_init is unstable
                if let Some(val) = reader.get() {
                    Ok(Arc::clone(val))
                } else {
                    let inner = layout_reader(
                        Arc::clone(&self.segment_source),
                        &self.footer,
                        &self.session,
                    )?;
                    Ok(if let Err(val) = reader.set(Arc::clone(&inner)) {
                        val
                    } else {
                        inner
                    })
                }
            }
        }
    }

    pub(crate) fn scan_plan_root(&self) -> VortexResult<ScanPlanRef> {
        if let Some(root) = self.scan_plan_root_cache.get() {
            return Ok(Arc::clone(root));
        }

        let root = scan_v2::build_file_scan_plan_root(self)?;
        if self.scan_plan_root_cache.set(Arc::clone(&root)).is_err()
            && let Some(root) = self.scan_plan_root_cache.get()
        {
            return Ok(Arc::clone(root));
        }
        Ok(root)
    }

    pub(crate) fn scan_plan_state_cache(&self) -> PreparedStateCacheRef {
        Arc::clone(&self.scan_plan_state_cache)
    }

    pub(crate) fn scan_plan_segment_future_cache(&self) -> Arc<SegmentFutureCache> {
        Arc::clone(&self.scan_plan_segment_future_cache)
    }

    /// Create a [`DataSource`](vortex_scan::DataSource) from this file for scanning.
    ///
    /// Wraps the file's layout reader with [`FileStatsLayoutReader`] (when file-level
    /// statistics are available) and [`LayoutReaderDataSource`].
    pub fn data_source(&self) -> VortexResult<DataSourceRef> {
        let reader = self.layout_reader()?;

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

    /// Execute a ScanPlan-backed scan for this file.
    pub fn scan_plan_stream(&self, request: ScanRequest) -> VortexResult<SendableArrayStream> {
        scan_v2::scan_plan_file_stream(self.clone(), request)
    }

    /// Return ScanPlan-backed aggregate-function statistics for this file.
    pub async fn scan_plan_statistics(
        &self,
        expr: &Expression,
        funcs: &[AggregateFnRef],
    ) -> VortexResult<Vec<Precision<Scalar>>> {
        scan_v2::scan_plan_file_statistics(self.clone(), expr, funcs).await
    }

    /// Return ScanPlan-backed aggregate-function statistics for several expressions in this file.
    pub async fn scan_plan_statistics_many(
        &self,
        exprs: &[Expression],
        funcs: &[AggregateFnRef],
    ) -> VortexResult<Vec<Vec<Precision<Scalar>>>> {
        scan_v2::scan_plan_file_statistics_many(self.clone(), exprs, funcs).await
    }

    /// Return ScanPlan natural row split ranges for this file.
    pub fn scan_plan_splits(&self) -> VortexResult<Vec<Range<u64>>> {
        scan_v2::scan_plan_file_splits(self)
    }

    /// Plan ScanPlan natural row split ranges for a projected scan of this file.
    pub async fn plan_splits(&self, projection: &Expression) -> VortexResult<Vec<Range<u64>>> {
        scan_v2::scan_plan_file_plan_splits(self.clone(), projection).await
    }

    /// Returns `true` if file-level statistics prove the expression cannot
    /// match any rows in this file.
    ///
    /// Row-count-aware pruning predicates are evaluated with the file's total
    /// row count as their scope.
    pub fn can_prune(&self, filter: &Expression) -> VortexResult<bool> {
        let Some((stats, fields)) = self
            .footer
            .statistics()
            .zip(self.footer.dtype().as_struct_fields_opt())
        else {
            return Ok(false);
        };

        can_prune_file_stats(
            filter,
            self.footer.dtype(),
            self.footer.row_count(),
            stats,
            fields,
            &self.session,
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
