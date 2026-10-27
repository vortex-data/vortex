// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The planner-driven scan executor.
//!
//! A V2 scan lowers the file's layout once into a physical plan and runs it through the planning
//! protocol: an engine composes planners over a [`PreparedScan`], a [`PrunePlanner`] into a
//! [`ChunkPlanner`], admits the first to a [`Run`](vortex_scan::planning::driver::Run) of its own
//! driver, and every segment read is a protocol request served from the file's
//! [`SegmentSource`]. The layout reader is never asked to evaluate anything, and this crate never
//! decides where compute runs or how IO is awaited: that is the engine's driver.
//!
//! Import this module's [`ScanBuilder`] to prepare a V2 scan. It accepts a [`ScanFile`] directly
//! and supports projection, filters, row ranges, and selections. Unsupported layouts return an
//! error when the plan is prepared; limits are not part of this builder's API. Work is cut into
//! chunks of up to 262,144 rows, or up to 1,048,576 rows for filtered scans, independently of the
//! stored layout's chunk boundaries. Filtered scans use smaller chunks when the projection reads
//! many bytes per row or the row range would give fewer chunks than threads. Full-file
//! projections use smaller 8,192-row multiples when serialized bytes per row are high.
//!
//! A [`ScanFile`] carrying a [`ScanCache`] shares its lowered plan, filter plans and pruned
//! zones, segment index, dictionary values, and recently read segments with every other scan of
//! that file, like a cached layout reader does for V1 scans.

mod cache;
pub(crate) mod io;
mod retain;
#[cfg(test)]
mod tests;

use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

pub use self::cache::ScanCache;
pub use self::cache::WeakScanCache;
use crate::LayoutRef;
use crate::plan::physical::lower;
use crate::plan::physical::optimize;
use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::plan_row_idx_expression;
pub use crate::scan::execution::ChunkPlanner;
pub use crate::scan::execution::FilterPlans;
use crate::scan::execution::MAX_ROWS_PER_CHUNK;
pub use crate::scan::execution::PreparedScan;
pub use crate::scan::execution::PruneNext;
pub use crate::scan::execution::PrunePlanner;
use crate::scan::execution::ROW_BLOCK_SIZE;
pub use crate::scan::execution::SegmentLocation;
use crate::scan::execution::filtered_rows_per_chunk;
use crate::scan::execution::rows_per_chunk;
use crate::scan::execution::total_bytes;
use crate::scan::v2::io::segment_ranges;
use crate::segments::SegmentSource;

const MIN_PARTITION_ROWS: usize = 8 * ROW_BLOCK_SIZE;

/// The rows a byte range of the file stands for when an engine splits the file by bytes before
/// opening it. Consecutive byte ranges give consecutive row ranges that tile the file, with
/// boundaries on 65,536-row blocks. Keeping small files in fewer partitions avoids repeatedly
/// slicing the same encoded buffers into sparse batches.
pub fn partition_row_range(bytes: Range<u64>, file_size: u64, row_count: u64) -> Range<u64> {
    let row = |byte: u64| -> u64 {
        if byte >= file_size {
            return row_count;
        }
        let row = u128::from(byte) * u128::from(row_count) / u128::from(file_size);
        let row = u64::try_from(row).vortex_expect("a row of the file fits u64");
        row / MIN_PARTITION_ROWS as u64 * MIN_PARTITION_ROWS as u64
    };
    let start = row(bytes.start);
    start..row(bytes.end).max(start)
}

/// The parts of a file the planner-driven executor reads through.
#[derive(Clone)]
pub struct ScanFile {
    /// The root layout to scan.
    pub layout: LayoutRef,
    /// The location of every segment, indexed by segment id.
    pub locations: Arc<[SegmentLocation]>,
    /// Fetches segment bytes by id.
    pub segments: Arc<dyn SegmentSource>,
    /// State shared by every scan of this file, or `None` to rebuild it for each scan.
    pub cache: Option<ScanCache>,
}

/// Prepares a scan on the planner-driven V2 executor.
///
/// This builder reads the file's layout and segments directly and plans the scan without IO.
/// Driving the plan is the engine's: see [`PreparedScan`]. Row limits are not supported.
pub struct ScanBuilder {
    session: VortexSession,
    file: ScanFile,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    row_range: Option<Range<u64>>,
    selection: Selection,
    row_offset: u64,
    prefetch_batches: usize,
}

impl ScanBuilder {
    /// Creates a V2 scan over the file's layout and segment source.
    pub fn new(session: VortexSession, file: ScanFile) -> Self {
        Self {
            projection: BoundExpression::new_root(file.layout.dtype().clone()),
            filter: None,
            session,
            file,
            row_range: None,
            selection: Selection::default(),
            row_offset: 0,
            prefetch_batches: 1,
        }
    }

    /// The output dtype after projection.
    pub fn dtype(&self) -> &DType {
        self.projection.dtype()
    }

    /// The session the scan decodes and evaluates expressions with.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Sets a projection bound against the file's dtype.
    pub fn with_projection(mut self, projection: BoundExpression) -> Self {
        self.projection = projection;
        self
    }

    /// Evaluates a bound predicate before projecting the surviving rows.
    pub fn with_some_filter(mut self, filter: Option<BoundExpression>) -> Self {
        self.filter = filter;
        self
    }

    /// Restricts scanning to a contiguous file-local row range.
    pub fn with_row_range(mut self, row_range: Range<u64>) -> Self {
        self.row_range = Some(row_range);
        self
    }

    /// Applies a row selection within the scan's row range.
    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// Selects rows by strictly sorted file-local indices.
    pub fn with_row_indices(self, row_indices: StrictSortedBuffer<u64>) -> Self {
        self.with_selection(Selection::IncludeByIndex(row_indices))
    }

    /// Sets the root row offset used by row-index expressions.
    pub fn with_row_offset(mut self, row_offset: u64) -> Self {
        self.row_offset = row_offset;
        self
    }

    /// Groups up to four output batches into one unfiltered projection read window.
    /// Drivers running one chunk at a time can widen IO discovery without changing output sizes.
    pub fn with_prefetch_batches(mut self, batches: usize) -> Self {
        self.prefetch_batches = batches.clamp(1, 4);
        self
    }

    /// Lowers the file's layout and plans the filter and the projection, from the file's cache
    /// where an earlier scan planned them already. Touches no segment. Lowering and optimizing a
    /// wide file is real CPU, so a driver runs this where it runs compute.
    pub fn prepare(self) -> VortexResult<PreparedScan> {
        let file = self.file;
        let cache = file.cache.unwrap_or_default();
        let root = match cache.root() {
            Some(root) => root,
            None => cache.insert_root(lower(&file.layout)?),
        };
        let row_count = root.row_count();
        let filter = self
            .filter
            .map(|filter| -> VortexResult<_> {
                match cache.filter_plans(&filter) {
                    Some(plans) => Ok(plans),
                    None => {
                        let plans =
                            Arc::new(FilterPlans::new(filter.clone(), &root, &self.session)?);
                        Ok(cache.insert_filter_plans(filter, plans))
                    }
                }
            })
            .transpose()?;
        // Optimizing drops the zone tables a projection never reads and keeps the chunks of a
        // chunked column lazy, so an identity projection is optimized like any other.
        let identity = self.projection.is_root() && self.projection.dtype() == root.dtype();
        let projection = match cache.projection(&self.projection) {
            Some(plan) => plan,
            None => {
                let plan = optimize(
                    plan_row_idx_expression(self.projection.clone(), root)?,
                    &self.session,
                )?;
                cache.insert_projection(self.projection.clone(), plan)
            }
        };
        let rows = self.row_range.unwrap_or(0..row_count);
        let chunk_rows = if filter.is_some() {
            let projected_bytes = if identity {
                total_bytes(&file.locations)
            } else {
                cache.projection_bytes(&self.projection, || {
                    let segments = Scan::segments(
                        &self.session,
                        &projection,
                        0..row_count,
                        &Mask::new_true(usize::try_from(row_count)?),
                    )?;
                    Ok(segments
                        .iter()
                        .map(|segment| u64::from(file.locations[**segment as usize].length))
                        .sum())
                })?
            };
            filtered_rows_per_chunk(
                rows.end - rows.start,
                row_count,
                projected_bytes,
                get_available_parallelism().unwrap_or(1),
            )
        } else if identity {
            rows_per_chunk(row_count, &file.locations)
        } else if self.prefetch_batches > 1 && row_count > MAX_ROWS_PER_CHUNK as u64 {
            // The engine already supplies parallelism. Group read discovery across several
            // output morsels without creating fewer parallel tasks.
            self.prefetch_batches * MAX_ROWS_PER_CHUNK
        } else {
            MAX_ROWS_PER_CHUNK
        };
        let ranges = match cache.ranges() {
            Some(ranges) => ranges,
            None => cache.insert_ranges(segment_ranges(&file.locations)),
        };
        let values = cache.values();
        Ok(PreparedScan {
            session: self.session,
            rows,
            selection: self.selection,
            segments: cache.retaining(file.segments),
            ranges,
            cache,
            locations: file.locations,
            projection,
            filter,
            row_offset: self.row_offset,
            values,
            chunk_rows,
        })
    }
}
