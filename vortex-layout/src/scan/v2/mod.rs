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
//! error when the plan is prepared; limits are not part of this builder's API. Work uses chunks of
//! up to 1,048,576 rows, or 131,072 rows for filtered scans. Filtered scans reading only one
//! primitive or boolean field use 1,048,576 rows to amortize setup. Chunks are independent of
//! stored layout boundaries. Unfiltered projections stream stored batches without concatenating
//! them. Full-file projections use smaller 8,192-row multiples when serialized bytes per row
//! are high.
//!
//! Sparse positive input selections use [`AccessMode::RandomAccess`]: only selected 65,536-row
//! regions are scheduled, and segment leaves gather encoded values before materialization.
//! Dense inputs retain [`AccessMode::Scan`]. [`ScanBuilder::with_access_mode`] overrides the
//! automatic choice. The access mode does not change the rows selected or their order.
//!
//! Engine drivers share these selection and predicate policies: sparse string predicates gather
//! referenced dictionary entries and cache their results, while dense scans and numeric
//! dictionary predicates retain bulk evaluation. Drivers supply scheduling and IO, rather than
//! their own selection heuristics.
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
use vortex_expr::analysis::referenced_field_paths;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

pub use self::cache::ScanCache;
pub use self::cache::WeakScanCache;
use crate::LayoutRef;
use crate::plan::physical::lower;
use crate::plan::physical::optimize;
pub use crate::plan::physical::pipeline::AccessMode;
use crate::plan::physical::plan_row_idx_expression;
pub use crate::scan::execution::ChunkPlanner;
use crate::scan::execution::FILTERED_ROWS_PER_CHUNK;
pub use crate::scan::execution::FilterPlans;
use crate::scan::execution::MAX_ROWS_PER_CHUNK;
pub use crate::scan::execution::PreparedScan;
pub use crate::scan::execution::PruneNext;
pub use crate::scan::execution::PrunePlanner;
use crate::scan::execution::ROW_BLOCK_SIZE;
pub use crate::scan::execution::SegmentLocation;
use crate::scan::execution::access_mode;
use crate::scan::execution::rows_per_chunk;
use crate::scan::v2::io::segment_ranges;
use crate::segments::SegmentSource;

/// The rows a byte range of the file stands for when an engine splits the file by bytes before
/// opening it. Consecutive byte ranges give consecutive row ranges that tile the file, with
/// boundaries on the grid's 8,192-row blocks so the executor's chunks stay block aligned.
pub fn partition_row_range(bytes: Range<u64>, file_size: u64, row_count: u64) -> Range<u64> {
    let row = |byte: u64| -> u64 {
        if byte >= file_size {
            return row_count;
        }
        let row = u128::from(byte) * u128::from(row_count) / u128::from(file_size);
        let row = u64::try_from(row).vortex_expect("a row of the file fits u64");
        row / ROW_BLOCK_SIZE as u64 * ROW_BLOCK_SIZE as u64
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
    access_mode: Option<AccessMode>,
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
            access_mode: None,
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

    /// Overrides automatic access selection. By default, sparse positive row selections use
    /// random access; other inputs stream and filter batches. Both modes keep row order.
    pub fn with_access_mode(mut self, access_mode: AccessMode) -> Self {
        self.access_mode = Some(access_mode);
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

    /// Lowers the file's layout and plans the filter and the projection, from the file's cache
    /// where an earlier scan planned them already. Touches no segment. Lowering and optimizing a
    /// wide file is real CPU, so a driver runs this where it runs compute.
    pub fn prepare(self) -> VortexResult<PreparedScan> {
        let file = self.file;
        let cache = file.cache.unwrap_or_default();
        let root = cache.root(|| lower(&file.layout))?;
        let row_count = root.row_count();
        let filtered_chunk_rows = self
            .filter
            .as_ref()
            .map(|filter| filtered_rows_per_chunk(root.dtype(), &self.projection, filter))
            .transpose()?;
        let filter = self
            .filter
            .map(|filter| {
                cache.filter_plans(&filter, || {
                    Ok(Arc::new(FilterPlans::new(
                        filter.clone(),
                        &root,
                        &self.session,
                    )?))
                })
            })
            .transpose()?;
        // Optimizing drops the zone tables a projection never reads and keeps the chunks of a
        // chunked column lazy, so an identity projection is optimized like any other.
        let identity = self.projection.is_root() && self.projection.dtype() == root.dtype();
        let projection = cache.projection(&self.projection, || {
            optimize(
                plan_row_idx_expression(self.projection.clone(), root)?,
                &self.session,
            )
        })?;
        let chunk_rows = if let Some(rows) = filtered_chunk_rows {
            rows
        } else if identity {
            rows_per_chunk(row_count, &file.locations)
        } else {
            MAX_ROWS_PER_CHUNK
        };
        let ranges = cache.ranges(|| segment_ranges(&file.locations));
        let values = cache.values();
        let rows = self.row_range.unwrap_or(0..row_count);
        let access_mode = self
            .access_mode
            .unwrap_or_else(|| access_mode(&self.selection, &rows));
        tracing::debug!(
            ?access_mode,
            selected_rows = self.selection.row_count_in_range(&rows),
            input_rows = rows.end.saturating_sub(rows.start),
            "v2 scan access mode"
        );
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
            access_mode,
        })
    }
}

fn filtered_rows_per_chunk(
    dtype: &DType,
    projection: &BoundExpression,
    filter: &BoundExpression,
) -> VortexResult<usize> {
    let mut fields = referenced_field_paths(projection)?;
    fields.extend(referenced_field_paths(filter)?);
    let mut fields = fields.into_iter();
    let first = fields.next();
    // A scan of one fixed-width field is cheap enough that smaller chunks mainly add setup.
    // Wider and variable-width scans need more chunks to avoid long tails on a few workers.
    let narrow = fields.next().is_none()
        && first
            .and_then(|field| field.resolve(dtype.clone()))
            .is_some_and(|dtype| matches!(dtype, DType::Primitive(..) | DType::Bool(..)));
    Ok(if narrow {
        128 * ROW_BLOCK_SIZE
    } else {
        FILTERED_ROWS_PER_CHUNK
    })
}
