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
//! chunks of up to 65,536 rows, or 1,048,576 rows for filtered scans, independently of the stored
//! layout's chunk boundaries. When the file records its columns' uncompressed sizes, chunks shrink
//! to 8,192-row multiples that decode about 16 MiB of the columns the scan reads; otherwise only
//! full-file projections shrink, when serialized bytes per row are high.
//!
//! A [`ScanFile`] carrying a [`ScanCache`] shares its lowered plan, filter plans and pruned
//! zones, segment index, and dictionary values with every other scan of that file, like a cached
//! layout reader does for V1 scans.

mod cache;
pub(crate) mod io;
#[cfg(test)]
mod tests;

use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

pub use self::cache::ScanCache;
pub use self::cache::WeakScanCache;
use crate::LayoutRef;
use crate::plan::physical::lower;
use crate::plan::physical::optimize;
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
use crate::scan::execution::rows_for_uncompressed_bytes;
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
    /// The uncompressed bytes of each column the file records a size for, by field path, or
    /// `None` if it records none. A scan sizes its chunks by the columns it reads.
    pub column_bytes: Option<Arc<[(FieldPath, u64)]>>,
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
        let bytes_read = file.column_bytes.as_deref().and_then(|column_bytes| {
            read_bytes(
                column_bytes,
                self.filter.iter().chain(std::iter::once(&self.projection)),
            )
        });
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
                cache.insert_projection(self.projection, plan)
            }
        };
        let max_rows = if filter.is_some() {
            FILTERED_ROWS_PER_CHUNK
        } else {
            MAX_ROWS_PER_CHUNK
        };
        let chunk_rows = match bytes_read {
            Some(bytes) => rows_for_uncompressed_bytes(bytes, row_count, max_rows),
            None if identity && filter.is_none() => rows_per_chunk(row_count, &file.locations),
            None => max_rows,
        };
        let ranges = match cache.ranges() {
            Some(ranges) => ranges,
            None => cache.insert_ranges(segment_ranges(&file.locations)),
        };
        let values = cache.values();
        Ok(PreparedScan {
            session: self.session,
            rows: self.row_range.unwrap_or(0..row_count),
            selection: self.selection,
            segments: file.segments,
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

/// The uncompressed bytes of the columns `exprs` read, or `None` if one of them has no size in
/// `column_bytes`.
fn read_bytes<'a>(
    column_bytes: &[(FieldPath, u64)],
    exprs: impl IntoIterator<Item = &'a BoundExpression>,
) -> Option<u64> {
    let mut read = vec![false; column_bytes.len()];
    let mut pending: Vec<&BoundExpression> = exprs.into_iter().collect();
    while let Some(expr) = pending.pop() {
        let Some(paths) = accessed_paths(expr) else {
            pending.extend(expr.children());
            continue;
        };
        for path in paths {
            let mut found = false;
            for ((column, _), read) in column_bytes.iter().zip(&mut read) {
                if column.parts().starts_with(path.parts()) {
                    *read = true;
                    found = true;
                }
            }
            if !found {
                return None;
            }
        }
    }
    Some(
        column_bytes
            .iter()
            .zip(read)
            .filter(|(_, read)| *read)
            .map(|((_, bytes), _)| bytes)
            .sum(),
    )
}

/// The field paths `expr` reads when it is a chain of field accesses from the scope root, or a
/// selection of fields from one, or `None` if it computes something from its children.
fn accessed_paths(expr: &BoundExpression) -> Option<Vec<FieldPath>> {
    if let Some(selection) = expr.as_opt::<Select>() {
        let input = expr.child(0);
        let path = field_path(input)?;
        let Some(fields) = input
            .dtype()
            .as_struct_fields_opt()
            .and_then(|fields| selection.normalize_to_included_fields(fields.names()).ok())
        else {
            return Some(vec![path]);
        };
        return Some(
            fields
                .iter()
                .map(|name| path.clone().push(name.clone()))
                .collect(),
        );
    }
    field_path(expr).map(|path| vec![path])
}

/// The field path of a chain of field accesses from the scope root.
fn field_path(expr: &BoundExpression) -> Option<FieldPath> {
    let mut names = Vec::new();
    let mut node = expr;
    while let Some(name) = node.as_opt::<GetItem>() {
        names.push(Field::from(name.clone()));
        node = node.child(0);
    }
    node.is_root().then(|| names.into_iter().rev().collect())
}
