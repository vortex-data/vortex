// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native layout inputs for the planner-driven scan engine.

use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_expr::BoundExpression;
use vortex_scan::plan::LazyPlanRef;
pub use vortex_scan::prepare::ChunkPlanner;
pub use vortex_scan::prepare::FilterPlans;
pub use vortex_scan::prepare::PreparedScan;
pub use vortex_scan::prepare::PruneNext;
pub use vortex_scan::prepare::PrunePlanner;
use vortex_scan::prepare::ScanBuilder as EngineScanBuilder;
pub use vortex_scan::prepare::ScanCache;
pub use vortex_scan::prepare::SegmentLocation;
pub use vortex_scan::prepare::WeakScanCache;
use vortex_scan::selection::Selection;
use vortex_scan::source::ScanInput;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::segments::SegmentSource;

// Match the scan executor's row grid so byte partitions and scan chunks stay aligned.
const ROW_BLOCK_SIZE: usize = 8192;

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

impl From<ScanFile> for ScanInput {
    fn from(file: ScanFile) -> Self {
        Self {
            dtype: file.layout.dtype().clone(),
            root: LazyPlanRef::lazy(move || file.layout.new_plan()),
            locations: file.locations,
            segments: file.segments,
            cache: file.cache,
        }
    }
}

/// Prepares a scan of a native layout using the shared scan engine.
pub struct ScanBuilder {
    session: VortexSession,
    file: ScanFile,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    row_range: Option<Range<u64>>,
    selection: Selection,
    row_offset: u64,
    skip_unselective_dynamic: bool,
}

impl ScanBuilder {
    /// Creates a scan over the file's layout and segment source.
    pub fn new(session: VortexSession, file: ScanFile) -> Self {
        Self {
            projection: BoundExpression::new_root(file.layout.dtype().clone()),
            filter: None,
            session,
            file,
            row_range: None,
            selection: Selection::default(),
            row_offset: 0,
            skip_unselective_dynamic: false,
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

    /// Allows the scan to switch off a filter conjunct containing dynamic expressions once it has
    /// kept almost every row on several consecutive chunks, skipping its zone pruning and its
    /// evaluation until its dynamic values change.
    ///
    /// The scan may then return rows the conjunct would reject, so only enable this when the
    /// dynamic expressions are hints whose producer applies them again, such as DataFusion's
    /// TopK and hash-join dynamic filters. Off by default.
    pub fn with_skip_unselective_dynamic(mut self, skip: bool) -> Self {
        self.skip_unselective_dynamic = skip;
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

    /// Lowers the native layout and prepares the scan without reading payloads.
    pub fn prepare(self) -> VortexResult<PreparedScan> {
        self.into_scan_builder().prepare()
    }

    fn into_scan_builder(self) -> EngineScanBuilder {
        let mut builder = EngineScanBuilder::new(self.session, self.file)
            .with_projection(self.projection)
            .with_some_filter(self.filter)
            .with_selection(self.selection)
            .with_row_offset(self.row_offset)
            .with_skip_unselective_dynamic(self.skip_unselective_dynamic);
        if let Some(rows) = self.row_range {
            builder = builder.with_row_range(rows);
        }
        builder
    }
}

impl From<ScanBuilder> for EngineScanBuilder {
    fn from(builder: ScanBuilder) -> Self {
        builder.into_scan_builder()
    }
}

#[cfg(test)]
mod tests;
