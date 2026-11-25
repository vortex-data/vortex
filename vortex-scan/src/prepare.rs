// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Preparation of source-independent scan plans.

use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

pub use crate::execution::ChunkPlanner;
pub use crate::execution::FilterPlans;
pub use crate::execution::FilterPrep;
pub use crate::execution::PreparedScan;
pub use crate::execution::PruneNext;
pub use crate::execution::PrunePlanner;
pub use crate::execution::SegmentLocation;
pub use crate::execution::SharedFilterPrep;
pub use crate::execution::cache::ScanCache;
pub use crate::execution::cache::WeakScanCache;
use crate::execution::io::segment_ranges;
use crate::execution::rows_per_chunk;
use crate::plan::optimize;
use crate::plan::plan_row_idx_expression;
use crate::selection::Selection;
use crate::source::ScanInput;
use crate::split::part_rows;
use crate::strict_sorted_buffer::StrictSortedBuffer;

/// Prepares a scan on the planner-driven executor.
///
/// This builder plans a source without reading its payloads.
/// Driving the plan is the engine's: see [`PreparedScan`]. Row limits are not supported.
pub struct ScanBuilder {
    session: VortexSession,
    file: ScanInput,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    filter_prep: Option<Arc<SharedFilterPrep>>,
    row_range: Option<Range<u64>>,
    selection: Selection,
    row_offset: u64,
    skip_unselective_dynamic: bool,
}

impl ScanBuilder {
    /// Creates a scan over a lazily lowered plan and its byte source.
    pub fn new(session: VortexSession, file: impl Into<ScanInput>) -> Self {
        let file = file.into();
        Self {
            projection: BoundExpression::new_root(file.dtype.clone()),
            filter: None,
            filter_prep: None,
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

    /// Shares the [`FilterPrep`] of the scan's filter with the scans of other files of the same
    /// schema, so a scan whose file has not planned the filter yet does not rewrite it again.
    pub fn with_filter_prep(mut self, prep: Arc<SharedFilterPrep>) -> Self {
        self.filter_prep = Some(prep);
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

    /// Restricts scanning to a contiguous source-local row range. A range that starts after it
    /// ends selects no rows.
    pub fn with_row_range(mut self, row_range: Range<u64>) -> Self {
        self.row_range = Some(row_range.start..row_range.end.max(row_range.start));
        self
    }

    /// Applies a row selection within the scan's row range.
    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// Selects rows by strictly sorted source-local indices.
    pub fn with_row_indices(self, row_indices: StrictSortedBuffer<u64>) -> Self {
        self.with_selection(Selection::IncludeByIndex(row_indices))
    }

    /// Sets the root row offset used by row-index expressions.
    pub fn with_row_offset(mut self, row_offset: u64) -> Self {
        self.row_offset = row_offset;
        self
    }

    /// Initializes the source plan and plans the filter and the projection, from the file's cache
    /// where an earlier scan planned them already. Touches no segment. Lowering and optimizing a
    /// wide file is real CPU, so a driver runs this where it runs compute.
    pub fn prepare(self) -> VortexResult<PreparedScan> {
        let file = self.file;
        let cache = file.cache.unwrap_or_default();
        let root = match cache.root() {
            Some(root) => root,
            None => cache.insert_root(file.root.get()?.clone()),
        };
        let row_count = root.row_count();
        let part_rows = part_rows(root.dtype(), &self.projection, self.filter.as_ref())?;
        let filter = self
            .filter
            .map(|filter| -> VortexResult<_> {
                let skip = self.skip_unselective_dynamic;
                let session = &self.session;
                let plan = |filter: &BoundExpression| -> VortexResult<Arc<FilterPlans>> {
                    let prep = match &self.filter_prep {
                        Some(prep) => prep.get_or_try_init(filter, session)?,
                        None => Arc::new(FilterPrep::new(filter.clone(), session)?),
                    };
                    Ok(Arc::new(FilterPlans::new(&prep, &root, session, skip)?))
                };
                match cache.filter_plans(&filter) {
                    Some(plans) if plans.skips_unselective_dynamic() == skip => Ok(plans),
                    // A scan whose skipping differs from the cached plans' plans its own.
                    Some(_) => plan(&filter),
                    None => {
                        let plans = plan(&filter)?;
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
                    plan_row_idx_expression(self.projection.clone(), root, &self.session)?,
                    &self.session,
                )?;
                cache.insert_projection(self.projection, plan)
            }
        };
        // A part is one chunk, except that an identity projection's chunks are also bounded by
        // their bytes, which only this engine knows, so its parts can hold several chunks.
        let chunk_rows = if identity && filter.is_none() {
            rows_per_chunk(row_count, &file.locations)
        } else {
            usize::try_from(part_rows)?
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
            segments: cache.retaining(file.segments),
            ranges,
            cache,
            locations: file.locations,
            projection,
            filter,
            row_offset: self.row_offset,
            values,
            chunk_rows,
            part_rows,
        })
    }
}
