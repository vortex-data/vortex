// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Preparation of source-independent scan plans.

use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use rustc_hash::FxHashSet;
use vortex_error::VortexResult;
use vortex_expr::analysis::referenced_field_paths;
use vortex_session::VortexSession;

pub use crate::execution::ChunkPlanner;
use crate::execution::FILTERED_ROWS_PER_CHUNK;
pub use crate::execution::FilterPlans;
use crate::execution::MAX_ROWS_PER_CHUNK;
pub use crate::execution::PreparedScan;
pub use crate::execution::PruneNext;
pub use crate::execution::PrunePlanner;
use crate::execution::ROW_BLOCK_SIZE;
use crate::execution::list_rows_per_chunk;
use crate::plan::Eval;
use crate::plan::ListPack;
use crate::plan::PlanByPtr;
use crate::plan::PlanRef;
pub use crate::execution::SegmentLocation;
pub use crate::execution::cache::ScanCache;
pub use crate::execution::cache::WeakScanCache;
use crate::execution::io::segment_ranges;
use crate::execution::rows_per_chunk;
use crate::plan::optimize;
use crate::plan::plan_row_idx_expression;
use crate::selection::Selection;
use crate::source::ScanInput;
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
        let filtered_chunk_rows = self
            .filter
            .as_ref()
            .map(|filter| filtered_rows_per_chunk(root.dtype(), &self.projection, filter))
            .transpose()?;
        let filter = self
            .filter
            .map(|filter| -> VortexResult<_> {
                let skip = self.skip_unselective_dynamic;
                match cache.filter_plans(&filter) {
                    Some(plans) if plans.skips_unselective_dynamic() == skip => Ok(plans),
                    // A scan whose skipping differs from the cached plans' plans its own.
                    Some(_) => Ok(Arc::new(FilterPlans::new(
                        filter,
                        &root,
                        &self.session,
                        skip,
                    )?)),
                    None => {
                        let plans = Arc::new(FilterPlans::new(
                            filter.clone(),
                            &root,
                            &self.session,
                            skip,
                        )?);
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
        let values = cache.values();
        let list_chunk_rows = list_chunk_rows(&projection, filter.as_deref(), row_count)?;
        let chunk_rows = if let Some(rows) = list_chunk_rows {
            rows
        } else if let Some(rows) = filtered_chunk_rows {
            rows
        } else if identity {
            rows_per_chunk(row_count, &file.locations)
        } else {
            MAX_ROWS_PER_CHUNK
        };
        let ranges = match cache.ranges() {
            Some(ranges) => ranges,
            None => cache.insert_ranges(segment_ranges(&file.locations)),
        };
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
        })
    }
}

/// The rows of a chunk of a scan whose projection or filter reads lists, sized by the elements
/// those lists hold over the file. `None` for a scan that reads no list.
fn list_chunk_rows(
    projection: &PlanRef,
    filter: Option<&FilterPlans>,
    row_count: u64,
) -> VortexResult<Option<usize>> {
    let mut lists = Vec::new();
    let mut seen = FxHashSet::default();
    let predicates = filter.into_iter().flat_map(|filter| filter.predicates());
    for plan in std::iter::once(projection).chain(predicates) {
        list_elements(plan, &mut seen, &mut lists)?;
    }
    if row_count == 0 || lists.is_empty() {
        return Ok(None);
    }
    Ok(Some(list_rows_per_chunk(row_count, lists.iter().sum())))
}

/// Collects the elements the lists of every [`ListPack`] under `plan` hold in all, each list
/// once, outer lists first.
fn list_elements(
    plan: &PlanRef,
    seen: &mut FxHashSet<PlanByPtr>,
    found: &mut Vec<u64>,
) -> VortexResult<()> {
    if plan.is_data_source() || !seen.insert(PlanByPtr(plan.clone())) {
        return Ok(());
    }
    // Walking a plan builds its children, the chunks of every column among them. Only an
    // expression produces a dtype without lists from a child with them, so nothing else whose
    // dtype holds no list is walked.
    if !plan.is::<Eval>() && !holds_list(plan.dtype()) {
        return Ok(());
    }
    if let Some(list) = plan.as_opt::<ListPack>() {
        // Several assemblies may share one list's offsets: they read its elements once.
        let elements = list.elements()?;
        if seen.insert(PlanByPtr(list.offsets()?)) {
            found.push(elements.row_count());
        }
        // Lists nest through the elements; offsets and validity hold none.
        return list_elements(&elements, seen, found);
    }
    for child in plan.children() {
        list_elements(child?, seen, found)?;
    }
    Ok(())
}

/// Whether a value of `dtype` holds a list at any depth.
fn holds_list(dtype: &DType) -> bool {
    match dtype {
        DType::List(..) | DType::FixedSizeList(..) => true,
        DType::Struct(fields, _) => fields.fields().any(|field| holds_list(&field)),
        DType::Extension(extension) => holds_list(extension.storage_dtype()),
        _ => false,
    }
}

pub(crate) fn filtered_rows_per_chunk(
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
