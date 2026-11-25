// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Preparation of source-independent scan plans.

use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::analysis::referenced_field_paths;
pub use crate::execution::ChunkPlanner;
use crate::execution::FILTERED_ROWS_PER_CHUNK;
pub use crate::execution::FilterPlans;
use crate::execution::MAX_ROWS_PER_CHUNK;
pub use crate::execution::PreparedScan;
pub use crate::execution::PruneNext;
pub use crate::execution::PrunePlanner;
use crate::execution::ROW_BLOCK_SIZE;
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
        let chunk_rows = if let Some(rows) = filtered_chunk_rows {
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
        })
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::Alignment;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_expr::Expression;
    use vortex_expr::get_item;
    use vortex_expr::gt;
    use vortex_expr::lit;
    use vortex_expr::root;

    use super::*;
    use crate::plan::LazyPlanRef;
    use crate::test::NoSegments;
    use crate::test::SESSION;
    use crate::test::Store;

    fn struct_input(store: &Arc<Store>, rows: usize) -> VortexResult<ScanInput> {
        let ints = PrimitiveArray::from_iter(0..i32::try_from(rows)?).into_array();
        let flags = BoolArray::from_iter((0..rows).map(|i| i % 2 == 0)).into_array();
        let array = StructArray::from_fields(&[("ints", ints), ("flags", flags)])?.into_array();
        let plan = store.flat(&array)?;
        Ok(store.input(plan))
    }

    #[test]
    fn builder_dtype_tracks_projection_and_row_range_clamps_inverted_bounds() -> VortexResult<()> {
        let store = Arc::new(Store::default());
        let input = struct_input(&store, 10)?;
        let root_dtype = input.dtype.clone();
        let builder = ScanBuilder::new(SESSION.clone(), input);
        assert_eq!(builder.dtype(), &root_dtype);

        let projection = get_item("ints", root()).bind(&root_dtype)?;
        let builder = builder
            .with_projection(projection.clone())
            .with_row_range(Range { start: 8, end: 3 })
            .with_row_offset(42)
            .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(vec![1u64, 4]))?);
        assert_eq!(builder.dtype(), projection.dtype());

        let prepared = builder.prepare()?;
        assert_eq!(prepared.rows(), &(8..8));
        assert_eq!(prepared.row_offset, 42);
        assert!(matches!(
            prepared.selection(),
            Selection::IncludeByIndex(indices) if indices.as_slice() == [1, 4]
        ));
        Ok(())
    }

    #[test]
    fn prepare_defaults_row_range_to_full_root_and_caches_lazy_root() -> VortexResult<()> {
        let store = Arc::new(Store::default());
        let base = struct_input(&store, 12)?;
        let plan = base.root.get()?.clone();
        let inits = Arc::new(AtomicUsize::new(0));
        let cache = ScanCache::default();

        let make_input = || {
            let plan = plan.clone();
            let inits = Arc::clone(&inits);
            ScanInput {
                dtype: base.dtype.clone(),
                root: LazyPlanRef::lazy(move || {
                    inits.fetch_add(1, Ordering::SeqCst);
                    Ok(plan.clone())
                }),
                locations: Arc::clone(&base.locations),
                segments: Arc::new(NoSegments),
                cache: Some(cache.clone()),
            }
        };

        let first = ScanBuilder::new(SESSION.clone(), make_input()).prepare()?;
        let second = ScanBuilder::new(SESSION.clone(), make_input()).prepare()?;
        assert_eq!(first.rows(), &(0..12));
        assert_eq!(second.rows(), &(0..12));
        assert_eq!(inits.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn prepare_propagates_lazy_root_and_dtype_mismatch_errors() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let failing_input = ScanInput {
            dtype,
            root: LazyPlanRef::lazy(|| Err(vortex_err!("corrupt footer"))),
            locations: Arc::new([]),
            segments: Arc::new(NoSegments),
            cache: None,
        };
        let err = ScanBuilder::new(SESSION.clone(), failing_input)
            .prepare()
            .err()
            .ok_or_else(|| vortex_err!("failing lazy root must error"))?;
        assert!(err.to_string().contains("corrupt footer"), "{err}");

        let store = Arc::new(Store::default());
        let input = struct_input(&store, 4)?;
        let other_struct = DType::struct_(
            [(
                "missing",
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )],
            Nullability::NonNullable,
        );
        let wrong_proj = get_item("missing", root()).bind(&other_struct)?;
        let err = ScanBuilder::new(SESSION.clone(), input.clone())
            .with_projection(wrong_proj)
            .prepare()
            .err()
            .ok_or_else(|| vortex_err!("mismatched projection field must error"))?;
        assert!(
            err.to_string()
                .contains("Couldn't find the missing field in the input scope"),
            "{err}"
        );

        let wrong_dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let wrong_filter = gt(root(), lit(0i64)).bind(&wrong_dtype)?;
        let err = ScanBuilder::new(SESSION.clone(), input)
            .with_some_filter(Some(wrong_filter))
            .prepare()
            .err()
            .ok_or_else(|| vortex_err!("mismatched filter dtype must error"))?;
        assert!(
            err.to_string()
                .contains("Comparison requires equal dtypes ignoring nullability"),
            "{err}"
        );
        Ok(())
    }

    #[test]
    fn filter_cache_respects_skip_unselective_dynamic_flag() -> VortexResult<()> {
        let store = Arc::new(Store::default());
        let mut input = struct_input(&store, 10)?;
        let cache = ScanCache::default();
        input.cache = Some(cache.clone());
        let filter = gt(get_item("ints", root()), lit(5i32)).bind(&input.dtype)?;

        let first = ScanBuilder::new(SESSION.clone(), input.clone())
            .with_some_filter(Some(filter.clone()))
            .with_skip_unselective_dynamic(false)
            .prepare()?;
        let second = ScanBuilder::new(SESSION.clone(), input.clone())
            .with_some_filter(Some(filter.clone()))
            .with_skip_unselective_dynamic(false)
            .prepare()?;
        let skipping = ScanBuilder::new(SESSION.clone(), input)
            .with_some_filter(Some(filter.clone()))
            .with_skip_unselective_dynamic(true)
            .prepare()?;

        let first_plans = first
            .filter
            .as_ref()
            .ok_or_else(|| vortex_err!("missing first filter plans"))?;
        let second_plans = second
            .filter
            .as_ref()
            .ok_or_else(|| vortex_err!("missing second filter plans"))?;
        let skipping_plans = skipping
            .filter
            .as_ref()
            .ok_or_else(|| vortex_err!("missing skipping filter plans"))?;
        assert!(Arc::ptr_eq(first_plans, second_plans));
        assert!(!Arc::ptr_eq(first_plans, skipping_plans));
        assert!(!first_plans.skips_unselective_dynamic());
        assert!(skipping_plans.skips_unselective_dynamic());
        let cached = cache
            .filter_plans(&filter)
            .ok_or_else(|| vortex_err!("cached filter entry missing"))?;
        assert!(!cached.skips_unselective_dynamic());
        Ok(())
    }

    #[test]
    fn unfiltered_chunk_rows_scales_only_for_identity_projection() -> VortexResult<()> {
        let dtype = DType::struct_(
            [(
                "ints",
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )],
            Nullability::NonNullable,
        );
        let root_plan = crate::test::struct_plan(
            1_000_000,
            dtype.clone(),
            vec![crate::test::flat(
                1_000_000,
                DType::Primitive(PType::I32, Nullability::NonNullable),
                0,
            )],
        );
        // 384 bytes/row shrinks identity chunk sizing to 16_384 rows.
        let locations: Arc<[SegmentLocation]> = Arc::from([SegmentLocation {
            offset: 0,
            length: 384_000_000,
            alignment: Alignment::none(),
        }]);
        let make_input = || ScanInput {
            dtype: dtype.clone(),
            root: LazyPlanRef::ready(root_plan.clone()),
            locations: Arc::clone(&locations),
            segments: Arc::new(NoSegments),
            cache: None,
        };

        let identity = ScanBuilder::new(SESSION.clone(), make_input()).prepare()?;
        assert_eq!(identity.chunk_rows, 16_384);

        let projected = ScanBuilder::new(SESSION.clone(), make_input())
            .with_projection(get_item("ints", root()).bind(&dtype)?)
            .prepare()?;
        assert_eq!(projected.chunk_rows, MAX_ROWS_PER_CHUNK);
        Ok(())
    }

    #[rstest]
    #[case::bool_only(get_item("flags", root()), get_item("flags", root()), 128 * ROW_BLOCK_SIZE)]
    #[case::two_fixed_fields(
        get_item("ints", root()),
        get_item("flags", root()),
        FILTERED_ROWS_PER_CHUNK
    )]
    #[case::no_fields_referenced(lit(1i32), lit(true), FILTERED_ROWS_PER_CHUNK)]
    fn filtered_rows_per_chunk_edge_cases(
        #[case] projection: Expression,
        #[case] filter: Expression,
        #[case] expected: usize,
    ) -> VortexResult<()> {
        let store = Arc::new(Store::default());
        let input = struct_input(&store, 4)?;
        let actual = filtered_rows_per_chunk(
            &input.dtype,
            &projection.bind(&input.dtype)?,
            &filter.bind(&input.dtype)?,
        )?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
