// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// See https://github.com/vortex-data/vortex/issues/9062

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;

use crate::ArrayFuture;
use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;

pub(crate) struct PlanV2 {
    projection: ScanPlanRef,
    predicates: Vec<ScanPlanRef>,
    filter: Option<Expression>,
}

impl PlanV2 {
    pub(crate) fn new(
        projection: ScanPlanRef,
        predicates: Vec<ScanPlanRef>,
        filter: Option<Expression>,
    ) -> Self {
        Self {
            projection,
            predicates,
            filter,
        }
    }

    pub(crate) fn task_context<A>(
        &self,
        mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    ) -> Arc<TaskContext<A>> {
        Arc::new(TaskContext {
            filter: self
                .filter
                .clone()
                .map(|filter| Arc::new(FilterExpr::new(filter))),
            predicates: self.predicates.clone(),
            projection: Arc::clone(&self.projection),
            mapper,
        })
    }
}

/// Shared handle to a heap-allocated V2 physical scan plan.
pub type ScanPlanRef = Arc<dyn ScanPlan>;

/// A heap-allocated physical scan plan.
///
/// A source plan represents an instantiated layout. [`apply_expr`](Self::apply_expr) derives
/// another plan whose root value is the applied expression, and [`optimize`](Self::optimize)
/// rewrites that derived plan before execution. Execution therefore selects an already-bound plan
/// and supplies only its row range and mask.
pub trait ScanPlan: 'static + Send + Sync {
    /// Apply `expr` to this plan's root value and return the resulting plan.
    fn apply_expr(self: Arc<Self>, expr: Expression) -> VortexResult<ScanPlanRef>;

    /// Optimize this plan and return the resulting plan.
    fn optimize(self: Arc<Self>) -> VortexResult<ScanPlanRef>;

    /// Returns the name of the underlying layout reader for debugging.
    fn name(&self) -> &Arc<str>;

    /// Returns the dtype produced by this plan.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in this plan's row domain.
    fn row_count(&self) -> u64;

    /// Returns a mask where all false values are proven false for this plan.
    fn pruning_evaluation(&self, row_range: &Range<u64>, mask: Mask) -> VortexResult<MaskFuture>;

    /// Evaluates this boolean plan and intersects it with `mask`.
    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture>;

    /// Evaluates this plan over the selected rows.
    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture>;
}

/// Compatibility V2 source and expression plan backed by a layout reader.
///
/// Applying an expression and optimizing it produce new heap-allocated plans. Execution delegates
/// the resulting expression to the established reader implementation. Layout-specific source plans
/// can replace this compatibility node without changing the split execution loop.
pub struct LayoutReaderScanPlanV2 {
    reader: LayoutReaderRef,
    expr: Expression,
    dtype: DType,
}

impl LayoutReaderScanPlanV2 {
    /// Create a V2 source plan for `reader`.
    pub fn new(reader: LayoutReaderRef) -> Self {
        let dtype = reader.dtype().clone();
        Self {
            reader,
            expr: root(),
            dtype,
        }
    }

    fn try_new(reader: LayoutReaderRef, expr: Expression) -> VortexResult<Self> {
        let dtype = expr.return_dtype(reader.dtype())?;
        Ok(Self {
            reader,
            expr,
            dtype,
        })
    }
}

impl ScanPlan for LayoutReaderScanPlanV2 {
    fn apply_expr(self: Arc<Self>, expr: Expression) -> VortexResult<ScanPlanRef> {
        let expr = replace(expr, &root(), self.expr.clone());
        Ok(Arc::new(Self::try_new(Arc::clone(&self.reader), expr)?))
    }

    fn optimize(self: Arc<Self>) -> VortexResult<ScanPlanRef> {
        let expr = self.expr.optimize_recursive(self.reader.dtype())?;
        Ok(Arc::new(Self::try_new(Arc::clone(&self.reader), expr)?))
    }

    fn name(&self) -> &Arc<str> {
        self.reader.name()
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.reader.row_count()
    }

    fn pruning_evaluation(&self, row_range: &Range<u64>, mask: Mask) -> VortexResult<MaskFuture> {
        self.reader.pruning_evaluation(row_range, &self.expr, mask)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.reader.filter_evaluation(row_range, &self.expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        self.reader
            .projection_evaluation(row_range, &self.expr, mask)
    }
}

/// Environment variable selecting the scan planning implementation.
pub const SCAN_IMPL_ENV: &str = "VORTEX_SCAN_IMPL";

/// Returns whether V2 heap-allocated planning is enabled for this process.
///
/// The existing `plan` path remains the default on this extraction branch. Set
/// `VORTEX_SCAN_IMPL=planv2` to exercise the V2 path with the same execution implementation.
pub fn plan_v2_enabled() -> VortexResult<bool> {
    match std::env::var(SCAN_IMPL_ENV) {
        Ok(value) => parse_scan_impl(&value),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(value)) => {
            vortex_bail!("{SCAN_IMPL_ENV} must be valid unicode, got {value:?}")
        }
    }
}

fn parse_scan_impl(value: &str) -> VortexResult<bool> {
    match value {
        "" | "plan" | "v1" | "legacy" | "layout-reader" => Ok(false),
        "planv2" | "plan-v2" | "v2" | "planned" | "scan-plan" => Ok(true),
        other => vortex_bail!(
            "{SCAN_IMPL_ENV} must be one of plan, v1, legacy, layout-reader, planv2, plan-v2, v2, planned, or scan-plan, got {other:?}"
        ),
    }
}

/// Execute one split using a V2 physical scan plan.
///
/// The execution order intentionally mirrors [`crate::scan::plan::split_exec`]. Expressions were
/// consumed during planning, so execution selects a predicate or projection plan without passing
/// an expression.
pub(crate) fn split_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: RowMask,
    limit: Option<&mut u64>,
) -> VortexResult<BoxFuture<'static, VortexResult<Option<A>>>> {
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();

    let filter_mask = match ctx.filter.as_ref() {
        None => {
            let row_mask = match limit {
                Some(l) if *l == 0 => Mask::new_false(row_mask.len()),
                Some(l) => {
                    let true_count = row_mask.true_count();
                    let mask_limit = usize::try_from(*l)
                        .map(|l| l.min(true_count))
                        .unwrap_or(true_count);
                    let row_mask = row_mask.limit(mask_limit);
                    *l -= mask_limit as u64;
                    row_mask
                }
                None => row_mask,
            };

            MaskFuture::ready(row_mask)
        }
        Some(filter) => {
            if filter.conjuncts().len() != ctx.predicates.len() {
                vortex_bail!(
                    "physical predicate count {} does not match conjunct count {}",
                    ctx.predicates.len(),
                    filter.conjuncts().len()
                );
            }

            let ctx = Arc::clone(&ctx);
            let filter = Arc::clone(filter);
            let row_range = row_range.clone();

            MaskFuture::new(row_mask.len(), async move {
                let mut mask = row_mask;
                let mut dynamic_versions = vec![None; filter.conjuncts().len()];

                for (idx, predicate) in ctx.predicates.iter().enumerate() {
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());
                    let conjunct_mask = predicate
                        .pruning_evaluation(&row_range, mask.clone())?
                        .await?;
                    mask = mask.bitand(&conjunct_mask);
                }

                let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);
                while let Some(idx) = filter.next_conjunct(&remaining) {
                    remaining.set(idx, false);
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                    if let Some(version) = current_version
                        && dynamic_versions[idx].is_none_or(|old| old < version)
                    {
                        dynamic_versions[idx] = Some(version);
                        let conjunct_mask = ctx.predicates[idx]
                            .pruning_evaluation(&row_range, mask.clone())?
                            .await?;
                        mask = mask.bitand(&conjunct_mask);
                    }
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let conjunct_mask = ctx.predicates[idx]
                        .filter_evaluation(&row_range, MaskFuture::ready(mask))?
                        .await?;
                    filter.report_selectivity(idx, conjunct_mask.density());
                    mask = conjunct_mask;
                }

                Ok(mask)
            })
        }
    };

    let projection_future = ctx
        .projection
        .projection_evaluation(&row_range, filter_mask.clone())?;

    let mapper = Arc::clone(&ctx.mapper);
    let array_fut = async move {
        let mask = filter_mask.await?;
        if mask.all_false() {
            return Ok(None);
        }

        let array = projection_future.await?;
        mapper(array).map(Some)
    };

    Ok(array_fut.boxed())
}

/// Information needed to execute one split from a V2 physical scan plan.
pub(crate) struct TaskContext<A> {
    filter: Option<Arc<FilterExpr>>,
    predicates: Vec<ScanPlanRef>,
    projection: ScanPlanRef,
    mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use vortex_array::dtype::FieldMask;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::lit;

    use super::*;
    use crate::LayoutReader;
    use crate::RowSplits;
    use crate::SplitRange;

    #[test]
    fn scan_impl_accepts_v1_and_v2_values() -> VortexResult<()> {
        for value in ["", "plan", "v1", "legacy", "layout-reader"] {
            assert!(!parse_scan_impl(value)?);
        }
        for value in ["planv2", "plan-v2", "v2", "planned", "scan-plan"] {
            assert!(parse_scan_impl(value)?);
        }
        Ok(())
    }

    #[test]
    fn scan_impl_rejects_unknown_value() {
        assert!(parse_scan_impl("unknown").is_err());
    }

    struct TestLayoutReader {
        name: Arc<str>,
        dtype: DType,
    }

    impl TestLayoutReader {
        fn new() -> Self {
            Self {
                name: Arc::from("test"),
                dtype: DType::Struct(
                    StructFields::from_iter([(
                        FieldName::from("a"),
                        DType::Primitive(PType::I32, Nullability::NonNullable),
                    )]),
                    Nullability::NonNullable,
                ),
            }
        }
    }

    impl LayoutReader for TestLayoutReader {
        fn name(&self) -> &Arc<str> {
            &self.name
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn row_count(&self) -> u64 {
            1
        }

        fn register_splits(
            &self,
            _field_mask: &[FieldMask],
            _split_range: &SplitRange,
            _splits: &mut RowSplits,
        ) -> VortexResult<()> {
            unimplemented!("not needed for scan-plan construction")
        }

        fn pruning_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: Mask,
        ) -> VortexResult<MaskFuture> {
            unimplemented!("not needed for scan-plan construction")
        }

        fn filter_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: MaskFuture,
        ) -> VortexResult<MaskFuture> {
            unimplemented!("not needed for scan-plan construction")
        }

        fn projection_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: MaskFuture,
        ) -> VortexResult<ArrayFuture> {
            unimplemented!("not needed for scan-plan construction")
        }
    }

    #[test]
    fn scan_plan_v2_applies_expressions_to_the_current_root() -> VortexResult<()> {
        let reader: LayoutReaderRef = Arc::new(TestLayoutReader::new());
        let source: ScanPlanRef = Arc::new(LayoutReaderScanPlanV2::new(reader));

        let field = Arc::clone(&source)
            .apply_expr(get_item("a", root()))?
            .optimize()?;
        assert_eq!(
            field.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        let predicate = field.apply_expr(eq(root(), lit(1_i32)))?.optimize()?;
        assert_eq!(predicate.dtype(), &DType::Bool(Nullability::NonNullable));

        Ok(())
    }
}
