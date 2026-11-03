// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;

use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::dynamic::DynamicComparison;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::layouts::sparse::INDICES_DTYPE;
use crate::layouts::sparse::SparseBlocks;
use crate::plan::physical::Eval;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::Plan;
use crate::plan::physical::PlanChildren;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::operators::OnceSource;
use crate::plan::physical::operators::SparseSource;
use crate::plan::physical::operators::UNBOUNDED;
use crate::plan::physical::optimizer::PlanParentReduceRule;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;

const INDICES: usize = 0;
const VALUES: usize = 1;

/// Scatters the rows of `values` to the row positions in `indices`, filling every other row.
#[derive(Clone, Debug)]
pub struct Sparse;

/// The fill of a [`Sparse`] plan and where its values lie.
#[derive(Clone, Debug)]
pub struct SparseData {
    blocks: SparseBlocks,
    fill: Scalar,
}

/// A plan that scatters its present rows over a row domain of fill rows.
pub type SparsePlan = Plan<Sparse>;

impl SparsePlan {
    /// Creates a sparse plan from potentially unresolved children without validation.
    ///
    /// # Safety
    ///
    /// `children` must be `[indices, values]`. `indices` must be the sorted, non-null `u64`
    /// positions of the present rows within `0..row_count`, and `values` those rows, both with
    /// `blocks.present_count()` rows. `values` and `fill` must have `dtype`, and `blocks` must
    /// count the present rows of every block of `row_count` rows.
    pub(crate) unsafe fn from_children_unchecked(
        dtype: DType,
        row_count: u64,
        blocks: SparseBlocks,
        fill: Scalar,
        children: PlanChildren,
    ) -> Self {
        PlanParts {
            vtable: Sparse,
            dtype,
            row_count,
            children,
            data: SparseData { blocks, fill },
        }
        .into_typed()
    }

    /// Returns the plan producing the present-row positions.
    pub fn indices(&self) -> VortexResult<PlanRef> {
        self.child_required(INDICES)
    }

    /// Returns the plan producing the present rows.
    pub fn values(&self) -> VortexResult<PlanRef> {
        self.child_required(VALUES)
    }

    /// Returns the value of every row that is not present.
    pub fn fill(&self) -> &Scalar {
        &self.data().fill
    }

    /// Returns the present-row count of every block of rows.
    pub fn blocks(&self) -> &SparseBlocks {
        &self.data().blocks
    }
}

impl PlanVTable for Sparse {
    type PlanData = SparseData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.sparse");
        *ID
    }

    fn fmt(plan: &Plan<Self>, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            " fill={} present={}",
            plan.fill(),
            plan.blocks().present_count()
        )
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // The block counts and fill are not wired up to plan serialization.
        None
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        if children.len() != 2 {
            vortex_bail!("Sparse expects 2 children but got {}", children.len());
        }
        let present = plan.blocks().present_count();
        let indices = children
            .get(INDICES)?
            .ok_or_else(|| vortex_err!("Sparse indices child is absent"))?;
        if indices.dtype() != &INDICES_DTYPE || indices.row_count() != present {
            vortex_bail!(
                "Sparse indices child must be {INDICES_DTYPE} with {present} rows, got {} with {}",
                indices.dtype(),
                indices.row_count()
            );
        }
        let values = children
            .get(VALUES)?
            .ok_or_else(|| vortex_err!("Sparse values child is absent"))?;
        if values.dtype() != plan.dtype() || values.row_count() != present {
            vortex_bail!(
                "Sparse values child must be {} with {present} rows, got {} with {}",
                plan.dtype(),
                values.dtype(),
                values.row_count()
            );
        }
        Ok(())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        match index {
            INDICES => Cow::Borrowed("indices"),
            VALUES => Cow::Borrowed("values"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }

    /// Reads the indices and values of every block `rows` overlaps, found from the block counts
    /// without IO, and scatters those inside `rows` that `mask` selects.
    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        if mask.all_false() {
            return Ok(None);
        }
        let values_rows = plan.blocks().values_range(&rows);
        if values_rows.is_empty() {
            let fill = ConstantArray::new(plan.fill().clone(), mask.true_count()).into_array();
            return Ok(Some(Chain::new(OnceSource::new(fill))));
        }
        let all = Mask::new_true(usize::try_from(values_rows.end - values_rows.start)?);
        let indices = compiler
            .compile(&plan.indices()?, values_rows.clone(), &all)?
            .ok_or_else(|| vortex_err!("Sparse indices compiled to no rows"))?;
        let values = compiler
            .compile(&plan.values()?, values_rows, &all)?
            .ok_or_else(|| vortex_err!("Sparse values compiled to no rows"))?;
        let source = SparseSource::new(
            plan.dtype().clone(),
            plan.fill().clone(),
            rows.start,
            mask.clone(),
        );
        Ok(Some(compiler.join_with(
            vec![indices, values],
            |_| UNBOUNDED,
            source,
        )))
    }
}

/// Pushes an expression onto the values of a [`Sparse`], filling its absent rows with the
/// expression's result on the fill.
///
/// Expressions are evaluated row by row, so a row's result depends on that row alone. A dynamic
/// comparison stays above, since its result on the fill changes with its bound.
#[derive(Debug)]
pub(crate) struct ExpressionSparseRule;

impl PlanParentReduceRule<Sparse> for ExpressionSparseRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &Plan<Sparse>,
        parent: &Plan<Eval>,
        _child_idx: usize,
        session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        let expression = parent.expression();
        if expression.contains::<DynamicComparison>()? {
            return Ok(None);
        }
        let mut ctx = session.create_execution_ctx();
        // An expression that fails on the fill keeps its failure for the rows that need it.
        let Ok(fill) = ConstantArray::new(child.fill().clone(), 1)
            .into_array()
            .apply_bound(expression)
            .and_then(|result| result.execute_scalar(0, &mut ctx))
        else {
            return Ok(None);
        };
        let dtype = expression.dtype().clone();
        let fill = if fill.dtype() == &dtype {
            fill
        } else {
            fill.cast(&dtype)?
        };
        let values = EvalPlan::try_new(expression.clone(), child.values()?)?.into_plan();
        let children = PlanChildren::from(vec![child.indices()?, values]);
        // SAFETY: the indices and blocks are the child's, and the values and fill are its own
        // evaluated by `expression`, so they share its dtype and keep their row counts.
        Ok(Some(
            unsafe {
                SparsePlan::from_children_unchecked(
                    dtype,
                    child.row_count(),
                    child.blocks().clone(),
                    fill,
                    children,
                )
            }
            .into_plan(),
        ))
    }
}
