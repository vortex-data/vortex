// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::scalar_fn::fns::dynamic::DynamicComparison;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use crate::plan::Eval;
use crate::plan::PlanRef;
use crate::plan::Share;
use crate::plan::Take;
use crate::scan::v2::conjuncts::map_children;

/// Removes each [`Share`] of dictionary values that no plan of the scan reads whole.
///
/// Values read whole are canonicalized anyway, so an expression pushed onto the same values
/// reuses that work by running over the share, as the default scan's dictionary reader does once
/// it holds the canonical values. Without such a reader, the expression runs over the encoded
/// values instead, where kernels such as a compare over FSST avoid decompressing them.
pub(super) fn unshare_unread(plans: Vec<PlanRef>) -> VortexResult<Vec<PlanRef>> {
    let mut read = HashSet::default();
    for plan in &plans {
        collect_read(plan, &mut read)?;
    }
    plans.into_iter().map(|plan| unshare(plan, &read)).collect()
}

/// Records the shares that a take reads whole in `plan`, or evaluates a dynamic comparison over:
/// a take evaluates that again on every execution, so the values under it stay shared.
fn collect_read(plan: &PlanRef, read: &mut HashSet<usize>) -> VortexResult<()> {
    if let Some(take) = plan.as_opt::<Take>() {
        let mut values = take.values()?;
        if let Some(eval) = values.as_opt::<Eval>()
            && eval.expression().contains::<DynamicComparison>()?
        {
            values = eval.child_plan()?;
        }
        if values.is::<Share>() {
            read.insert(values.addr());
        }
    }
    for child in plan.children().iter() {
        collect_read(&child?, read)?;
    }
    Ok(())
}

fn unshare(plan: PlanRef, read: &HashSet<usize>) -> VortexResult<PlanRef> {
    if let Some(share) = plan.as_opt::<Share>()
        && !read.contains(&plan.addr())
    {
        return unshare(share.child_plan()?, read);
    }
    map_children(plan, |child| unshare(child, read))
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::registry::ReadContext;

    use super::unshare_unread;
    use crate::LayoutRef;
    use crate::layouts::dict::DictLayout;
    use crate::layouts::flat::FlatLayout;
    use crate::plan::Eval;
    use crate::plan::PlanRef;
    use crate::plan::Share;
    use crate::plan::Take;
    use crate::plan::optimize;
    use crate::plan::plan_row_idx_expression;
    use crate::scan::v2::lower::lower;
    use crate::segments::SegmentId;

    fn dictionary() -> LayoutRef {
        let flat = |row_count, dtype, segment: u32| {
            FlatLayout::new(
                row_count,
                dtype,
                SegmentId::from(segment),
                ReadContext::new([]),
            )
            .into_layout()
        };
        DictLayout::new(
            flat(2, DType::Utf8(NonNullable), 0),
            flat(3, DType::Primitive(PType::U8, NonNullable), 1),
        )
        .into_layout()
    }

    fn plan(root_plan: &PlanRef, expression: Expression) -> VortexResult<PlanRef> {
        let expression = expression.bind(root_plan.dtype())?;
        optimize(plan_row_idx_expression(expression, root_plan.clone())?)
    }

    /// The values under the pushed predicate: the child of the `Eval` on the take's values.
    fn predicate_input(filter: &PlanRef) -> VortexResult<PlanRef> {
        let values = filter
            .as_opt::<Take>()
            .ok_or_else(|| vortex_err!("filter is not a take: {filter}"))?
            .values()?;
        values
            .as_opt::<Eval>()
            .ok_or_else(|| vortex_err!("predicate was not pushed onto the values: {values}"))?
            .child_plan()
    }

    #[test]
    fn share_is_kept_where_values_are_read_whole() -> VortexResult<()> {
        let root_plan = lower(&dictionary())?;
        let plans = unshare_unread(vec![
            plan(&root_plan, root())?,
            plan(&root_plan, eq(root(), lit("a")))?,
        ])?;

        let read_whole = plans[0]
            .as_opt::<Take>()
            .ok_or_else(|| vortex_err!("projection is not a take"))?
            .values()?;
        assert!(read_whole.is::<Share>());
        let under_predicate = predicate_input(&plans[1])?;
        assert!(PlanRef::ptr_eq(&read_whole, &under_predicate));
        Ok(())
    }

    #[test]
    fn share_is_removed_where_values_are_not_read_whole() -> VortexResult<()> {
        let root_plan = lower(&dictionary())?;
        let plans = unshare_unread(vec![plan(&root_plan, eq(root(), lit("a")))?])?;
        assert!(!predicate_input(&plans[0])?.is::<Share>());
        Ok(())
    }
}
