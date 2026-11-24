// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rewriting an expression over a list onto the list children it needs.
//!
//! This is the V2 counterpart of what the list layout's reader does with the expression it is
//! handed: `is_null($)` reads only the validity, and `list_length($)` only the offsets and the
//! validity. Anything else reads the lists.

use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::not;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::builtin::is_not_null::IsNotNull;
use vortex_array::scalar_fn::builtin::is_null::IsNull;
use vortex_error::VortexResult;
use vortex_functions::list_length::ListLength;
use vortex_session::VortexSession;

use super::ListPack;
use super::lengths::ListLengthsPlan;
use crate::plan::Eval;
use crate::plan::EvalPlan;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::plan::optimize::PlanParentReduceRule;

/// What an expression needs of the list it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Needs {
    /// Only whether each list is null (`is_null`, `is_not_null`).
    Validity,
    /// Each list's length, and whether it is null (`list_length`).
    Lengths,
    /// The lists themselves.
    Lists,
}

/// Pushes an expression over a list onto the list's children.
#[derive(Debug)]
pub(crate) struct ExpressionListPackRule;

impl PlanParentReduceRule<ListPack> for ExpressionListPackRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &Plan<ListPack>,
        parent: &Plan<Eval>,
        _child_idx: usize,
        _session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        let expression = parent.expression();
        let rewritten = match needs(expression) {
            Some(Needs::Validity) => validity_only(child, expression)?,
            Some(Needs::Lengths) => lengths_only(child, expression)?,
            // The whole lists, or no list at all: nothing to push.
            Some(Needs::Lists) | None => None,
        };
        // A rewrite that changed the result's type would be a bug in it, not a reason to fail.
        Ok(rewritten.filter(|plan| plan.dtype() == parent.dtype()))
    }
}

/// Evaluates an expression that only asks whether lists are null over their validity.
fn validity_only(
    list: &Plan<ListPack>,
    expression: &BoundExpression,
) -> VortexResult<Option<PlanRef>> {
    // A list that cannot be null has no validity to read: its lengths say as much, and are the
    // cheapest thing it has.
    let Some(validity) = list.validity()? else {
        return lengths_only(list, expression);
    };
    let root = BoundExpression::new_root(validity.dtype().clone());
    let rewritten = expression
        .clone()
        .transform_down(|node| {
            let replacement = if is_of_root::<IsNotNull>(&node) {
                Some(root.clone())
            } else if is_of_root::<IsNull>(&node) {
                Some(not(root.clone()))
            } else {
                None
            };
            Ok(replace_or_keep(node, replacement))
        })?
        .into_inner();
    Ok(Some(EvalPlan::try_new(rewritten, validity)?.into_plan()))
}

/// Evaluates an expression that reads lists only through their lengths over the lengths, which
/// carry the lists' validity.
fn lengths_only(
    list: &Plan<ListPack>,
    expression: &BoundExpression,
) -> VortexResult<Option<PlanRef>> {
    let nullability = list.dtype().nullability();
    let lengths = ListLengthsPlan::try_new(
        nullability,
        list.row_count(),
        list.offsets()?,
        list.validity()?,
    )?
    .into_plan();
    let root = BoundExpression::new_root(DType::Primitive(PType::U64, nullability));
    let rewritten = expression
        .clone()
        .transform_down(|node| {
            // A null check of the list is a null check of its length.
            let replacement =
                (is_of_root::<ListLength>(&node) || node.is_root()).then(|| root.clone());
            Ok(replace_or_keep(node, replacement))
        })?
        .into_inner();
    Ok(Some(EvalPlan::try_new(rewritten, lengths)?.into_plan()))
}

/// `node`, or `replacement` in its place without visiting what it replaced.
fn replace_or_keep(
    node: BoundExpression,
    replacement: Option<BoundExpression>,
) -> Transformed<BoundExpression> {
    match replacement {
        Some(value) => Transformed {
            value,
            changed: true,
            order: TraversalOrder::Skip,
        },
        None => Transformed::no(node),
    }
}

/// Whether `node` is the scalar function `V` applied to the root alone.
fn is_of_root<V: ScalarFnVTable>(node: &BoundExpression) -> bool {
    node.as_scalar().is_some_and(|scalar_fn| scalar_fn.is::<V>())
        && node.children().len() == 1
        && node.child(0).is_root()
}

/// What `expression` needs of the list it is over: `None` when it does not read the list.
fn needs(expression: &BoundExpression) -> Option<Needs> {
    if is_of_root::<IsNull>(expression) || is_of_root::<IsNotNull>(expression) {
        Some(Needs::Validity)
    } else if is_of_root::<ListLength>(expression) {
        Some(Needs::Lengths)
    } else if expression.is_root() {
        Some(Needs::Lists)
    } else {
        expression.children().iter().filter_map(needs).max()
    }
}
