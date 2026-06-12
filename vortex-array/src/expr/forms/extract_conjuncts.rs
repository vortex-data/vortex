// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::BoundExpr;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;

/// Converting an expression to a conjunctive normal form can lead to a large number of expression
/// nodes.
/// For now, we will just extract the conjuncts from the expression, and return a vector of conjuncts.
/// We could look at try cnf with a size cap and otherwise return the original conjuncts.
pub fn conjuncts(expr: &BoundExpr) -> Vec<BoundExpr> {
    let mut conjuncts = vec![];
    conjuncts_impl(expr, &mut conjuncts);
    if conjuncts.is_empty() {
        conjuncts.push(expr.clone());
    }
    conjuncts
}

fn conjuncts_impl(expr: &BoundExpr, conjuncts: &mut Vec<BoundExpr>) {
    if let Some(operator) = expr.as_opt::<Binary>()
        && *operator == Operator::And
    {
        conjuncts_impl(expr.child(0), conjuncts);
        conjuncts_impl(expr.child(1), conjuncts);
    } else {
        conjuncts.push(expr.clone())
    }
}
