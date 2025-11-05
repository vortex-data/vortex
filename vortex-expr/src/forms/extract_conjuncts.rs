// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Expression;
use crate::exprs::binary::Binary;
use crate::exprs::operators::Operator;

/// Converting an expression to a conjunctive normal form can lead to a large number of expression
/// nodes.
/// For now, we will just extract the conjuncts from the expression, and return a vector of conjuncts.
/// We could look at try cnf with a size cap and otherwise return the original conjuncts.
pub fn conjuncts(expr: &Expression) -> Vec<Expression> {
    let mut conjuncts = vec![];
    conjuncts_impl(expr, &mut conjuncts);
    if conjuncts.is_empty() {
        conjuncts.push(expr.clone());
    }
    conjuncts
}

fn conjuncts_impl(expr: &Expression, conjuncts: &mut Vec<Expression>) {
    if let Some(expr) = expr.as_opt::<Binary>()
        && expr.operator() == Operator::And
    {
        conjuncts_impl(expr.lhs(), conjuncts);
        conjuncts_impl(expr.rhs(), conjuncts);
    } else {
        conjuncts.push(expr.clone())
    }
}
