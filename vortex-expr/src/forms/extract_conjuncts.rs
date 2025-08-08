// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{BinaryVTable, ExprRef};

/// Converting an expression to a conjunctive normal form can lead to a large number of expression
/// nodes.
/// For now, we will just extract the conjuncts from the expression, and return a vector of conjuncts.
/// We could look at try cnf with a size cap and otherwise return the original conjuncts.
pub fn conjuncts(expr: &ExprRef) -> Vec<ExprRef> {
    let mut conjuncts = vec![];
    conjuncts_impl(expr, &mut conjuncts);
    if conjuncts.is_empty() {
        conjuncts.push(expr.clone());
    }
    conjuncts
}

fn conjuncts_impl(expr: &ExprRef, conjuncts: &mut Vec<ExprRef>) {
    if let Some(expr) = expr.as_opt::<BinaryVTable>()
        && expr.op() == crate::Operator::And
    {
        conjuncts_impl(expr.lhs(), conjuncts);
        conjuncts_impl(expr.rhs(), conjuncts);
    } else {
        conjuncts.push(expr.clone())
    }
}
