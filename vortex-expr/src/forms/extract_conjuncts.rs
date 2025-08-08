// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{BinaryVTable, ExprRef};

pub fn conjuncts(expr: &ExprRef) -> Vec<ExprRef> {
    let mut conjuncts = vec![];
    conjuncts_impl(expr, &mut conjuncts);
    conjuncts
}

fn conjuncts_impl(expr: &ExprRef, conjuncts: &mut Vec<ExprRef>) {
    if let Some(expr) = expr.as_opt::<BinaryVTable>() {
        conjuncts_impl(expr.lhs(), conjuncts);
        conjuncts_impl(expr.rhs(), conjuncts);
    }
}
