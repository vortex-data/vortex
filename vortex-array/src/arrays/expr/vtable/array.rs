// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::Precision;
use crate::arrays::expr::{ExprArray, ExprVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<ExprVTable> for ExprVTable {
    fn len(array: &ExprArray) -> usize {
        array.child.len()
    }

    fn dtype(array: &ExprArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ExprArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ExprArray, state: &mut H, precision: Precision) {
        array.child.array_hash(state, precision);
        array.dtype.hash(state);
        array.expr.hash(state)
    }

    fn array_eq(array: &ExprArray, other: &ExprArray, precision: Precision) -> bool {
        array.child.array_eq(&other.child, precision)
            && array.dtype == other.dtype
            && array.expr == other.expr
    }
}
