// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::Precision;
use crate::arrays::expr::{ExprArray, ExprVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<ExprVTable> for ExprVTable {
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
        // Note: Expression doesn't implement Hash, so we skip it
        // This is acceptable since expressions are typically transient
    }

    fn array_eq(array: &ExprArray, other: &ExprArray, precision: Precision) -> bool {
        array.child.array_eq(&other.child, precision) && array.dtype == other.dtype
        // Note: We don't compare expressions here as they don't implement Eq
        // This is acceptable since ExprArray is typically transient
    }
}
