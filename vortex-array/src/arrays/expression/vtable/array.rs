// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use vortex_dtype::DType;

use crate::Array;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::Precision;
use crate::arrays::expression::ExpressionArray;
use crate::arrays::expression::ExpressionVTable;
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<ExpressionVTable> for ExpressionVTable {
    fn len(array: &ExpressionArray) -> usize {
        array.input.len()
    }

    fn dtype(array: &ExpressionArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ExpressionArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &ExpressionArray, state: &mut H, precision: Precision) {
        array.input.array_hash(state, precision);
        array.expression.hash(state);
    }

    fn array_eq(array: &ExpressionArray, other: &ExpressionArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.expression == other.expression
            && array.input.array_eq(&other.input, precision)
    }
}
