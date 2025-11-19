// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::arrays::expr::{ExprArray, ExprVTable};
use crate::stats::ArrayStats;
use crate::vtable::OperationsVTable;
use crate::{Array, ArrayRef, IntoArray};

impl OperationsVTable<ExprVTable> for ExprVTable {
    fn slice(array: &ExprArray, range: Range<usize>) -> ArrayRef {
        let child = array.child.slice(range);

        ExprArray {
            child,
            expr: array.expr.clone(),
            dtype: array.dtype.clone(),
            stats: ArrayStats::default(),
        }
        .into_array()
    }

    fn scalar_at(array: &ExprArray, index: usize) -> Scalar {
        // TODO(joe): this is unchecked
        array
            .expr
            .evaluate(&ConstantArray::new(array.child.scalar_at(index), 1).into_array())
            .vortex_expect("cannot fail")
            .as_constant()
            .vortex_expect("expr are scalar so cannot fail")
    }
}
