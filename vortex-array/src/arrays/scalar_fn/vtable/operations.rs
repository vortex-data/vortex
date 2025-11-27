// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::vtable::OperationsVTable;
use crate::{ArrayRef, IntoArray};
use std::ops::Range;
use vortex_scalar::Scalar;
use crate::functions::ExecutionCtx;

impl OperationsVTable<ScalarFnVTable> for ScalarFnVTable {
    fn slice(array: &ScalarFnArray, range: Range<usize>) -> ArrayRef {
        let children: Vec<_> = array
            .children()
            .iter()
            .map(|c| c.slice(range.clone()))
            .collect();

        ScalarFnArray {
            vtable: array.vtable.clone(),
            scalar_fn: array.scalar_fn.clone(),
            dtype: array.dtype.clone(),
            len: range.len(),
            children,
            stats: Default::default(),
        }
        .into_array()
    }

    fn scalar_at(array: &ScalarFnArray, index: usize) -> Scalar {
        // TODO(ngates): we should evaluate the scalar function over the scalar inputs.
        let children: Vec<_> = array
            .children()
            .iter()
            .map(|c| c.scalar_at(index))
            .collect();

        let ctx = ExecutionCtx::new(
            1,
            array.dtype.clone(),
            children.iter().map(|s| s.dtype().clone()).collect(),
            children.iter().map(|s| s.to_vector()).collect(),
        )

        todo!()
    }
}
