// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_scalar::Scalar;
use vortex_vector::Datum;

use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::expr::functions::ExecutionArgs;
use crate::vtable::OperationsVTable;
use crate::ArrayRef;
use crate::IntoArray;

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
        let input_datums: Vec<_> = array
            .children()
            .iter()
            .map(|c| c.scalar_at(index))
            .map(|scalar| Datum::from(scalar.to_vector_scalar()))
            .collect();

        let ctx = ExecutionArgs::new(
            1,
            array.dtype.clone(),
            array.children().iter().map(|s| s.dtype().clone()).collect(),
            input_datums,
        );

        let _result = array
            .scalar_fn
            .execute(&ctx)
            .vortex_expect("Scalar function execution should be fallible")
            .into_scalar()
            .vortex_expect("Scalar function execution should return scalar");

        // Convert the vector scalar back into a legacy Scalar for now.
        todo!("Implement legacy scalar conversion")
    }
}
