// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::compute::FillNullReduce;
use vortex_array::expr::EmptyOptions;
use vortex_array::expr::FillNull as FillNullExpr;
use vortex_array::expr::ScalarFn;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::RunEndArray;
use crate::RunEndVTable;

impl FillNullReduce for RunEndVTable {
    fn fill_null(array: &RunEndArray, fill_value: &Scalar) -> VortexResult<Option<ArrayRef>> {
        let values_len = array.values().len();
        let fill_value_array = ConstantArray::new(fill_value.clone(), values_len).into_array();
        let scalar_fn = ScalarFn::new_static(&FillNullExpr, EmptyOptions);
        let new_values = ScalarFnArray::try_new(
            scalar_fn,
            vec![array.values().clone(), fill_value_array],
            values_len,
        )?
        .into_array();
        // SAFETY: modifying values only, does not affect ends
        Ok(Some(
            unsafe {
                RunEndArray::new_unchecked(
                    array.ends().clone(),
                    new_values,
                    array.offset(),
                    array.len(),
                )
            }
            .into_array(),
        ))
    }
}
