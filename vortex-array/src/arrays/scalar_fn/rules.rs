// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::Scalar;
use vortex_vector::Datum;
use vortex_vector::VectorOps;
use vortex_vector::scalar_matches_dtype;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::AnyScalarFn;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::ScalarFnArray;
use crate::expr::ExecutionArgs;
use crate::optimizer::rules::ArrayReduceRule;

#[derive(Debug)]
pub(crate) struct ScalarFnConstantRule;
impl ArrayReduceRule<AnyScalarFn> for ScalarFnConstantRule {
    fn matcher(&self) -> AnyScalarFn {
        AnyScalarFn
    }

    fn reduce(&self, array: &ScalarFnArray) -> VortexResult<Option<ArrayRef>> {
        if !array.children.iter().all(|c| c.is::<ConstantVTable>()) {
            return Ok(None);
        }

        let input_datums: Vec<_> = array
            .children
            .iter()
            .map(|c| c.as_::<ConstantVTable>().scalar().to_vector_scalar())
            .map(Datum::Scalar)
            .collect();
        let input_dtypes = array.children.iter().map(|c| c.dtype().clone()).collect();

        let result = array.scalar_fn.execute(ExecutionArgs {
            datums: input_datums,
            dtypes: input_dtypes,
            row_count: array.len,
            return_dtype: array.dtype.clone(),
        })?;

        let result = match result {
            Datum::Scalar(s) => s,
            Datum::Vector(v) => {
                tracing::warn!(
                    "Scalar function {} returned vector from execution over all scalar inputs",
                    array.scalar_fn
                );
                v.scalar_at(0)
            }
        };
        assert!(scalar_matches_dtype(&result, &array.dtype));

        let _fn = format!("{}", array.scalar_fn);
        Ok(Some(
            ConstantArray::new(Scalar::from_vector_scalar(result, &array.dtype)?, array.len)
                .into_array(),
        ))
    }
}
