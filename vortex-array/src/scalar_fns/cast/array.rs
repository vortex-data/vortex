// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::ArrayVisitor;
use crate::arrays::ExactScalarFn;
use crate::arrays::ScalarFnArrayView;
use crate::optimizer::rules::ArrayReduceRule;
use crate::scalar_fns::cast::CastFn;

#[derive(Debug)]
pub(crate) struct CastArrayReduce;

impl ArrayReduceRule<ExactScalarFn<CastFn>> for CastArrayReduce {
    fn matcher(&self) -> ExactScalarFn<CastFn> {
        ExactScalarFn::from(&CastFn)
    }

    fn reduce(&self, array: ScalarFnArrayView<'_, CastFn>) -> VortexResult<Option<ArrayRef>> {
        let target_dtype = array.options;

        // If the array is already of the target dtype, then return the input node as-is.
        if array.dtype() == target_dtype {
            return Ok(Some(array.children()[0].clone()));
        }

        Ok(None)
    }
}

#[cfg(test)]
mod test {
    use vortex_error::VortexResult;

    use super::CastArrayReduce;
    use crate::Array;
    use crate::array::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::ConstantVTable;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar_fns::BuiltinScalarFns;

    #[test]
    fn test_same_dtype() -> VortexResult<()> {
        let mut optimizer = ArrayOptimizer::default();
        optimizer.register_reduce_rule(CastArrayReduce);

        let array = ConstantArray::new(true, 10).into_array();
        let cast_same_dtype = array.cast(array.dtype().clone())?;

        let optimized = optimizer.optimize_array(cast_same_dtype)?;
        assert!(optimized.is::<ConstantVTable>());

        Ok(())
    }
}
