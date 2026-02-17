// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::register_kernel;

impl MinMaxKernel for ConstantVTable {
    fn min_max(&self, array: &ConstantArray) -> VortexResult<Option<MinMaxResult>> {
        let scalar = array.scalar();
        if scalar.is_null() || scalar.as_primitive_opt().is_some_and(|p| p.is_nan()) {
            return Ok(None);
        }
        let non_nullable_dtype = scalar.dtype().as_nonnullable();
        Ok(Some(MinMaxResult {
            min: scalar.cast(&non_nullable_dtype)?,
            max: scalar.cast(&non_nullable_dtype)?,
        }))
    }
}

register_kernel!(MinMaxKernelAdapter(ConstantVTable).lift());

#[cfg(test)]
mod test {
    use vortex_dtype::Nullability;
    use vortex_dtype::half::f16;

    use crate::arrays::ConstantArray;
    use crate::compute::min_max;
    use crate::scalar::Scalar;

    #[test]
    fn test_min_max_nan() {
        let scalar = Scalar::primitive(f16::NAN, Nullability::NonNullable);
        let array = ConstantArray::new(scalar, 2).to_array();
        let result = min_max(&array).unwrap();
        assert_eq!(result, None);
    }
}
