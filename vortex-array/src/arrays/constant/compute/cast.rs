// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{
    ConstantArray,
    ConstantVTable,
};
use crate::compute::{
    CastKernel,
    CastKernelAdapter,
};
use crate::{
    ArrayRef,
    IntoArray,
    register_kernel,
};

impl CastKernel for ConstantVTable {
    fn cast(&self, array: &ConstantArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        match array.scalar().cast(dtype) {
            Ok(scalar) => Ok(Some(ConstantArray::new(scalar, array.len()).into_array())),
            Err(_e) => Ok(None),
        }
    }
}

register_kernel!(CastKernelAdapter(ConstantVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_scalar::Scalar;

    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::compute::conformance::cast::test_cast_conformance;

    #[rstest]
    #[case(ConstantArray::new(Scalar::from(42u32), 5).into_array())]
    #[case(ConstantArray::new(Scalar::from(-100i32), 10).into_array())]
    #[case(ConstantArray::new(Scalar::from(3.5f32), 3).into_array())]
    #[case(ConstantArray::new(Scalar::from(true), 7).into_array())]
    #[case(ConstantArray::new(Scalar::null_typed::<i32>(), 4).into_array())]
    #[case(ConstantArray::new(Scalar::from(255u8), 1).into_array())]
    fn test_cast_constant_conformance(#[case] array: crate::ArrayRef) {
        test_cast_conformance(array.as_ref());
    }
}
