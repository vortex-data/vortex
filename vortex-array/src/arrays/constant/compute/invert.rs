use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::compute::{InvertKernel, InvertKernelAdapter};
use crate::{ArrayRef, IntoArray, register_kernel};

impl InvertKernel for ConstantVTable {
    fn invert(&self, array: &ConstantArray) -> VortexResult<ArrayRef> {
        match array.scalar().as_bool().value() {
            None => Ok(array.to_array()),
            Some(b) => Ok(ConstantArray::new(
                Scalar::bool(!b, array.dtype().nullability()),
                array.len(),
            )
            .into_array()),
        }
    }
}

register_kernel!(InvertKernelAdapter(ConstantVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability::Nullable;
    use vortex_scalar::Scalar;

    use crate::Array;
    use crate::arrays::ConstantArray;
    use crate::compute::invert;

    #[test]
    fn invert_nullable_const() {
        let constant = ConstantArray::new(Scalar::bool(false, Nullable), 10);

        let inverted = invert(constant.as_ref()).unwrap();
        assert_eq!(inverted.dtype(), constant.dtype());

        let orig = invert(inverted.as_ref()).unwrap();

        assert_eq!(orig.dtype(), constant.dtype());
        assert_eq!(orig.as_constant(), constant.as_constant())
    }
}
