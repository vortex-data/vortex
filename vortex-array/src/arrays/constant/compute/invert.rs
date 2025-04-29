use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ConstantEncoding};
use crate::compute::{InvertKernel, InvertKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl InvertKernel for ConstantEncoding {
    fn invert(&self, array: &ConstantArray) -> VortexResult<ArrayRef> {
        match array.scalar().as_bool().value() {
            None => Ok(array.to_array().into_array()),
            Some(b) => Ok(ConstantArray::new(
                Scalar::bool(!b, array.dtype().nullability()),
                array.len(),
            )
            .into_array()),
        }
    }
}

register_kernel!(InvertKernelAdapter(ConstantEncoding).lift());

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability::Nullable;
    use vortex_scalar::Scalar;

    use crate::arrays::{ConstantArray, ConstantEncoding};
    use crate::compute::InvertKernel;
    use crate::{Array, ArrayStatistics};

    #[test]
    fn invert_nullable_const() {
        let constant = ConstantArray::new(Scalar::bool(false, Nullable), 10);

        let invert = ConstantEncoding.invert(&constant).unwrap();
        assert_eq!(invert.dtype(), constant.dtype());

        let orig = ConstantEncoding
            .invert(invert.as_any().downcast_ref::<ConstantArray>().unwrap())
            .unwrap();

        assert_eq!(orig.dtype(), constant.dtype());
        assert_eq!(orig.as_constant(), constant.as_constant())
    }
}
