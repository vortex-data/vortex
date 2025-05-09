use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::PrimitiveEncoding;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl MaskKernel for PrimitiveEncoding {
    fn mask(&self, array: &PrimitiveArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let validity = array.validity().mask(mask)?;
        Ok(
            PrimitiveArray::from_byte_buffer(array.byte_buffer().clone(), array.ptype(), validity)
                .into_array(),
        )
    }
}

register_kernel!(MaskKernelAdapter(PrimitiveEncoding).lift());

#[cfg(test)]
mod test {
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::mask::test_mask;

    #[test]
    fn test_mask_non_nullable_array() {
        let non_nullable_array = PrimitiveArray::from_iter([1, 2, 3, 4, 5]);
        test_mask(&non_nullable_array);
    }
}
