use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::VarBinEncoding;
use crate::arrays::varbin::VarBinArray;
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl MaskKernel for VarBinEncoding {
    fn mask(&self, array: &VarBinArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(VarBinArray::try_new(
            array.offsets().clone(),
            array.bytes().clone(),
            array.dtype().as_nullable(),
            array.validity().mask(mask)?,
        )?
        .into_array())
    }
}

register_kernel!(MaskKernelAdapter(VarBinEncoding).lift());

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability};

    use crate::arrays::VarBinArray;
    use crate::compute::conformance::mask::test_mask;

    #[test]
    fn test_mask_var_bin_array() {
        let array = VarBinArray::from_vec(
            vec!["hello", "world", "filter", "good", "bye"],
            DType::Utf8(Nullability::NonNullable),
        );
        test_mask(&array);

        let array = VarBinArray::from_iter(
            vec![Some("hello"), None, Some("filter"), Some("good"), None],
            DType::Utf8(Nullability::Nullable),
        );
        test_mask(&array);
    }
}
