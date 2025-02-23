use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::varbin::VarBinArray;
use crate::arrays::VarBinEncoding;
use crate::compute::MaskFn;
use crate::{Array, ArrayRef, IntoArray};

impl MaskFn<&VarBinArray> for VarBinEncoding {
    fn mask(&self, array: &VarBinArray, mask: Mask) -> VortexResult<ArrayRef> {
        Ok(VarBinArray::try_new(
            array.offsets().clone(),
            array.bytes().clone(),
            array.dtype().as_nullable(),
            array.validity().mask(&mask)?,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability};

    use crate::array::Array;
    use crate::arrays::VarBinArray;
    use crate::compute::test_harness::test_mask;
    use crate::IntoArray as _;

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
