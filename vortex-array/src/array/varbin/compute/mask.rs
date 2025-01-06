use vortex_error::VortexResult;

use crate::array::varbin::VarBinArray;
use crate::array::VarBinEncoding;
use crate::compute::{FilterMask, MaskFn};
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl MaskFn<VarBinArray> for VarBinEncoding {
    fn mask(&self, array: &VarBinArray, mask: FilterMask) -> VortexResult<ArrayData> {
        VarBinArray::try_new(
            array.offsets(),
            array.bytes(),
            array.dtype().as_nullable(),
            array.validity().mask(&mask)?,
        )
        .map(IntoArrayData::into_array)
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability};

    use crate::array::VarBinArray;
    use crate::compute::test_harness::test_mask;
    use crate::IntoArrayData as _;

    #[test]
    fn test_mask_var_bin_array() {
        let array = VarBinArray::from_vec(
            vec!["hello", "world", "filter", "good", "bye"],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        test_mask(array);

        let array = VarBinArray::from_iter(
            vec![Some("hello"), None, Some("filter"), Some("good"), None],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        test_mask(array);
    }
}
