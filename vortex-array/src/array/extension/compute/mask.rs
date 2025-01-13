use std::sync::Arc;

use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexResult};

use crate::array::extension::ExtensionArray;
use crate::array::ExtensionEncoding;
use crate::compute::{mask, FilterMask, MaskFn};
use crate::{ArrayDType as _, ArrayData, IntoArrayData};

impl MaskFn<ExtensionArray> for ExtensionEncoding {
    fn mask(&self, array: &ExtensionArray, filter_mask: FilterMask) -> VortexResult<ArrayData> {
        let DType::Extension(ext_dtype) = array.dtype() else {
            vortex_bail!("extension array must have extension dtype");
        };
        Ok(ExtensionArray::new(
            Arc::from(ext_dtype.with_nullability(Nullability::Nullable)),
            mask(&array.storage(), filter_mask)?,
        )
        .into_array())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_buffer::buffer;
    use vortex_dtype::{DType, ExtDType, ExtID, PType};

    use crate::array::ExtensionArray;
    use crate::compute::test_harness::test_mask;
    use crate::IntoArrayData as _;

    #[test]
    fn test_mask_extension_array() {
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("timestamp".into()),
            DType::from(PType::I64).into(),
            None,
        ));

        test_mask(
            ExtensionArray::new(ext_dtype, buffer![1i64, 2, 3, 4, 5].into_array()).into_array(),
        );
    }
}
