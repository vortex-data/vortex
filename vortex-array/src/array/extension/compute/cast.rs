use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::extension::ExtensionArray;
use crate::array::ExtensionEncoding;
use crate::compute::{try_cast, CastFn};
use crate::{ArrayDType as _, ArrayData, IntoArrayData as _};

impl CastFn<ExtensionArray> for ExtensionEncoding {
    fn cast(&self, array: &ExtensionArray, dtype: &DType) -> VortexResult<ArrayData> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            vortex_bail!("cannot cast from {} to {}", array.dtype(), dtype);
        }
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!(
                "dtype must have extension dtype {} {}",
                array.dtype(),
                dtype
            );
        };
        Ok(ExtensionArray::new(
            ext_dtype.clone(),
            try_cast(array.storage(), ext_dtype.storage_dtype())?,
        )
        .into_array())
    }
}
