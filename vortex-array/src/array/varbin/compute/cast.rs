use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::varbin::VarBinArray;
use crate::array::VarBinEncoding;
use crate::compute::CastFn;
use crate::{Array, IntoArray};

impl CastFn<VarBinArray> for VarBinEncoding {
    fn cast(&self, array: &VarBinArray, dtype: &DType) -> VortexResult<Array> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            vortex_bail!("cannot cast {} to {}", array.dtype(), dtype);
        }

        let new_nullability = dtype.nullability();
        let new_validity = array.validity().cast_nullability(new_nullability)?;
        let new_dtype = array.dtype().with_nullability(new_nullability);
        VarBinArray::try_new(array.offsets(), array.bytes(), new_dtype, new_validity)
            .map(IntoArray::into_array)
    }
}
