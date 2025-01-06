use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::varbin::VarBinArray;
use crate::array::VarBinEncoding;
use crate::compute::CastFn;
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl CastFn<VarBinArray> for VarBinEncoding {
    fn cast(&self, array: &VarBinArray, dtype: &DType) -> VortexResult<ArrayData> {
        match dtype {
            DType::Utf8(nullability) => {
                let validity = array.validity().with_nullability(*nullability)?;
                VarBinArray::try_new(
                    array.offsets(),
                    array.bytes(),
                    array.dtype().with_nullability(*nullability),
                    validity,
                )
                .map(IntoArrayData::into_array)
            }
            _ => vortex_bail!("cannot cast {} to {}", array.dtype(), dtype),
        }
    }
}
