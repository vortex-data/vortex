use vortex_array::compute::{try_cast, CastFn};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl CastFn<DateTimePartsArray> for DateTimePartsEncoding {
    fn cast(&self, array: &DateTimePartsArray, dtype: &DType) -> VortexResult<ArrayData> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            vortex_bail!("cannot cast from {} to {}", array.dtype(), dtype);
        };

        Ok(DateTimePartsArray::try_new(
            array.dtype().clone().as_nullable(),
            try_cast(
                array.days().as_ref(),
                &array.days().dtype().with_nullability(dtype.nullability()),
            )?,
            array.seconds(),
            array.subsecond(),
        )?
        .into_array())
    }
}
