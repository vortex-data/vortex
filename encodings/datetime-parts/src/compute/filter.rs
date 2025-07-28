// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{FilterKernel, FilterKernelAdapter, filter};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{DateTimePartsArray, DateTimePartsVTable};

impl FilterKernel for DateTimePartsVTable {
    fn filter(&self, array: &DateTimePartsArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            filter(array.days().as_ref(), mask)?,
            filter(array.seconds().as_ref(), mask)?,
            filter(array.subseconds().as_ref(), mask)?,
        )?
        .into_array())
    }
}
register_kernel!(FilterKernelAdapter(DateTimePartsVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::filter::test_filter;
    use vortex_dtype::{DType, Nullability, PType};
    
    use crate::DateTimePartsArray;
    
    #[test]
    fn test_filter_datetime_parts() {
        // Create a datetime parts array with days, seconds, and subseconds
        let days = PrimitiveArray::from_iter([0i64, 1, 2, 3, 4]).into_array();
        let seconds = PrimitiveArray::from_iter([0i64, 3600, 7200, 10800, 14400]).into_array();
        let subseconds = PrimitiveArray::from_iter([0i64, 500_000_000, 0, 250_000_000, 750_000_000]).into_array();
        
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let array = DateTimePartsArray::try_new(dtype, days, seconds, subseconds).unwrap();
        test_filter(array.as_ref());
        
        // Test with nullable values
        let days = PrimitiveArray::from_option_iter([Some(0i64), None, Some(2), Some(3), None]).into_array();
        let seconds = PrimitiveArray::from_option_iter([Some(0i64), Some(3600), None, Some(10800), Some(14400)]).into_array();
        let subseconds = PrimitiveArray::from_option_iter([Some(0i64), Some(500_000_000), Some(0), None, None]).into_array();
        
        let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
        let array = DateTimePartsArray::try_new(dtype, days, seconds, subseconds).unwrap();
        test_filter(array.as_ref());
    }
}
