// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::compute::IsConstantKernel;
use crate::compute::IsConstantKernelAdapter;
use crate::compute::IsConstantOpts;
use crate::match_each_decimal_value_type;
use crate::register_kernel;

impl IsConstantKernel for DecimalVTable {
    fn is_constant(
        &self,
        array: &DecimalArray,
        _opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        let constant = match_each_decimal_value_type!(array.values_type, |S| {
            array.buffer::<S>().iter().all_equal()
        });
        Ok(Some(constant))
    }
}

register_kernel!(IsConstantKernelAdapter(DecimalVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::arrays::DecimalArray;
    use crate::compute::is_constant;
    use crate::dtype::DecimalDType;
    use crate::validity::Validity;

    #[test]
    fn test_is_constant() {
        let array = DecimalArray::new(
            buffer![0i128, 1i128, 2i128],
            DecimalDType::new(19, 0),
            Validity::NonNullable,
        );

        assert!(!is_constant(&array.to_array()).unwrap().unwrap());

        let array = DecimalArray::new(
            buffer![100i128, 100i128, 100i128],
            DecimalDType::new(19, 0),
            Validity::NonNullable,
        );

        assert!(is_constant(&array.to_array()).unwrap().unwrap());
    }
}
