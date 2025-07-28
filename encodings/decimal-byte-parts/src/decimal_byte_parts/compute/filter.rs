// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{FilterKernel, FilterKernelAdapter, filter};
use vortex_array::{ArrayRef, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{DecimalBytePartsArray, DecimalBytePartsVTable};

impl FilterKernel for DecimalBytePartsVTable {
    fn filter(&self, array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef> {
        DecimalBytePartsArray::try_new(filter(&array.msp, mask)?, *array.decimal_dtype())
            .map(|d| d.to_array())
    }
}

register_kernel!(FilterKernelAdapter(DecimalBytePartsVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::arrays::FixedSizeBinaryArray;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_dtype::DecimalDType;

    use crate::DecimalBytePartsArray;

    #[test]
    fn test_filter_decimal_byte_parts() {
        // Create test data with 5 16-byte values for Decimal128
        let bytes: Vec<u8> = (0..5)
            .flat_map(|i| {
                let mut bytes = vec![0u8; 16];
                bytes[0] = i;
                bytes
            })
            .collect();

        let msp = FixedSizeBinaryArray::from_iter(bytes.chunks(16).map(|chunk| chunk.to_vec()), 16)
            .into_array();

        let decimal_dtype = DecimalDType::new(38, 5);
        let array = DecimalBytePartsArray::try_new(msp, decimal_dtype).unwrap();
        test_filter_conformance(array.as_ref());
    }
}
