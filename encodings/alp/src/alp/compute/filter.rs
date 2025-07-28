// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{FilterKernel, FilterKernelAdapter, filter};
use vortex_array::{ArrayRef, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ALPArray, ALPVTable};

impl FilterKernel for ALPVTable {
    fn filter(&self, array: &ALPArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let patches = array
            .patches()
            .map(|p| p.filter(mask))
            .transpose()?
            .flatten();

        Ok(
            ALPArray::try_new(filter(array.encoded(), mask)?, array.exponents(), patches)?
                .to_array(),
        )
    }
}

register_kernel!(FilterKernelAdapter(ALPVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_numeric;
    use vortex_array::compute::conformance::filter::test_filter;
    use vortex_buffer::buffer;

    use crate::ALPEncoding;

    #[test]
    fn test_filter_alp_array() {
        // Test with f32 values
        let values = buffer![1.23f32, 4.56, 7.89, 10.11, 12.13];
        let array = values.into_array();
        let alp = ALPEncoding.encode(&array, None).unwrap().unwrap();
        test_filter(alp.as_ref());

        // Test with f64 values
        let values = buffer![100.1f64, 200.2, 300.3, 400.4, 500.5];
        let array = values.into_array();
        let alp = ALPEncoding.encode(&array, None).unwrap().unwrap();
        test_filter(alp.as_ref());

        // Test with nullable values
        let array =
            PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]);
        let alp = ALPEncoding.encode(array.as_ref(), None).unwrap().unwrap();
        test_filter(alp.as_ref());
    }

    #[test]
    fn test_numeric_alp_array() {
        // Test binary numeric operations with f32
        let values1 = buffer![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let array1 = values1.into_array();
        let alp1 = ALPEncoding.encode(&array1, None).unwrap().unwrap();

        let values2 = buffer![5.0f32, 10.0, 15.0, 20.0, 25.0];
        let array2 = values2.into_array();
        let alp2 = ALPEncoding.encode(&array2, None).unwrap().unwrap();

        test_numeric(alp1.to_array());
        test_numeric(alp2.to_array());
    }
}
