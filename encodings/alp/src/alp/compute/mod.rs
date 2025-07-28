// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod between;
mod compare;
mod filter;
mod nan_count;

use vortex_array::compute::{TakeKernel, TakeKernelAdapter, MaskKernel, MaskKernelAdapter, take, mask};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ALPArray, ALPVTable};

impl TakeKernel for ALPVTable {
    fn take(&self, array: &ALPArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let taken_encoded = take(array.encoded(), indices)?;
        let taken_patches = array
            .patches()
            .map(|p| p.take(indices))
            .transpose()?
            .flatten()
            .map(|patches| {
                patches.cast_values(
                    &array
                        .dtype()
                        .with_nullability(taken_encoded.dtype().nullability()),
                )
            })
            .transpose()?;
        Ok(ALPArray::try_new(taken_encoded, array.exponents(), taken_patches)?.into_array())
    }
}

register_kernel!(TakeKernelAdapter(ALPVTable).lift());

impl MaskKernel for ALPVTable {
    fn mask(&self, array: &ALPArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let masked_encoded = mask(array.encoded(), mask)?;
        let masked_patches = array
            .patches()
            .map(|p| p.mask(mask))
            .transpose()?
            .flatten()
            .map(|patches| {
                patches.cast_values(
                    &array
                        .dtype()
                        .with_nullability(masked_encoded.dtype().nullability()),
                )
            })
            .transpose()?;
        Ok(ALPArray::try_new(masked_encoded, array.exponents(), masked_patches)?.into_array())
    }
}

register_kernel!(MaskKernelAdapter(ALPVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::filter::test_filter;
    use vortex_array::compute::conformance::mask::test_mask;
    use vortex_array::compute::conformance::binary_numeric::test_numeric;
    use vortex_array::{ToCanonical, IntoArray};
    use vortex_buffer::buffer;
    
    use crate::{ALPEncoding, ALPVTable};
    
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
        let array = PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]);
        let alp = ALPEncoding.encode(array.as_ref(), None).unwrap().unwrap();
        test_filter(alp.as_ref());
    }
    
    #[test]
    fn test_mask_alp_array() {
        // Test with f32 values
        let values = buffer![10.5f32, 20.5, 30.5, 40.5, 50.5];
        let array = values.into_array();
        let alp = ALPEncoding.encode(&array, None).unwrap().unwrap();
        test_mask(alp.as_ref());
        
        // Test with f64 values 
        let values = buffer![1000.123f64, 2000.456, 3000.789, 4000.012, 5000.345];
        let array = values.into_array();
        let alp = ALPEncoding.encode(&array, None).unwrap().unwrap();
        test_mask(alp.as_ref());
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
        
        test_numeric(alp1.as_ref(), alp2.as_ref());
        
        // Test with same arrays
        test_numeric(alp1.as_ref(), alp1.as_ref());
    }
}
