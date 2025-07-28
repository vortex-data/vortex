// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{MaskKernel, MaskKernelAdapter, mask};
use vortex_array::{ArrayRef, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ALPArray, ALPVTable};

impl MaskKernel for ALPVTable {
    fn mask(&self, array: &ALPArray, filter_mask: &Mask) -> VortexResult<ArrayRef> {
        let masked_encoded = mask(array.encoded(), filter_mask)?;
        let masked_patches = array
            .patches()
            .map(|p| p.filter(filter_mask))
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
        Ok(ALPArray::try_new(masked_encoded, array.exponents(), masked_patches)?.to_array())
    }
}

register_kernel!(MaskKernelAdapter(ALPVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_buffer::buffer;

    use crate::ALPEncoding;

    #[test]
    fn test_mask_alp_array() {
        // Test with f32 values
        let values = buffer![10.5f32, 20.5, 30.5, 40.5, 50.5];
        let array = values.into_array();
        let alp = ALPEncoding
            .encode(&array.to_canonical().unwrap(), None)
            .unwrap()
            .unwrap();
        test_mask_conformance(alp.as_ref());

        // Test with f64 values
        let values = buffer![1000.123f64, 2000.456, 3000.789, 4000.012, 5000.345];
        let array = values.into_array();
        let alp = ALPEncoding
            .encode(&array.to_canonical().unwrap(), None)
            .unwrap()
            .unwrap();
        test_mask_conformance(alp.as_ref());
    }
}
