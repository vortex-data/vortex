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
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_buffer::buffer;

    use crate::ALPEncoding;

    #[rstest]
    #[case(buffer![10.5f32, 20.5, 30.5, 40.5, 50.5].into_array())]
    #[case(buffer![1000.123f64, 2000.456, 3000.789, 4000.012, 5000.345].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]).into_array())]
    #[case(buffer![99.99f64].into_array())]
    #[case(buffer![
        0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0
    ].into_array())]
    fn test_mask_alp_conformance(#[case] array: vortex_array::ArrayRef) {
        let alp = ALPEncoding
            .encode(&array.to_canonical().unwrap(), None)
            .unwrap()
            .unwrap();
        test_mask_conformance(alp.as_ref());
    }
}
