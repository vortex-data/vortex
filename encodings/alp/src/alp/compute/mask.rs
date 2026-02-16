// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_array::ArrayRef;
use vortex_array::compute::MaskReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::ALPArray;
use crate::ALPVTable;

impl MaskReduce for ALPVTable {
    fn mask(array: &ALPArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // Masking sparse patches requires reading indices, fall back to kernel.
        if array.patches().is_some() {
            return Ok(None);
        }
        let vortex_mask = Validity::Array(mask.clone()).to_mask(array.len()).not();
        let masked_encoded = array.encoded().mask(&vortex_mask)?;
        Ok(Some(
            ALPArray::new(masked_encoded, array.exponents(), None).to_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_buffer::buffer;

    use crate::alp_encode;

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
        let alp = alp_encode(&array.to_primitive(), None).unwrap();
        test_mask_conformance(alp.as_ref());
    }
}
