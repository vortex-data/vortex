use vortex_array::compute::{mask, try_cast, FilterMask, MaskFn};
use vortex_array::{ArrayDType as _, ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::{ALPArray, ALPEncoding};

impl MaskFn<ALPArray> for ALPEncoding {
    fn mask(&self, array: &ALPArray, filter_mask: FilterMask) -> VortexResult<ArrayData> {
        ALPArray::try_new(
            mask(&array.encoded(), filter_mask)?,
            array.exponents(),
            array
                .patches()
                .map(|patches| {
                    patches.map_values(|values| try_cast(&values, &values.dtype().as_nullable()))
                })
                .transpose()?,
        )
        .map(IntoArrayData::into_array)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::test_harness::test_mask;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayData as _;
    use vortex_buffer::buffer;

    use crate::alp_encode;

    #[test]
    fn test_mask_no_patches_alp_array() {
        test_mask(
            alp_encode(&PrimitiveArray::new(
                buffer![1.0f32, 2.0, 3.0, 4.0, 5.0],
                Validity::AllValid,
            ))
            .unwrap()
            .into_array(),
        );

        test_mask(
            alp_encode(&PrimitiveArray::new(
                buffer![1.0f32, 2.0, 3.0, 4.0, 5.0],
                Validity::NonNullable,
            ))
            .unwrap()
            .into_array(),
        );
    }

    #[test]
    fn test_mask_patched_alp_array() {
        let alp_array = alp_encode(&PrimitiveArray::new(
            buffer![1.0f32, 2.0, 3.0, 4.0, 1e10],
            Validity::AllValid,
        ))
        .unwrap();
        assert!(alp_array.patches().is_some());
        test_mask(alp_array.into_array());

        let alp_array = alp_encode(&PrimitiveArray::new(
            buffer![1.0f32, 2.0, 3.0, 4.0, 1e10],
            Validity::NonNullable,
        ))
        .unwrap();
        assert!(alp_array.patches().is_some());
        test_mask(alp_array.into_array());
    }
}
