// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::alp::{ALPArray, ALPVTable};

impl CastKernel for ALPVTable {
    fn cast(&self, array: &ALPArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Check if this is just a nullability change
        if array.dtype().eq_ignore_nullability(dtype) {
            // For nullability-only changes, we can avoid decoding
            // Cast the encoded array (integers) to handle nullability
            let new_encoded = cast(
                array.encoded(),
                &array
                    .encoded()
                    .dtype()
                    .with_nullability(dtype.nullability()),
            )?;

            // SAFETY: casting nullability doesn't alter the invariants
            unsafe {
                Ok(Some(
                    ALPArray::new_unchecked(
                        new_encoded,
                        array.exponents(),
                        array.patches().cloned(),
                        dtype.clone(),
                    )
                    .into_array(),
                ))
            }
        } else {
            Ok(None)
        }
    }
}

register_kernel!(CastKernelAdapter(ALPVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::ALPEncoding;

    #[test]
    fn test_cast_alp_f32_to_f64() {
        let values = buffer![1.5f32, 2.5, 3.5, 4.5].into_array();
        let alp = ALPEncoding
            .encode(&values.to_canonical().unwrap(), None)
            .unwrap()
            .unwrap();

        let casted = cast(
            alp.as_ref(),
            &DType::Primitive(PType::F64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::F64, Nullability::NonNullable)
        );

        let decoded = casted.to_canonical().unwrap().into_primitive().unwrap();
        let values = decoded.as_slice::<f64>();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 1.5).abs() < f64::EPSILON);
        assert!((values[1] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cast_alp_to_int() {
        let values = buffer![1.0f32, 2.0, 3.0, 4.0].into_array();
        let alp = ALPEncoding
            .encode(&values.to_canonical().unwrap(), None)
            .unwrap()
            .unwrap();

        let casted = cast(
            alp.as_ref(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        let decoded = casted.to_canonical().unwrap().into_primitive().unwrap();
        assert_eq!(decoded.as_slice::<i32>(), &[1i32, 2, 3, 4]);
    }

    #[rstest]
    #[case(buffer![1.23f32, 4.56, 7.89, 10.11, 12.13].into_array())]
    #[case(buffer![100.1f64, 200.2, 300.3, 400.4, 500.5].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]).into_array())]
    #[case(buffer![42.42f64].into_array())]
    #[case(buffer![0.0f32, -1.5, 2.5, -3.5, 4.5].into_array())]
    fn test_cast_alp_conformance(#[case] array: vortex_array::ArrayRef) {
        let alp = ALPEncoding
            .encode(&array.to_canonical().unwrap(), None)
            .unwrap()
            .unwrap();
        test_cast_conformance(alp.as_ref());
    }
}
