// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::ALPRD;

impl TakeExecute for ALPRD {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let taken_left_parts = array.left_parts().take(indices.clone())?;
        let left_parts_exceptions = array
            .left_parts_patches()
            .map(|patches| patches.take(indices, ctx))
            .transpose()?
            .flatten()
            .map(|p| {
                let values_dtype = p
                    .values()
                    .dtype()
                    .with_nullability(taken_left_parts.dtype().nullability());
                p.cast_values(&values_dtype)
            })
            .transpose()?;
        let right_parts = array
            .right_parts()
            .take(indices.clone())?
            .fill_null(Scalar::zero_value(array.right_parts().dtype()))?;

        Ok(Some(
            ALPRD::try_new(
                array
                    .dtype()
                    .with_nullability(taken_left_parts.dtype().nullability()),
                taken_left_parts,
                array.left_parts_dictionary().clone(),
                right_parts,
                array.right_bit_width(),
                left_parts_exceptions,
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::take::test_take_conformance;

    use crate::ALPRDFloat;
    use crate::RDEncoder;

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_take<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        use vortex_array::IntoArray as _;
        use vortex_buffer::buffer;

        let array = PrimitiveArray::from_iter([a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(&array);

        assert!(encoded.left_parts_patches().is_some());
        assert!(
            encoded
                .left_parts_patches()
                .unwrap()
                .dtype()
                .is_unsigned_int()
        );

        let taken = encoded
            .take(buffer![0, 2].into_array())
            .unwrap()
            .to_primitive();

        assert_arrays_eq!(taken, PrimitiveArray::from_iter([a, outlier]));
    }

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn take_with_nulls<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let array = PrimitiveArray::from_iter([a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(&array);

        assert!(encoded.left_parts_patches().is_some());
        assert!(
            encoded
                .left_parts_patches()
                .unwrap()
                .dtype()
                .is_unsigned_int()
        );

        let taken = encoded
            .take(PrimitiveArray::from_option_iter([Some(0), Some(2), None]).into_array())
            .unwrap()
            .to_primitive();

        assert_arrays_eq!(
            taken,
            PrimitiveArray::from_option_iter([Some(a), Some(outlier), None])
        );
    }

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_take_conformance_alprd<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        test_take_conformance(
            &RDEncoder::new(&[a, b])
                .encode(&PrimitiveArray::from_iter([a, b, outlier, b, outlier]))
                .into_array(),
        );
    }

    #[rstest]
    #[case(0.1f32, 3e25f32)]
    #[case(0.5f64, 1e100f64)]
    fn test_take_with_nulls_conformance<T: ALPRDFloat>(#[case] a: T, #[case] outlier: T) {
        test_take_conformance(
            &RDEncoder::new(&[a])
                .encode(&PrimitiveArray::from_option_iter([
                    Some(a),
                    None,
                    Some(outlier),
                    Some(a),
                    None,
                ]))
                .into_array(),
        );
    }
}
