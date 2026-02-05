// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::TakeExecute;
use vortex_array::arrays::TakeExecuteAdaptor;
use vortex_array::compute::fill_null;
use vortex_array::compute::take;
use vortex_array::kernel::ParentKernelSet;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;
use vortex_scalar::ScalarValue;

use crate::ALPRDArray;
use crate::ALPRDVTable;

fn take_alprd(array: &ALPRDArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
    let taken_left_parts = take(array.left_parts(), indices)?;
    let left_parts_exceptions = array
        .left_parts_patches()
        .map(|patches| patches.take(indices))
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
    let right_parts = fill_null(
        &take(array.right_parts(), indices)?,
        &Scalar::new(array.right_parts().dtype().clone(), ScalarValue::from(0)),
    )?;

    Ok(ALPRDArray::try_new(
        array
            .dtype()
            .with_nullability(taken_left_parts.dtype().nullability()),
        taken_left_parts,
        array.left_parts_dictionary().clone(),
        right_parts,
        array.right_bit_width(),
        left_parts_exceptions,
    )?
    .into_array())
}

impl TakeExecute for ALPRDVTable {
    fn take(
        array: &ALPRDArray,
        indices: &dyn Array,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        take_alprd(array, indices).map(Some)
    }
}

impl ALPRDVTable {
    pub const TAKE_KERNELS: ParentKernelSet<Self> =
        ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor::<Self>(Self))]);
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::compute::take;

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

        let taken = take(encoded.as_ref(), buffer![0, 2].into_array().as_ref())
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

        let taken = take(
            encoded.as_ref(),
            PrimitiveArray::from_option_iter([Some(0), Some(2), None]).as_ref(),
        )
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
                .to_array(),
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
                .to_array(),
        );
    }
}
