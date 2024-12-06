#[cfg(test)]
mod test {
    use vortex_scalar::Scalar;

    use crate::array::primitive::PrimitiveArray;
    use crate::compute::{scalar_at, sub_scalar};
    use crate::{ArrayLen as _, IntoArrayData, IntoCanonical};

    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = vec![1u16, 2, 3].into_array();
        let results = sub_scalar(&values, 1u16.into())
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap()
            .maybe_null_slice::<u16>()
            .to_vec();
        assert_eq!(results, &[0u16, 1, 2]);
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let values = vec![1i64, 2, 3].into_array();
        let results = sub_scalar(&values, (-1i64).into())
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap()
            .maybe_null_slice::<i64>()
            .to_vec();
        assert_eq!(results, &[2i64, 3, 4]);
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let values = PrimitiveArray::from_nullable_vec(vec![Some(1u16), Some(2), None, Some(3)])
            .into_array();
        let result = sub_scalar(&values, Some(1u16).into())
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap();

        let actual = (0..result.len())
            .map(|index| scalar_at(&result, index).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                Scalar::from(Some(0u16)),
                Scalar::from(Some(1u16)),
                Scalar::from(None::<u16>),
                Scalar::from(Some(2u16))
            ]
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let values = vec![1.0f64, 2.0, 3.0].into_array();
        let to_subtract = -1f64;
        let results = sub_scalar(&values, to_subtract.into())
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap()
            .maybe_null_slice::<f64>()
            .to_vec();
        assert_eq!(results, &[2.0f64, 3.0, 4.0]);
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        let values = vec![f32::MIN, 2.0, 3.0].into_array();
        let _results = sub_scalar(&values, 1.0f32.into()).unwrap();
        let _results = sub_scalar(&values, f32::MAX.into()).unwrap();
    }

    #[test]
    fn test_scalar_subtract_type_mismatch_fails() {
        let values = vec![1u64, 2, 3].into_array();
        // Subtracting incompatible dtypes should fail
        let _results =
            sub_scalar(&values, 1.5f64.into()).expect_err("Expected type mismatch error");
    }
}
