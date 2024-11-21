use itertools::Itertools;
use num_traits::WrappingSub;
use vortex_dtype::{match_each_float_ptype, match_each_integer_ptype, NativePType};
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_scalar::{PrimitiveScalar, Scalar};

use crate::array::constant::ConstantArray;
use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::unary::SubtractScalarFn;
use crate::validity::ArrayValidity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, ArrayLen, IntoArrayData};

impl SubtractScalarFn<PrimitiveArray> for PrimitiveEncoding {
    fn subtract_scalar(
        &self,
        array: &PrimitiveArray,
        to_subtract: &Scalar,
    ) -> VortexResult<ArrayData> {
        if array.dtype() != to_subtract.dtype() {
            vortex_bail!(MismatchedTypes: array.dtype(), to_subtract.dtype())
        }

        let validity = array.validity().to_logical(array.len());
        if validity.all_invalid() {
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            );
        }

        let result = if to_subtract.dtype().is_int() {
            match_each_integer_ptype!(array.ptype(), |$T| {
                let to_subtract: $T = PrimitiveScalar::try_from(to_subtract)?
                    .typed_value::<$T>()
                    .ok_or_else(|| vortex_err!("expected primitive"))?;
                subtract_scalar_integer::<$T>(array, to_subtract)?
            })
        } else {
            match_each_float_ptype!(array.ptype(), |$T| {
                let to_subtract: $T = PrimitiveScalar::try_from(to_subtract)?
                    .typed_value::<$T>()
                    .ok_or_else(|| vortex_err!("expected primitive"))?;
                let sub_vec : Vec<$T> = array.maybe_null_slice::<$T>()
                .iter()
                .map(|&v| v - to_subtract).collect_vec();
                PrimitiveArray::from(sub_vec)
            })
        };
        Ok(result.into_array())
    }
}

fn subtract_scalar_integer<T: NativePType + WrappingSub>(
    subtract_from: &PrimitiveArray,
    to_subtract: T,
) -> VortexResult<PrimitiveArray> {
    if to_subtract.is_zero() {
        // if to_subtract is zero, skip operation
        return Ok(subtract_from.clone());
    }

    let contains_nulls = !subtract_from.logical_validity().all_valid();
    let subtraction_result = if contains_nulls {
        let sub_vec = subtract_from
            .maybe_null_slice()
            .iter()
            .map(|&v: &T| v.wrapping_sub(&to_subtract))
            .collect_vec();
        PrimitiveArray::from_vec(sub_vec, subtract_from.validity())
    } else {
        PrimitiveArray::from(
            subtract_from
                .maybe_null_slice::<T>()
                .iter()
                .map(|&v| v - to_subtract)
                .collect_vec(),
        )
    };
    Ok(subtraction_result)
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::array::primitive::PrimitiveArray;
    use crate::compute::unary::subtract_scalar;
    use crate::{ArrayLen, IntoArrayData, IntoArrayVariant};

    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = vec![1u16, 2, 3].into_array();
        let results = subtract_scalar(&values, &1u16.into())
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
        let results = subtract_scalar(&values, &(-1i64).into())
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
        let flattened = subtract_scalar(&values, &Some(1u16).into())
            .unwrap()
            .into_primitive()
            .unwrap();

        let results = flattened.maybe_null_slice::<u16>().to_vec();
        assert_eq!(results, &[0u16, 1, 65535, 2]);
        let valid_indices = flattened
            .validity()
            .to_logical(flattened.len())
            .to_null_buffer()
            .unwrap()
            .unwrap()
            .valid_indices()
            .collect_vec();
        assert_eq!(valid_indices, &[0, 1, 3]);
    }

    #[test]
    fn test_scalar_subtract_float() {
        let values = vec![1.0f64, 2.0, 3.0].into_array();
        let to_subtract = -1f64;
        let results = subtract_scalar(&values, &to_subtract.into())
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
        let _results = subtract_scalar(&values, &1.0f32.into()).unwrap();
        let _results = subtract_scalar(&values, &f32::MAX.into()).unwrap();
    }

    #[test]
    fn test_scalar_subtract_type_mismatch_fails() {
        let values = vec![1u64, 2, 3].into_array();
        // Subtracting incompatible dtypes should fail
        let _results =
            subtract_scalar(&values, &1.5f64.into()).expect_err("Expected type mismatch error");
    }
}
