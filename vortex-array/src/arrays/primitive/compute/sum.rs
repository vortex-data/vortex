use arrow_buffer::BooleanBuffer;
use itertools::Itertools;
use num_traits::{CheckedAdd, Float, ToPrimitive};
use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::AllOr;
use vortex_scalar::Scalar;

use crate::arrays::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::SumFn;
use crate::stats::Stat;
use crate::variants::PrimitiveArrayTrait;
use crate::Array;

impl SumFn<&PrimitiveArray> for PrimitiveEncoding {
    fn sum(&self, array: &PrimitiveArray) -> VortexResult<Scalar> {
        let scalar_value = match array.validity_mask()?.boolean_buffer() {
            AllOr::All => {
                // All-valid
                match_each_native_ptype!(
                    array.ptype(),
                    unsigned: |$T| { sum_integer::<_, u64>(array.as_slice::<$T>()).into() }
                    signed: |$T| { sum_integer::<_, i64>(array.as_slice::<$T>()).into() }
                    floating: |$T| { sum_float(array.as_slice::<$T>()).into() }
                )
            }
            AllOr::None => {
                // All-invalid
                return Ok(Scalar::null(Stat::Sum.dtype(array.dtype())));
            }
            AllOr::Some(validity_mask) => {
                // Some-valid
                match_each_native_ptype!(
                    array.ptype(),
                    unsigned: |$T| {
                        sum_integer_with_validity::<_, u64>(array.as_slice::<$T>(), validity_mask)
                            .into()
                    }
                    signed: |$T| {
                        sum_integer_with_validity::<_, i64>(array.as_slice::<$T>(), validity_mask)
                            .into()
                    }
                    floating: |$T| {
                        sum_float_with_validity(array.as_slice::<$T>(), validity_mask).into()
                    }
                )
            }
        };

        let sum_dtype = Stat::Sum.dtype(array.dtype());
        Ok(Scalar::new(sum_dtype, scalar_value))
    }
}

fn sum_integer<T: NativePType + ToPrimitive, R: NativePType + CheckedAdd>(
    values: &[T],
) -> Option<R> {
    let mut sum = R::zero();
    for &x in values {
        sum = sum.checked_add(&R::from(x)?)?;
    }
    Some(sum)
}

fn sum_integer_with_validity<T: NativePType + ToPrimitive, R: NativePType + CheckedAdd>(
    values: &[T],
    validity: &BooleanBuffer,
) -> Option<R> {
    let mut sum = R::zero();
    for (&x, valid) in values.iter().zip_eq(validity.iter()) {
        if valid {
            sum = sum.checked_add(&R::from(x)?)?;
        }
    }
    Some(sum)
}

fn sum_float<T: NativePType + Float>(values: &[T]) -> f64 {
    let mut sum = 0.0;
    for &x in values {
        sum += x.to_f64().vortex_expect("Failed to cast value to f64");
    }
    sum
}

fn sum_float_with_validity<T: NativePType + Float>(array: &[T], validity: &BooleanBuffer) -> f64 {
    let mut sum = 0.0;
    for (&x, valid) in array.iter().zip_eq(validity.iter()) {
        if valid {
            sum += x.to_f64().vortex_expect("Failed to cast value to f64");
        }
    }
    sum
}
