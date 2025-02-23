use arrow_buffer::BooleanBuffer;
use itertools::Itertools;
use num_traits::{Float, Signed, Unsigned};
use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, PType};
use vortex_error::VortexResult;
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
                match array.ptype() {
                    PType::U8 => sum_unsigned(array.as_slice::<u8>()).into(),
                    PType::U16 => sum_unsigned(array.as_slice::<u16>()).into(),
                    PType::U32 => sum_unsigned(array.as_slice::<u32>()).into(),
                    PType::U64 => sum_unsigned(array.as_slice::<u64>()).into(),
                    PType::I8 => sum_signed(array.as_slice::<i8>()).into(),
                    PType::I16 => sum_signed(array.as_slice::<i16>()).into(),
                    PType::I32 => sum_signed(array.as_slice::<i32>()).into(),
                    PType::I64 => sum_signed(array.as_slice::<i64>()).into(),
                    PType::F16 => sum_float(array.as_slice::<f16>()).into(),
                    PType::F32 => sum_float(array.as_slice::<f32>()).into(),
                    PType::F64 => sum_float(array.as_slice::<f64>()).into(),
                }
            }
            AllOr::None => {
                // All-invalid
                return Ok(Scalar::null(Stat::Sum.dtype(array.dtype())));
            }
            AllOr::Some(validity_mask) => match array.ptype() {
                PType::U8 => {
                    sum_unsigned_with_validity(array.as_slice::<u8>(), &validity_mask).into()
                }
                PType::U16 => {
                    sum_unsigned_with_validity(array.as_slice::<u16>(), &validity_mask).into()
                }
                PType::U32 => {
                    sum_unsigned_with_validity(array.as_slice::<u32>(), &validity_mask).into()
                }
                PType::U64 => {
                    sum_unsigned_with_validity(array.as_slice::<u64>(), &validity_mask).into()
                }
                PType::I8 => {
                    sum_signed_with_validity(array.as_slice::<i8>(), &validity_mask).into()
                }
                PType::I16 => {
                    sum_signed_with_validity(array.as_slice::<i16>(), &validity_mask).into()
                }
                PType::I32 => {
                    sum_signed_with_validity(array.as_slice::<i32>(), &validity_mask).into()
                }
                PType::I64 => {
                    sum_signed_with_validity(array.as_slice::<i64>(), &validity_mask).into()
                }
                PType::F16 => sum_float_with_validity(array, &validity_mask).into(),
                PType::F32 => {
                    sum_float_with_validity(array.as_slice::<f32>(), &validity_mask).into()
                }
                PType::F64 => {
                    sum_float_with_validity(array.as_slice::<f64>(), &validity_mask).into()
                }
            },
        };

        let sum_dtype = Stat::Sum.dtype(array.dtype());
        Ok(Scalar::new(sum_dtype, scalar_value))
    }
}

fn sum_unsigned<T: NativePType + Unsigned>(values: &[T]) -> Option<u64> {
    let mut sum = 0u64;
    for &x in values {
        sum = sum.checked_add(u64::from(x))?;
    }
    Some(sum)
}

fn sum_unsigned_with_validity<T: NativePType + Unsigned>(
    values: &[T],
    validity: &BooleanBuffer,
) -> Option<u64> {
    let mut sum = 0u64;
    for (&x, valid) in values.iter().zip_eq(validity.iter()) {
        if valid {
            sum = sum.checked_add(u64::from(x))?;
        }
    }
    Some(sum)
}

fn sum_signed<T: NativePType + Signed>(values: &[T]) -> Option<i64> {
    let mut sum = 0i64;
    for &x in values {
        sum = sum.checked_add(i64::from(x))?;
    }
    Some(sum)
}

fn sum_signed_with_validity<T: NativePType + Signed>(
    values: &[T],
    validity: &BooleanBuffer,
) -> Option<i64> {
    let mut sum = 0i64;
    for (&x, valid) in values.iter().zip_eq(validity.iter()) {
        if valid {
            sum = sum.checked_add(i64::from(x))?;
        }
    }
    Some(sum)
}

fn sum_float<T: NativePType + Float>(values: &[T]) -> f64 {
    let mut sum = 0.0;
    for &x in values {
        sum += f64::from(x);
    }
    sum
}

fn sum_float_with_validity<T: NativePType + Float>(array: &[T], validity: &BooleanBuffer) -> f64 {
    let mut sum = 0.0;
    for (&x, valid) in array.iter().zip_eq(validity.iter()) {
        if valid {
            sum += f64::from(x);
        }
    }
    sum
}
