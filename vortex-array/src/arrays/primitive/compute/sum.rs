// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::CheckedAdd;
use num_traits::Float;
use num_traits::ToPrimitive;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::compute::SumKernel;
use crate::compute::SumKernelAdapter;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::match_each_native_ptype;
use crate::register_kernel;
use crate::scalar::Scalar;

impl SumKernel for PrimitiveVTable {
    fn sum(&self, array: &PrimitiveArray, accumulator: &Scalar) -> VortexResult<Scalar> {
        let array_sum_scalar = match array.validity_mask()?.bit_buffer() {
            AllOr::All => {
                // All-valid
                match_each_native_ptype!(
                    array.ptype(),
                    unsigned: |T| {
                        Scalar::from(sum_integer::<_, u64>(
                            array.as_slice::<T>(),
                            accumulator.as_primitive().as_::<u64>().vortex_expect("cannot be null"),
                        ))
                    },
                    signed: |T| {
                        Scalar::from(sum_integer::<_, i64>(
                            array.as_slice::<T>(),
                            accumulator.as_primitive().as_::<i64>().vortex_expect("cannot be null"),
                        ))
                    },
                    floating: |T| {
                        Scalar::primitive(
                            sum_float(
                                array.as_slice::<T>(),
                                accumulator.as_primitive().as_::<f64>().vortex_expect("cannot be null"),
                            ),
                            Nullability::Nullable,
                        )
                    }
                )
            }
            AllOr::None => {
                // All-invalid, return accumulator
                return Ok(accumulator.clone());
            }
            AllOr::Some(validity_mask) => {
                // Some-valid
                match_each_native_ptype!(
                    array.ptype(),
                    unsigned: |T| {
                        Scalar::from(sum_integer_with_validity::<_, u64>(
                            array.as_slice::<T>(),
                            validity_mask,
                            accumulator.as_primitive().as_::<u64>().vortex_expect("cannot be null"),
                        ))
                    },
                    signed: |T| {
                        Scalar::from(sum_integer_with_validity::<_, i64>(
                            array.as_slice::<T>(),
                            validity_mask,
                            accumulator.as_primitive().as_::<i64>().vortex_expect("cannot be null"),
                        ))
                    },
                    floating: |T| {
                        Scalar::primitive(
                            sum_float_with_validity(
                                array.as_slice::<T>(),
                                validity_mask,
                                accumulator.as_primitive().as_::<f64>().vortex_expect("cannot be null"),
                            ),
                            Nullability::Nullable,
                        )
                    }
                )
            }
        };

        Ok(array_sum_scalar)
    }
}

register_kernel!(SumKernelAdapter(PrimitiveVTable).lift());

fn sum_integer<T: NativePType + ToPrimitive, R: NativePType + CheckedAdd>(
    values: &[T],
    accumulator: R,
) -> Option<R> {
    let mut sum = accumulator;
    for &x in values {
        sum = sum.checked_add(&R::from(x)?)?;
    }
    Some(sum)
}

fn sum_integer_with_validity<T: NativePType + ToPrimitive, R: NativePType + CheckedAdd>(
    values: &[T],
    validity: &BitBuffer,
    accumulator: R,
) -> Option<R> {
    let mut sum: R = accumulator;
    for (&x, valid) in values.iter().zip_eq(validity.iter()) {
        if valid {
            sum = sum.checked_add(&R::from(x)?)?;
        }
    }
    Some(sum)
}

fn sum_float<T: NativePType + Float>(values: &[T], accumulator: f64) -> f64 {
    let mut sum = accumulator;
    for &x in values {
        sum += x.to_f64().vortex_expect("Failed to cast value to f64");
    }
    sum
}

fn sum_float_with_validity<T: NativePType + Float>(
    array: &[T],
    validity: &BitBuffer,
    accumulator: f64,
) -> f64 {
    let mut sum = accumulator;
    for (&x, valid) in array.iter().zip_eq(validity.iter()) {
        if valid {
            sum += x.to_f64().vortex_expect("Failed to cast value to f64");
        }
    }
    sum
}
