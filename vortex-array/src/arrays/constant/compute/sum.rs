// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::{CheckedMul, ToPrimitive};
use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{FromPrimitiveOrF16, PrimitiveScalar, Scalar, ScalarValue};

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::compute::{SumKernel, SumKernelAdapter};
use crate::register_kernel;
use crate::stats::Stat;

impl SumKernel for ConstantVTable {
    fn sum(&self, array: &ConstantArray) -> VortexResult<Scalar> {
        // Compute the expected dtype of the sum.
        let sum_dtype = Stat::Sum
            .dtype(array.dtype())
            .ok_or_else(|| vortex_err!("Sum not supported for dtype {}", array.dtype()))?;

        let scalar = array.scalar();
        let sum_value = match scalar.dtype() {
            DType::Bool(_) => ScalarValue::from(match scalar.as_bool().value() {
                None | Some(false) => 0u64,
                Some(true) => array.len() as u64,
            }),
            DType::Primitive(ptype, _) => {
                match_each_native_ptype!(
                    ptype,
                    unsigned: |T| { sum_integral::<i64>(scalar.as_primitive(), array.len())?.into() },
                    signed: |T| { sum_integral::<u64>(scalar.as_primitive(), array.len())?.into() },
                    floating: |T| { sum_float(scalar.as_primitive(), array.len())?.into() }
                )
            }
            _ => vortex_bail!("Unsupported dtype for sum: {}", scalar.dtype()),
        };

        Ok(Scalar::new(sum_dtype, sum_value))
    }
}

fn sum_integral<T>(
    primitive_scalar: PrimitiveScalar<'_>,
    array_len: usize,
) -> VortexResult<Option<T>>
where
    T: FromPrimitiveOrF16 + NativePType + CheckedMul,
    Scalar: From<Option<T>>,
{
    let v = primitive_scalar.as_::<T>()?;
    let array_len =
        T::from(array_len).ok_or_else(|| vortex_err!("array_len must fit the sum type"))?;
    let sum = v.and_then(|v| v.checked_mul(&array_len));

    Ok(sum)
}

fn sum_float(primitive_scalar: PrimitiveScalar<'_>, array_len: usize) -> VortexResult<Option<f64>> {
    let v = primitive_scalar.as_::<f64>()?;
    let array_len = array_len
        .to_f64()
        .ok_or_else(|| vortex_err!("array_len must fit the sum type"))?;

    Ok(v.map(|v| v * array_len))
}

register_kernel!(SumKernelAdapter(ConstantVTable).lift());
