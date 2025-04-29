use num_traits::PrimInt;
use vortex_dtype::{NativePType, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_scalar::{FromPrimitiveOrF16, Scalar};

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{SumKernel, SumKernelAdapter, sum};
use crate::stats::Stat;
use crate::{Array, ArrayRef, register_kernel};

impl SumKernel for ChunkedEncoding {
    fn sum(&self, array: &ChunkedArray) -> VortexResult<Scalar> {
        let sum_dtype = Stat::Sum
            .dtype(array.dtype())
            .ok_or_else(|| vortex_err!("Sum not supported for dtype {}", array.dtype()))?;
        let sum_ptype = PType::try_from(&sum_dtype).vortex_expect("sum dtype must be primitive");

        let scalar_value = match_each_native_ptype!(
            sum_ptype,
            unsigned: |$T| { sum_int::<u64>(array.chunks())?.into() }
            signed: |$T| { sum_int::<i64>(array.chunks())?.into() }
            floating: |$T| { sum_float(array.chunks())?.into() }
        );

        Ok(Scalar::new(sum_dtype, scalar_value))
    }
}

register_kernel!(SumKernelAdapter(ChunkedEncoding).lift());

fn sum_int<T: NativePType + PrimInt + FromPrimitiveOrF16>(
    chunks: &[ArrayRef],
) -> VortexResult<Option<T>> {
    let mut result = T::zero();
    for chunk in chunks {
        let chunk_sum = sum(chunk)?;

        let Some(chunk_sum) = chunk_sum.as_primitive().as_::<T>()? else {
            // Bail out on overflow
            return Ok(None);
        };

        let Some(chunk_result) = result.checked_add(&chunk_sum) else {
            // Bail out on overflow
            return Ok(None);
        };

        result = chunk_result;
    }
    Ok(Some(result))
}

fn sum_float(chunks: &[ArrayRef]) -> VortexResult<f64> {
    let mut result = 0f64;
    for chunk in chunks {
        let chunk_sum = sum(chunk)?;
        let chunk_sum = chunk_sum
            .as_primitive()
            .as_::<f64>()?
            .vortex_expect("Float sum should never be null");
        result += chunk_sum;
    }
    Ok(result)
}
