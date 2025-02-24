use vortex_dtype::PType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{sum, SumFn};
use crate::stats::Stat;
use crate::{Array, ArrayRef};

impl SumFn<&ChunkedArray> for ChunkedEncoding {
    fn sum(&self, array: &ChunkedArray) -> VortexResult<Scalar> {
        let sum_dtype = Stat::Sum.dtype(array.dtype());
        let scalar_value = match PType::try_from(&sum_dtype)
            .vortex_expect("sum dtype must be primitive")
        {
            PType::U8 | PType::U16 | PType::U32 | PType::U64 => {
                sum_unsigned(array.chunks())?.into()
            }
            PType::I8 | PType::I16 | PType::I32 | PType::I64 => sum_signed(array.chunks())?.into(),
            PType::F16 | PType::F32 | PType::F64 => sum_float(array.chunks())?.into(),
        };
        Ok(Scalar::new(sum_dtype, scalar_value))
    }
}

fn sum_unsigned(chunks: &[ArrayRef]) -> VortexResult<Option<u64>> {
    let mut result = 0u64;
    for chunk in chunks {
        let chunk_sum = sum(chunk)?;

        let Some(chunk_sum) = chunk_sum.as_primitive().as_::<u64>()? else {
            // Bail out on overflow
            return Ok(None);
        };

        let Some(chunk_result) = result.checked_add(chunk_sum) else {
            // Bail out on overflow
            return Ok(None);
        };

        result = chunk_result;
    }
    Ok(Some(result))
}

fn sum_signed(chunks: &[ArrayRef]) -> VortexResult<Option<i64>> {
    let mut result = 0i64;
    for chunk in chunks {
        let chunk_sum = sum(chunk)?;

        let Some(chunk_sum) = chunk_sum.as_primitive().as_::<i64>()? else {
            // Bail out on overflow
            return Ok(None);
        };

        let Some(chunk_result) = result.checked_add(chunk_sum) else {
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
