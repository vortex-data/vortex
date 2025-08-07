// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::PrimInt;
use vortex_dtype::{NativePType, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_scalar::{FromPrimitiveOrF16, Scalar};

use crate::arrays::{ChunkedArray, ChunkedVTable};
use crate::compute::{SumKernel, SumKernelAdapter, sum};
use crate::stats::Stat;
use crate::{ArrayRef, register_kernel};

impl SumKernel for ChunkedVTable {
    fn sum(&self, array: &ChunkedArray) -> VortexResult<Scalar> {
        let sum_dtype = Stat::Sum
            .dtype(array.dtype())
            .ok_or_else(|| vortex_err!("Sum not supported for dtype {}", array.dtype()))?;
        let sum_ptype = PType::try_from(&sum_dtype).vortex_expect("sum dtype must be primitive");

        let scalar_value = match_each_native_ptype!(
            sum_ptype,
            unsigned: |T| { sum_int::<u64>(array.chunks())?.into() },
            signed: |T| { sum_int::<i64>(array.chunks())?.into() },
            floating: |T| { sum_float(array.chunks())?.into() }
        );

        Ok(Scalar::new(sum_dtype, scalar_value))
    }
}

register_kernel!(SumKernelAdapter(ChunkedVTable).lift());

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
        if let Some(chunk_sum) = sum(chunk)?.as_primitive().as_::<f64>()? {
            result += chunk_sum;
        };
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::array::IntoArray;
    use crate::arrays::{ChunkedArray, ConstantArray, PrimitiveArray};
    use crate::compute::sum;

    #[test]
    fn test_sum_chunked_floats_with_nulls() {
        // Create chunks with floats including nulls
        let chunk1 =
            PrimitiveArray::from_option_iter(vec![Some(1.5f64), None, Some(3.2), Some(4.8)]);

        let chunk2 = PrimitiveArray::from_option_iter(vec![Some(2.1f64), Some(5.7), None]);

        let chunk3 = PrimitiveArray::from_option_iter(vec![None, Some(1.0f64), Some(2.5), None]);

        // Create chunked array from the chunks
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )
        .unwrap();

        // Compute sum
        let result = sum(chunked.as_ref()).unwrap();

        // Expected sum: 1.5 + 3.2 + 4.8 + 2.1 + 5.7 + 1.0 + 2.5 = 20.8
        assert_eq!(result.as_primitive().as_::<f64>().unwrap(), Some(20.8));
    }

    #[test]
    fn test_sum_chunked_floats_all_nulls() {
        // Create chunks with all nulls
        let chunk1 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None, None]);
        let chunk2 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None]);

        let dtype = chunk1.dtype().clone();
        let chunked =
            ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype).unwrap();

        // Compute sum - should return null for all nulls
        let result = sum(chunked.as_ref()).unwrap();
        assert!(result.as_primitive().as_::<f64>().unwrap().is_none());
    }

    #[test]
    fn test_sum_chunked_floats_empty_chunks() {
        // Test with some empty chunks mixed with non-empty
        let chunk1 = PrimitiveArray::from_option_iter(vec![Some(10.5f64), Some(20.3)]);
        let chunk2 = ConstantArray::new(Scalar::primitive(0f64, Nullability::Nullable), 0);
        let chunk3 = PrimitiveArray::from_option_iter(vec![Some(5.2f64)]);

        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )
        .unwrap();

        // Compute sum: 10.5 + 20.3 + 5.2 = 36.0
        let result = sum(chunked.as_ref()).unwrap();
        assert_eq!(result.as_primitive().as_::<f64>().unwrap(), Some(36.0));
    }
}
