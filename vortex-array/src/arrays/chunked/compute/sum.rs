// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::compute::SumKernel;
use crate::compute::SumKernelAdapter;
use crate::compute::sum_with_accumulator;
use crate::register_kernel;
use crate::scalar::Scalar;

impl SumKernel for ChunkedVTable {
    fn sum(&self, array: &ChunkedArray, accumulator: &Scalar) -> VortexResult<Scalar> {
        array
            .chunks
            .iter()
            .try_fold(accumulator.clone(), |result, chunk| {
                sum_with_accumulator(chunk, &result)
            })
    }
}

register_kernel!(SumKernelAdapter(ChunkedVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::array::IntoArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::sum;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::i256;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

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
        assert_eq!(result.as_primitive().as_::<f64>(), Some(20.8));
    }

    #[test]
    fn test_sum_chunked_floats_all_nulls_is_zero() {
        // Create chunks with all nulls
        let chunk1 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None, None]);
        let chunk2 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None]);

        let dtype = chunk1.dtype().clone();
        let chunked =
            ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype).unwrap();
        // Compute sum - should return null for all nulls
        let result = sum(chunked.as_ref()).unwrap();
        assert_eq!(result, Scalar::primitive(0f64, Nullability::Nullable));
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
        assert_eq!(result.as_primitive().as_::<f64>(), Some(36.0));
    }

    #[test]
    fn test_sum_chunked_int_almost_all_null_chunks() {
        let chunk1 = PrimitiveArray::from_option_iter::<u32, _>(vec![Some(1)]);
        let chunk2 = PrimitiveArray::from_option_iter::<u32, _>(vec![None]);

        let dtype = chunk1.dtype().clone();
        let chunked =
            ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype).unwrap();

        let result = sum(chunked.as_ref()).unwrap();
        assert_eq!(result.as_primitive().as_::<u64>(), Some(1));
    }

    #[test]
    fn test_sum_chunked_decimals() {
        // Create decimal chunks with precision=10, scale=2
        let decimal_dtype = DecimalDType::new(10, 2);
        let chunk1 = DecimalArray::new(
            buffer![100i32, 100i32, 100i32, 100i32, 100i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk2 = DecimalArray::new(
            buffer![200i32, 200i32, 200i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk3 = DecimalArray::new(buffer![300i32, 300i32], decimal_dtype, Validity::AllValid);

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

        // Compute sum: 5*100 + 3*200 + 2*300 = 500 + 600 + 600 = 1700 (represents 17.00)
        let result = sum(chunked.as_ref()).unwrap();
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(1700)))
        );
    }

    #[test]
    fn test_sum_chunked_decimals_with_nulls() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create chunks with some nulls - all must have same nullability
        let chunk1 = DecimalArray::new(
            buffer![100i32, 100i32, 100i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk2 = DecimalArray::new(
            buffer![0i32, 0i32],
            decimal_dtype,
            Validity::from_iter([false, false]),
        );
        let chunk3 = DecimalArray::new(buffer![200i32, 200i32], decimal_dtype, Validity::AllValid);

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

        // Compute sum: 3*100 + 2*200 = 300 + 400 = 700 (nulls ignored)
        let result = sum(chunked.as_ref()).unwrap();
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(700)))
        );
    }

    #[test]
    fn test_sum_chunked_decimals_large() {
        // Create decimals with precision 3 (max value 999)
        // Sum will be 500 + 600 = 1100, which fits in result precision 13 (3+10)
        let decimal_dtype = DecimalDType::new(3, 0);
        let chunk1 = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I16(500),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            1,
        );
        let chunk2 = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I16(600),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            1,
        );

        let dtype = chunk1.dtype().clone();
        let chunked =
            ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype).unwrap();

        // Compute sum: 500 + 600 = 1100
        // Result should have precision 13 (3+10), scale 0
        let result = sum(chunked.as_ref()).unwrap();
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(1100)))
        );
        assert_eq!(
            result.dtype(),
            &DType::Decimal(DecimalDType::new(13, 0), Nullability::Nullable)
        );
    }
}
