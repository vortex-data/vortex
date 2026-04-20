// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Decimal32Array as ArrowDecimal32Array;
use arrow_array::Decimal64Array as ArrowDecimal64Array;
use arrow_array::Decimal128Array as ArrowDecimal128Array;
use arrow_array::Decimal256Array as ArrowDecimal256Array;
use arrow_buffer::i256;
use arrow_schema::DataType;
use itertools::Itertools;
use num_traits::AsPrimitive;
use num_traits::ToPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::DecimalArray;
use crate::arrow::null_buffer::to_null_buffer;
use crate::dtype::DecimalType;

pub(super) fn to_arrow_decimal(
    array: ArrayRef,
    data_type: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // Execute the array as a DecimalArray.
    let decimal_array = array.execute::<DecimalArray>(ctx)?;

    match data_type {
        DataType::Decimal32(..) => to_arrow_decimal32(decimal_array, ctx),
        DataType::Decimal64(..) => to_arrow_decimal64(decimal_array, ctx),
        DataType::Decimal128(..) => to_arrow_decimal128(decimal_array, ctx),
        DataType::Decimal256(..) => to_arrow_decimal256(decimal_array, ctx),
        _ => unreachable!("to_arrow_decimal called with non-decimal type"),
    }
}

fn to_arrow_decimal32(array: DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrowArrayRef> {
    let null_buffer = to_null_buffer(
        array
            .as_ref()
            .validity()?
            .to_mask(array.as_ref().len(), ctx)?,
    );
    let buffer: Buffer<i32> = match array.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(array.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(array.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => array.buffer::<i32>(),
        DecimalType::I64 => array
            .buffer::<i64>()
            .into_iter()
            .map(|x| {
                x.to_i32()
                    .ok_or_else(|| vortex_err!("i64 to i32 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
        DecimalType::I128 => array
            .buffer::<i128>()
            .into_iter()
            .map(|x| {
                x.to_i32()
                    .ok_or_else(|| vortex_err!("i128 to i32 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
        DecimalType::I256 => array
            .buffer::<crate::dtype::i256>()
            .into_iter()
            .map(|x| {
                x.to_i32()
                    .ok_or_else(|| vortex_err!("i256 to i32 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
    };
    Ok(Arc::new(
        ArrowDecimal32Array::new(buffer.into_arrow_scalar_buffer(), null_buffer)
            .with_precision_and_scale(
                array.decimal_dtype().precision(),
                array.decimal_dtype().scale(),
            )?,
    ))
}

fn to_arrow_decimal64(array: DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrowArrayRef> {
    let null_buffer = to_null_buffer(
        array
            .as_ref()
            .validity()?
            .to_mask(array.as_ref().len(), ctx)?,
    );
    let buffer: Buffer<i64> = match array.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(array.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(array.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => {
            Buffer::from_trusted_len_iter(array.buffer::<i32>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I64 => array.buffer::<i64>(),
        DecimalType::I128 => array
            .buffer::<i128>()
            .into_iter()
            .map(|x| {
                x.to_i64()
                    .ok_or_else(|| vortex_err!("i128 to i64 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
        DecimalType::I256 => array
            .buffer::<crate::dtype::i256>()
            .into_iter()
            .map(|x| {
                x.to_i64()
                    .ok_or_else(|| vortex_err!("i256 to i64 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
    };
    Ok(Arc::new(
        ArrowDecimal64Array::new(buffer.into_arrow_scalar_buffer(), null_buffer)
            .with_precision_and_scale(
                array.decimal_dtype().precision(),
                array.decimal_dtype().scale(),
            )?,
    ))
}

fn to_arrow_decimal128(array: DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrowArrayRef> {
    let null_buffer = to_null_buffer(
        array
            .as_ref()
            .validity()?
            .to_mask(array.as_ref().len(), ctx)?,
    );
    let buffer: Buffer<i128> = match array.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(array.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(array.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => {
            Buffer::from_trusted_len_iter(array.buffer::<i32>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I64 => {
            Buffer::from_trusted_len_iter(array.buffer::<i64>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I128 => array.buffer::<i128>(),
        DecimalType::I256 => array
            .buffer::<crate::dtype::i256>()
            .into_iter()
            .map(|x| {
                x.to_i128()
                    .ok_or_else(|| vortex_err!("i256 to i128 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
    };
    Ok(Arc::new(
        ArrowDecimal128Array::new(buffer.into_arrow_scalar_buffer(), null_buffer)
            .with_precision_and_scale(
                array.decimal_dtype().precision(),
                array.decimal_dtype().scale(),
            )?,
    ))
}

fn to_arrow_decimal256(array: DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrowArrayRef> {
    let null_buffer = to_null_buffer(
        array
            .as_ref()
            .validity()?
            .to_mask(array.as_ref().len(), ctx)?,
    );
    let buffer: Buffer<i256> = match array.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(array.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(array.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => {
            Buffer::from_trusted_len_iter(array.buffer::<i32>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I64 => {
            Buffer::from_trusted_len_iter(array.buffer::<i64>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I128 => Buffer::from_trusted_len_iter(
            array
                .buffer::<i128>()
                .into_iter()
                .map(|x| crate::dtype::i256::from_i128(x).into()),
        ),
        DecimalType::I256 => {
            Buffer::<i256>::from_byte_buffer(array.buffer_handle().clone().into_host_sync())
        }
    };
    Ok(Arc::new(
        ArrowDecimal256Array::new(buffer.into_arrow_scalar_buffer(), null_buffer)
            .with_precision_and_scale(
                array.decimal_dtype().precision(),
                array.decimal_dtype().scale(),
            )?,
    ))
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::Decimal128Array;
    use arrow_array::Decimal256Array;
    use arrow_buffer::i256;
    use arrow_schema::DataType;
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::array::IntoArray;
    use crate::arrow::ArrowArrayExecutor;
    use crate::arrow::IntoArrowArray;
    use crate::arrow::executor::decimal::DecimalArray;
    use crate::builders::ArrayBuilder;
    use crate::builders::DecimalBuilder;
    use crate::dtype::DecimalDType;
    use crate::dtype::NativeDecimalType;
    use crate::validity::Validity;

    #[test]
    fn decimal_to_arrow() -> VortexResult<()> {
        // Make a very simple i128 and i256 array.
        let decimal_vortex = DecimalArray::new(
            buffer![1i128, 2i128, 3i128, 4i128, 5i128],
            DecimalDType::new(19, 2),
            Validity::NonNullable,
        );
        let arrow = decimal_vortex.into_array().execute_arrow(
            Some(&DataType::Decimal128(19, 2)),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert_eq!(arrow.data_type(), &DataType::Decimal128(19, 2));
        let decimal_array = arrow.as_any().downcast_ref::<Decimal128Array>().unwrap();
        assert_eq!(
            decimal_array.values().as_ref(),
            &[1i128, 2i128, 3i128, 4i128, 5i128]
        );
        Ok(())
    }

    #[rstest]
    #[case(0i8)]
    #[case(0i16)]
    #[case(0i32)]
    #[case(0i64)]
    #[case(0i128)]
    #[case(crate::dtype::i256::ZERO)]
    fn test_to_arrow_decimal128<T: NativeDecimalType>(
        #[case] _decimal_type: T,
    ) -> VortexResult<()> {
        let mut decimal = DecimalBuilder::new::<T>(DecimalDType::new(2, 1), false.into());
        decimal.append_value(10);
        decimal.append_value(11);
        decimal.append_value(12);
        let decimal = decimal.finish();

        let arrow_array = decimal.into_arrow(&DataType::Decimal128(2, 1))?;
        let arrow_decimal = arrow_array
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap();
        assert_eq!(arrow_decimal.value(0), 10);
        assert_eq!(arrow_decimal.value(1), 11);
        assert_eq!(arrow_decimal.value(2), 12);
        Ok(())
    }

    #[rstest]
    #[case(0i8)]
    #[case(0i16)]
    #[case(0i32)]
    #[case(0i64)]
    #[case(0i128)]
    #[case(crate::dtype::i256::ZERO)]
    fn test_to_arrow_decimal32<T: NativeDecimalType>(#[case] _decimal_type: T) -> VortexResult<()> {
        use arrow_array::Decimal32Array;

        let mut decimal = DecimalBuilder::new::<T>(DecimalDType::new(2, 1), false.into());
        decimal.append_value(10);
        decimal.append_value(11);
        decimal.append_value(12);
        let decimal = decimal.finish();

        let arrow_array = decimal.into_arrow(&DataType::Decimal32(2, 1))?;
        let arrow_decimal = arrow_array
            .as_any()
            .downcast_ref::<Decimal32Array>()
            .unwrap();
        assert_eq!(arrow_decimal.value(0), 10);
        assert_eq!(arrow_decimal.value(1), 11);
        assert_eq!(arrow_decimal.value(2), 12);
        Ok(())
    }

    #[rstest]
    #[case(0i8)]
    #[case(0i16)]
    #[case(0i32)]
    #[case(0i64)]
    #[case(0i128)]
    #[case(crate::dtype::i256::ZERO)]
    fn test_to_arrow_decimal64<T: NativeDecimalType>(#[case] _decimal_type: T) -> VortexResult<()> {
        use arrow_array::Decimal64Array;

        let mut decimal = DecimalBuilder::new::<T>(DecimalDType::new(2, 1), false.into());
        decimal.append_value(10);
        decimal.append_value(11);
        decimal.append_value(12);
        let decimal = decimal.finish();

        let arrow_array = decimal.into_arrow(&DataType::Decimal64(2, 1))?;
        let arrow_decimal = arrow_array
            .as_any()
            .downcast_ref::<Decimal64Array>()
            .unwrap();
        assert_eq!(arrow_decimal.value(0), 10);
        assert_eq!(arrow_decimal.value(1), 11);
        assert_eq!(arrow_decimal.value(2), 12);
        Ok(())
    }

    #[rstest]
    #[case(0i8)]
    #[case(0i16)]
    #[case(0i32)]
    #[case(0i64)]
    #[case(0i128)]
    #[case(crate::dtype::i256::ZERO)]
    fn test_to_arrow_decimal256<T: NativeDecimalType>(
        #[case] _decimal_type: T,
    ) -> VortexResult<()> {
        let mut decimal = DecimalBuilder::new::<T>(DecimalDType::new(2, 1), false.into());
        decimal.append_value(10);
        decimal.append_value(11);
        decimal.append_value(12);
        let decimal = decimal.finish();

        let arrow_array = decimal.into_arrow(&DataType::Decimal256(2, 1))?;
        let arrow_decimal = arrow_array
            .as_any()
            .downcast_ref::<Decimal256Array>()
            .unwrap();
        assert_eq!(arrow_decimal.value(0), i256::from_i128(10));
        assert_eq!(arrow_decimal.value(1), i256::from_i128(11));
        assert_eq!(arrow_decimal.value(2), i256::from_i128(12));
        Ok(())
    }
}
