// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Device execution and CPU agreement for Widen arrays.

use num_traits::Bounded;
use rstest::rstest;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::WidenArray;
use vortex::array::arrays::decimal::DecimalArrayExt;
use vortex::array::assert_arrays_eq;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::encodings::fastlanes::FoR;
use vortex::error::VortexResult;

use crate::CanonicalCudaExt;
use crate::CudaDispatchMode;
use crate::CudaExecutionCtx;
use crate::CudaSession;
use crate::cuda_session;
use crate::executor::CudaArrayExt;

/// Copies fixture values to the device while preserving their dtype and validity.
async fn upload(child: ArrayRef, ctx: &mut CudaExecutionCtx) -> VortexResult<ArrayRef> {
    match child.execute::<Canonical>(ctx.execution_ctx())? {
        Canonical::Primitive(child) => {
            let parts = child.into_data_parts();
            Ok(PrimitiveArray::from_buffer_handle(
                ctx.ensure_on_device(parts.buffer).await?,
                parts.ptype,
                parts.validity,
            )
            .into_array())
        }
        Canonical::Decimal(child) => {
            let parts = child.into_data_parts();
            Ok(DecimalArray::try_new_handle(
                ctx.ensure_on_device(parts.values).await?,
                parts.values_type,
                parts.decimal_dtype,
                parts.validity,
            )?
            .into_array())
        }
        _ => unreachable!("Widen test fixtures are primitive or decimal"),
    }
}

async fn check_widen(
    host_child: ArrayRef,
    device_child: ArrayRef,
    dtype: DType,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    let host = WidenArray::try_new(host_child, dtype.clone())?.into_array();
    let expected = host.execute::<Canonical>(ctx.execution_ctx())?.into_array();
    let device = WidenArray::try_new(device_child, dtype.clone())?.into_array();
    assert!(!device.is_host());

    let result = device.execute_cuda(ctx).await?;
    assert_eq!(result.dtype(), &dtype);
    assert_eq!(result.len(), expected.len());
    match &result {
        Canonical::Primitive(array) => assert!(!array.buffer_handle().is_on_host()),
        Canonical::Decimal(array) => {
            assert!(!array.buffer_handle().is_on_host());
            assert_eq!(
                array.values_type(),
                dtype
                    .as_decimal_opt()
                    .expect("the result is decimal")
                    .canonical_values_type()
            );
        }
        _ => unreachable!("Widen returns primitive or decimal arrays"),
    }

    let actual = result.into_host(ctx.execution_ctx()).await?.into_array();
    assert_arrays_eq!(actual, expected, ctx.execution_ctx());

    Ok(())
}

fn validity(nullable: bool, len: usize) -> Validity {
    if nullable {
        Validity::from_iter((0..len).map(|i| i % 3 != 0))
    } else {
        Validity::NonNullable
    }
}

#[rstest]
#[case::i8_i16(Buffer::from_iter([i8::MIN, i8::MAX, -1, 0, 1]), PType::I16)]
#[case::i8_i32(Buffer::from_iter([i8::MIN, i8::MAX, -1, 0, 1]), PType::I32)]
#[case::i8_i64(Buffer::from_iter([i8::MIN, i8::MAX, -1, 0, 1]), PType::I64)]
#[case::i16_i32(Buffer::from_iter([i16::MIN, i16::MAX, -1, 0, 1]), PType::I32)]
#[case::i16_i64(Buffer::from_iter([i16::MIN, i16::MAX, -1, 0, 1]), PType::I64)]
#[case::i32_i64(Buffer::from_iter([i32::MIN, i32::MAX, -1, 0, 1]), PType::I64)]
#[case::u8_u16(Buffer::from_iter([u8::MIN, u8::MAX, 0, 1, 1 << 7]), PType::U16)]
#[case::u8_u32(Buffer::from_iter([u8::MIN, u8::MAX, 0, 1, 1 << 7]), PType::U32)]
#[case::u8_u64(Buffer::from_iter([u8::MIN, u8::MAX, 0, 1, 1 << 7]), PType::U64)]
#[case::u16_u32(Buffer::from_iter([u16::MIN, u16::MAX, 0, 1, 1 << 15]), PType::U32)]
#[case::u16_u64(Buffer::from_iter([u16::MIN, u16::MAX, 0, 1, 1 << 15]), PType::U64)]
#[case::u32_u64(Buffer::from_iter([u32::MIN, u32::MAX, 0, 1, 1 << 31]), PType::U64)]
#[crate::test]
async fn test_integer_pairs<T: NativePType>(
    #[case] values: Buffer<T>,
    #[case] output: PType,
    #[values(false, true)] nullable: bool,
    #[values(CudaDispatchMode::Auto, CudaDispatchMode::StandaloneOnly)] mode: CudaDispatchMode,
) -> VortexResult<()> {
    let validity = validity(nullable, values.len());
    let child = PrimitiveArray::new(values, validity).into_array();
    let dtype = DType::Primitive(output, child.dtype().nullability());
    let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?.with_dispatch_mode(mode);
    let device_child = upload(child.clone(), &mut ctx).await?;

    check_widen(child, device_child, dtype, &mut ctx).await
}

#[rstest]
#[case::i8_p4([-99i8, 99], 2, 4)]
#[case::i8_p9([-99i8, 99], 2, 9)]
#[case::i8_p18([-99i8, 99], 2, 18)]
#[case::i8_p38([-99i8, 99], 2, 38)]
#[case::i8_p76([-99i8, 99], 2, 76)]
#[case::i16_p9([-9_999i16, 9_999], 4, 9)]
#[case::i16_p18([-9_999i16, 9_999], 4, 18)]
#[case::i16_p38([-9_999i16, 9_999], 4, 38)]
#[case::i16_p76([-9_999i16, 9_999], 4, 76)]
#[case::i32_p18([-999_999_999i32, 999_999_999], 9, 18)]
#[case::i32_p38([-999_999_999i32, 999_999_999], 9, 38)]
#[case::i32_p76([-999_999_999i32, 999_999_999], 9, 76)]
#[case::i64_p38([-999_999_999_999_999_999i64, 999_999_999_999_999_999], 18, 38)]
#[case::i64_p76([-999_999_999_999_999_999i64, 999_999_999_999_999_999], 18, 76)]
#[case::i128_p76([-(1i128 << 100), (1i128 << 100)], 38, 76)]
#[crate::test]
async fn test_decimal_pairs<T: NativeDecimalType + Bounded>(
    #[case] bounds: [T; 2],
    #[case] child_precision: u8,
    #[case] precision: u8,
    #[values(false, true)] nullable: bool,
    #[values(CudaDispatchMode::Auto, CudaDispatchMode::StandaloneOnly)] mode: CudaDispatchMode,
) -> VortexResult<()> {
    let mut values = [
        T::default(),
        bounds[0],
        T::default(),
        T::default(),
        bounds[1],
    ];
    if nullable {
        values[0] = T::min_value();
        values[3] = T::max_value();
    }

    let child = DecimalArray::new(
        Buffer::from_iter(values),
        DecimalDType::new(child_precision, 1),
        validity(nullable, 5),
    )
    .into_array();
    let dtype = DType::Decimal(DecimalDType::new(precision, 1), child.dtype().nullability());
    let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?.with_dispatch_mode(mode);
    let device_child = upload(child.clone(), &mut ctx).await?;

    check_widen(child, device_child, dtype, &mut ctx).await
}

#[rstest]
#[crate::test]
async fn test_compressed_child(
    #[values(CudaDispatchMode::Auto, CudaDispatchMode::StandaloneOnly)] mode: CudaDispatchMode,
) -> VortexResult<()> {
    let child = PrimitiveArray::from_iter([0i8, 1, 50, 99]).into_array();
    let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?.with_dispatch_mode(mode);
    let device_child = upload(child.clone(), &mut ctx).await?;
    let host = FoR::try_new(child, (-100i8).into())?.into_array();
    let device = FoR::try_new(device_child, (-100i8).into())?.into_array();

    check_widen(host, device, PType::I64.into(), &mut ctx).await
}

#[rstest]
#[case::one(1)]
#[case::before_vector(15)]
#[case::vector(16)]
#[case::after_vector(17)]
#[case::before_block(2047)]
#[case::block(2048)]
#[case::after_block(2049)]
#[crate::test]
async fn test_kernel_boundaries(#[case] len: usize) -> VortexResult<()> {
    let child = PrimitiveArray::from_iter((0..len).map(|i| if i % 2 == 0 { -99i8 } else { 99 }))
        .into_array();
    let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
    let device_child = upload(child.clone(), &mut ctx).await?;

    check_widen(child, device_child, PType::I64.into(), &mut ctx).await
}

#[rstest]
#[case::equal_width(false)]
#[case::narrowing(true)]
#[crate::test]
async fn test_legacy_decimal_storage(#[case] narrowing: bool) -> VortexResult<()> {
    let dtype = DecimalDType::new(9, 1);
    let child = if narrowing {
        DecimalArray::from_iter([-99i128, 0, 99], dtype)
    } else {
        DecimalArray::from_iter([-99i64, 0, 99], dtype)
    }
    .into_array();
    let logical = DType::Decimal(DecimalDType::new(18, 1), Nullability::NonNullable);
    let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
    let device_child = upload(child.clone(), &mut ctx).await?;

    if narrowing {
        let array = WidenArray::try_new(device_child, logical)?.into_array();
        let error = array
            .execute_cuda(&mut ctx)
            .await
            .expect_err("i128 storage requires narrowing to i64");
        assert!(
            error
                .to_string()
                .contains("CUDA Widen requires decimal storage no wider than i64, got i128"),
            "{error}"
        );
    } else {
        check_widen(child, device_child, logical, &mut ctx).await?;
    }

    Ok(())
}
