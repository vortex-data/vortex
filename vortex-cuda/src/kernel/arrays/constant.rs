// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::Constant;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::NullArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::varbinview::BinaryView;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_decimal_value_type;
use vortex::array::match_each_native_simd_ptype;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::DecimalType;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA executor for constant arrays with flat types.
///
/// Materializes a constant array by filling a device buffer with the scalar value. Supports null,
/// boolean, primitive, decimal, UTF-8, binary, and extensions backed by those flat types.
#[derive(Debug)]
pub(crate) struct ConstantNumericExecutor;

impl ConstantNumericExecutor {
    fn try_specialize(array: ArrayRef) -> Option<ConstantArray> {
        array.try_downcast::<Constant>().ok()
    }
}

#[async_trait]
impl CudaExecute for ConstantNumericExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array =
            Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected ConstantArray"))?;

        let validity = match (array.dtype().is_nullable(), array.scalar().is_null()) {
            (false, _) => Validity::NonNullable,
            (true, false) => Validity::AllValid,
            (true, true) => Validity::AllInvalid,
        };

        match array.scalar().dtype() {
            DType::Null => Ok(Canonical::Null(NullArray::new(array.len()))),
            DType::Bool(_) => materialize_constant_bool(array, validity, ctx).await,
            DType::Primitive(ptype, _) => {
                match_each_native_simd_ptype!(*ptype, |P| {
                    materialize_constant_primitive::<P>(array, validity, ctx).await
                })
            }
            DType::Decimal(decimal_dtype, _) => {
                let decimal_dtype = *decimal_dtype;
                let values_type = DecimalType::smallest_decimal_value_type(&decimal_dtype);
                match_each_decimal_value_type!(values_type, |D| {
                    materialize_constant_decimal::<D>(array, decimal_dtype, validity, ctx).await
                })
            }
            DType::Utf8(_) => {
                let bytes = array
                    .scalar()
                    .as_utf8()
                    .value()
                    .map(|value| value.as_bytes().to_vec());
                materialize_constant_varbinview(array, bytes, validity, ctx).await
            }
            DType::Binary(_) => {
                let bytes = array
                    .scalar()
                    .as_binary()
                    .value()
                    .map(|value| value.as_slice().to_vec());
                materialize_constant_varbinview(array, bytes, validity, ctx).await
            }
            DType::Extension(ext_dtype) => {
                let storage_scalar = array.scalar().as_extension().to_storage_scalar();
                let storage = ConstantArray::new(storage_scalar, array.len())
                    .into_array()
                    .execute_cuda(ctx)
                    .await?
                    .into_array();
                Ok(Canonical::Extension(ExtensionArray::new(
                    ext_dtype.clone(),
                    storage,
                )))
            }
            dt => vortex_bail!("CUDA constant array only supports flat types, got {:?}", dt),
        }
    }
}

async fn materialize_constant_bool(
    array: ConstantArray,
    validity: Validity,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let len = array.len();
    if len == 0 {
        return Ok(Canonical::empty(array.dtype()));
    }

    let byte_len = len.div_ceil(8);
    let value = if array.scalar().as_bool().value().unwrap_or_default() {
        u8::MAX
    } else {
        0
    };
    let mut output = ctx.device_alloc::<u8>(byte_len)?;
    let byte_len_u64 = byte_len as u64;
    let cuda_function = ctx.load_function("constant_numeric", &[PType::U8])?;

    ctx.launch_kernel(&cuda_function, byte_len, |args| {
        args.arg(&mut output).arg(&value).arg(&byte_len_u64);
    })?;

    Ok(Canonical::Bool(BoolArray::new_handle(
        BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(output))),
        0,
        len,
        validity,
    )))
}

async fn materialize_constant_varbinview(
    array: ConstantArray,
    bytes: Option<Vec<u8>>,
    validity: Validity,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let len = array.len();
    if len == 0 {
        return Ok(Canonical::empty(array.dtype()));
    }

    let view = bytes
        .as_deref()
        .map(|bytes| BinaryView::make_view(bytes, 0, 0).as_u128() as i128)
        .unwrap_or_default();
    let buffers: Arc<[BufferHandle]> =
        if let Some(bytes) = bytes.filter(|bytes| bytes.len() > BinaryView::MAX_INLINED_SIZE) {
            Arc::from([ctx.stream().copy_to_device_sync(bytes.as_slice())?])
        } else {
            Arc::from([])
        };

    let mut views = ctx.device_alloc::<i128>(len)?;
    let len_u64 = len as u64;
    let cuda_function = ctx.load_function_with_suffixes("constant_numeric", &["i128"])?;
    ctx.launch_kernel(&cuda_function, len, |args| {
        args.arg(&mut views).arg(&view).arg(&len_u64);
    })?;

    Ok(Canonical::VarBinView(unsafe {
        VarBinViewArray::new_handle_unchecked(
            BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(views))),
            buffers,
            array.dtype().clone(),
            validity,
        )
    }))
}

async fn materialize_constant_primitive<P>(
    array: ConstantArray,
    validity: Validity,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical>
where
    P: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let array_len = array.len();
    if array_len == 0 {
        return Ok(Canonical::Primitive(PrimitiveArray::empty::<P>(
            validity.nullability(),
        )));
    }

    // Extract the scalar value
    let value: P = array
        .scalar()
        .as_primitive()
        .typed_value::<P>()
        .unwrap_or_default();

    // Allocate output buffer on device
    let mut output_buffer = ctx.device_alloc::<P>(array_len)?;
    let array_len_u64 = array_len as u64;

    // Load kernel function
    let kernel_ptypes = [P::PTYPE];
    let cuda_function = ctx.load_function("constant_numeric", &kernel_ptypes)?;

    ctx.launch_kernel(&cuda_function, array_len, |args| {
        args.arg(&mut output_buffer);
        args.arg(&value);
        args.arg(&array_len_u64);
    })?;

    // Wrap the CudaSlice in a CudaDeviceBuffer and then BufferHandle
    let device_buffer = CudaDeviceBuffer::new(output_buffer);
    let buffer_handle = BufferHandle::new_device(Arc::new(device_buffer));

    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        buffer_handle,
        P::PTYPE,
        validity,
    )))
}

async fn materialize_constant_decimal<D>(
    array: ConstantArray,
    decimal_dtype: DecimalDType,
    validity: Validity,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical>
where
    D: NativeDecimalType + DeviceRepr + Send + Sync + 'static,
{
    use vortex::buffer::Buffer;

    let array_len = array.len();
    if array_len == 0 {
        return Ok(Canonical::Decimal(DecimalArray::new(
            Buffer::<D>::empty(),
            decimal_dtype,
            validity,
        )));
    }

    // Extract the decimal scalar value
    let decimal_scalar = array.scalar().as_decimal();
    let value: D = decimal_scalar
        .decimal_value()
        .map(|value| {
            value
                .cast::<D>()
                .ok_or_else(|| vortex_err!("Failed to cast decimal value to native type"))
        })
        .transpose()?
        .unwrap_or_default();

    // Allocate output buffer on device
    let mut output_buffer = ctx.device_alloc::<D>(array_len)?;
    let array_len_u64 = array_len as u64;

    // Load kernel function
    let cuda_function =
        ctx.load_function_with_suffixes("constant_numeric", &[&D::DECIMAL_TYPE.to_string()])?;

    ctx.launch_kernel(&cuda_function, array_len, |args| {
        args.arg(&mut output_buffer);
        args.arg(&value);
        args.arg(&array_len_u64);
    })?;

    // Wrap the CudaSlice in a CudaDeviceBuffer and then BufferHandle
    let device_buffer = CudaDeviceBuffer::new(output_buffer);
    let buffer_handle = BufferHandle::new_device(Arc::new(device_buffer));

    Ok(Canonical::Decimal(DecimalArray::new_handle(
        buffer_handle,
        D::DECIMAL_TYPE,
        decimal_dtype,
        validity,
    )))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::ConstantArray;
    use vortex::array::assert_arrays_eq;
    use vortex::dtype::NativePType;
    use vortex::dtype::Nullability;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::extension::datetime::Date;
    use vortex::extension::datetime::TimeUnit;
    use vortex::scalar::Scalar;
    use vortex_array::VortexSessionExecute;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    fn make_constant_array<T: NativePType + Into<Scalar>>(value: T, len: usize) -> ConstantArray {
        ConstantArray::new(value, len)
    }

    #[rstest]
    #[case::u8(make_constant_array(42u8, 2050))]
    #[case::u16(make_constant_array(1234u16, 2050))]
    #[case::u32(make_constant_array(100000u32, 2050))]
    #[case::u64(make_constant_array(1000000u64, 2050))]
    #[case::i8(make_constant_array(-42i8, 2050))]
    #[case::i16(make_constant_array(-1234i16, 2050))]
    #[case::i32(make_constant_array(-100000i32, 2050))]
    #[case::i64(make_constant_array(-1000000i64, 2050))]
    #[case::f32(make_constant_array(1.23f32, 2050))]
    #[case::f64(make_constant_array(4.56789f64, 2050))]
    #[crate::test]
    async fn test_cuda_constant_materialization(
        #[case] constant_array: ConstantArray,
    ) -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let gpu_result = ConstantNumericExecutor
            .execute(constant_array.clone().into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU materialization failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(constant_array, gpu_result, &mut ctx);

        Ok(())
    }

    #[rstest]
    #[case::bool_true(ConstantArray::new(true, 2050))]
    #[case::bool_false(ConstantArray::new(false, 2050))]
    #[case::bool_nullable(ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 2050))]
    #[case::bool_null(ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 2050))]
    #[case::utf8_inline(ConstantArray::new(Scalar::utf8("US", Nullability::Nullable), 2050))]
    #[case::utf8_empty(ConstantArray::new(Scalar::utf8("", Nullability::NonNullable), 2050))]
    #[case::utf8_outlined(ConstantArray::new(
        Scalar::utf8("thirteen bytes", Nullability::NonNullable),
        2050
    ))]
    #[case::utf8_null(ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), 2050))]
    #[case::binary_inline(ConstantArray::new(Scalar::binary(vec![0, 1, 2, 255], Nullability::Nullable), 2050))]
    #[case::binary_outlined(ConstantArray::new(Scalar::binary(vec![7; 13], Nullability::NonNullable), 2050))]
    #[case::binary_null(ConstantArray::new(
        Scalar::null(DType::Binary(Nullability::Nullable)),
        2050
    ))]
    #[case::primitive_null(ConstantArray::new(
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        2050
    ))]
    #[case::decimal_null(ConstantArray::new(
        Scalar::null(DType::Decimal(DecimalDType::new(10, 2), Nullability::Nullable)),
        2050
    ))]
    #[case::null_dtype(ConstantArray::new(Scalar::null(DType::Null), 2050))]
    #[crate::test]
    async fn test_cuda_flat_constant_materialization(
        #[case] constant_array: ConstantArray,
    ) -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let gpu_result = ConstantNumericExecutor
            .execute(constant_array.clone().into_array(), &mut cuda_ctx)
            .await?;
        if !matches!(gpu_result, Canonical::Null(_)) {
            assert!(
                !gpu_result.clone().into_array().is_host(),
                "flat constant output stayed on the host"
            );
        }
        let gpu_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(constant_array, gpu_result, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::date(ConstantArray::new(
        Scalar::extension::<Date>(TimeUnit::Days, Scalar::from(42i32)),
        2050,
    ))]
    #[case::date_null(ConstantArray::new(
        Scalar::extension::<Date>(
            TimeUnit::Days,
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        ),
        2050,
    ))]
    #[crate::test]
    async fn test_cuda_flat_extension_constant(
        #[case] constant_array: ConstantArray,
    ) -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let gpu_result = ConstantNumericExecutor
            .execute(constant_array.clone().into_array(), &mut cuda_ctx)
            .await?;
        assert!(
            !gpu_result.clone().into_array().is_host(),
            "extension storage stayed on the host"
        );
        let gpu_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(constant_array, gpu_result, &mut ctx);
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_constant_empty_array() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let constant_array = ConstantArray::new(42i32, 0);
        let gpu_result = ConstantNumericExecutor
            .execute(constant_array.clone().into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU materialization failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(constant_array, gpu_result, &mut ctx);

        Ok(())
    }

    #[crate::test]
    async fn test_cuda_constant_small_array() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        // Test with array smaller than one block (< 2048 elements)
        let constant_array = ConstantArray::new(99i32, 100);
        let gpu_result = ConstantNumericExecutor
            .execute(constant_array.clone().into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU materialization failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(constant_array, gpu_result, &mut ctx);

        Ok(())
    }
}
