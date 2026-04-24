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
use vortex::array::arrays::Constant;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_decimal_value_type;
use vortex::array::match_each_native_simd_ptype;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::DecimalType;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::NativePType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::CudaDeviceBuffer;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA executor for constant arrays with numeric types.
///
/// Materializes a constant array by filling a device buffer with the scalar value.
/// Supports primitive types (integers, floats) and decimal types (i128, i256).
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

        // Check if scalar is null
        if array.scalar().is_null() {
            vortex_bail!("CUDA constant array does not support null scalar values");
        }

        match array.scalar().dtype() {
            DType::Primitive(ptype, nullability) => {
                let validity: Validity = nullability.into();
                match_each_native_simd_ptype!(*ptype, |P| {
                    materialize_constant_primitive::<P>(array, validity, ctx).await
                })
            }
            DType::Decimal(decimal_dtype, nullability) => {
                let decimal_dtype = *decimal_dtype;
                let validity: Validity = nullability.into();
                let values_type = DecimalType::smallest_decimal_value_type(&decimal_dtype);
                match_each_decimal_value_type!(values_type, |D| {
                    materialize_constant_decimal::<D>(array, decimal_dtype, validity, ctx).await
                })
            }
            dt => vortex_bail!(
                "CUDA constant array only supports numeric types, got {:?}",
                dt
            ),
        }
    }
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
        .ok_or_else(|| vortex_err!("Expected non-null primitive scalar value"))?;

    // Allocate output buffer on device
    let output_buffer = ctx.device_alloc::<P>(array_len)?;
    let output_view = output_buffer.as_view();
    let array_len_u64 = array_len as u64;

    // Load kernel function
    let kernel_ptypes = [P::PTYPE];
    let cuda_function = ctx.load_function("constant_numeric", &kernel_ptypes)?;

    ctx.launch_kernel(&cuda_function, array_len, |args| {
        args.arg(&output_view);
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
    let decimal_value = decimal_scalar
        .decimal_value()
        .ok_or_else(|| vortex_err!("Expected non-null decimal scalar value"))?;

    // Cast the decimal value to the native type
    let value: D = decimal_value
        .cast::<D>()
        .ok_or_else(|| vortex_err!("Failed to cast decimal value to native type"))?;

    // Allocate output buffer on device
    let output_buffer = ctx.device_alloc::<D>(array_len)?;
    let output_view = output_buffer.as_view();
    let array_len_u64 = array_len as u64;

    // Load kernel function
    let cuda_function =
        ctx.load_function_with_suffixes("constant_numeric", &[&D::DECIMAL_TYPE.to_string()])?;

    ctx.launch_kernel(&cuda_function, array_len, |args| {
        args.arg(&output_view);
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
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::scalar::Scalar;
    use vortex::session::VortexSession;

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
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let cpu_result = crate::canonicalize_cpu(constant_array.clone())?;

        let gpu_result = ConstantNumericExecutor
            .execute(constant_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU materialization failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[crate::test]
    async fn test_cuda_constant_empty_array() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let constant_array = ConstantArray::new(42i32, 0);
        let cpu_result = crate::canonicalize_cpu(constant_array.clone())?;

        let gpu_result = ConstantNumericExecutor
            .execute(constant_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU materialization failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[crate::test]
    async fn test_cuda_constant_small_array() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Test with array smaller than one block (< 2048 elements)
        let constant_array = ConstantArray::new(99i32, 100);
        let cpu_result = crate::canonicalize_cpu(constant_array.clone())?;

        let gpu_result = ConstantNumericExecutor
            .execute(constant_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU materialization failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }
}
