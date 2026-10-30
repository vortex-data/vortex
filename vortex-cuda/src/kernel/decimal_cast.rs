// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Widening casts of decimal values buffers on the device.
//!
//! The Arrow export widens decimal values to the width that the Arrow schema fixes, and the
//! DecimalByteParts decoder widens a narrow MSP to the canonical width of its decimal dtype. Both
//! launch the `decimal_cast` kernels through this module.
//!
//! These casts only widen. A narrowing cast can overflow, and to detect that on the device, the
//! kernel would need an overflow flag that synchronizes with the host. So this module rejects a
//! narrowing cast, and each caller decides how to handle values that are wider than its target.

use std::sync::Arc;

use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_decimal_value_type;
use vortex::buffer::Buffer;
use vortex::dtype::DecimalType;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::i256;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_panic;
use vortex_cuda_driver::DeviceRepr;
use vortex_cuda_driver::PushKernelArg;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;

/// Widen `len` decimal values from `values_type` to the wider `target_type` on the device.
///
/// The values can be on the host or on the device, and the result is on the device. Each value is
/// sign-extended, so the result holds the same values as the input.
///
/// # Errors
///
/// Returns an error if `target_type` is not wider than `values_type`, or if the copy to the device
/// or the kernel launch fails.
pub(crate) async fn widen_decimal_values(
    values: BufferHandle,
    values_type: DecimalType,
    target_type: DecimalType,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle> {
    vortex_ensure!(
        values_type < target_type,
        "a decimal cast on the GPU must widen to a type wider than {values_type}, got \
         {target_type}"
    );
    let values = ctx.ensure_on_device(values).await?;

    match_each_decimal_value_type!(values_type, |S| {
        decimal_cast_to::<S>(values, target_type, len, ctx).await
    })
}

/// Dispatch [`decimal_cast`] from `S` on the runtime `target_type`.
async fn decimal_cast_to<S>(
    values: BufferHandle,
    target_type: DecimalType,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle>
where
    S: NativeDecimalType + DeviceRepr,
{
    match target_type {
        DecimalType::I8 => vortex_panic!(
            "widen_decimal_values ensures a target wider than the values type, and no decimal \
             type is narrower than i8, got {target_type}"
        ),
        DecimalType::I16 => decimal_cast::<S, i16>(values, len, ctx).await,
        DecimalType::I32 => decimal_cast::<S, i32>(values, len, ctx).await,
        DecimalType::I64 => decimal_cast::<S, i64>(values, len, ctx).await,
        DecimalType::I128 => decimal_cast::<S, i128>(values, len, ctx).await,
        DecimalType::I256 => decimal_cast::<S, i256>(values, len, ctx).await,
    }
}

/// Launch the `decimal_cast_<S>_<D>` kernel, which casts device values of type `S` to type `D`.
///
/// The cast must widen. A narrowing cast truncates each value with no overflow check.
async fn decimal_cast<S, D>(
    values: BufferHandle,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle>
where
    S: NativeDecimalType + DeviceRepr,
    D: NativeDecimalType + DeviceRepr,
{
    if len == 0 {
        return ctx
            .ensure_on_device(BufferHandle::new_host(
                Buffer::<D>::empty().into_byte_buffer(),
            ))
            .await;
    }

    let mut output_buffer = ctx.device_alloc::<D>(len)?;

    let values_view = values.cuda_view::<S>()?;
    let len_u64 = len as u64;
    let cuda_function = ctx.load_function_with_suffixes(
        "decimal_cast",
        &[&S::DECIMAL_TYPE.to_string(), &D::DECIMAL_TYPE.to_string()],
    )?;

    ctx.launch_kernel(&cuda_function, len, |args| {
        args.arg(&values_view).arg(&mut output_buffer).arg(&len_u64);
    })?;

    Ok(BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(
        output_buffer,
    ))))
}
