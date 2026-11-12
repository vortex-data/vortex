// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Restores the logical width of integer and decimal [`Widen`] arrays on the device.
//!
//! Children execute at their own width before widening, so compressed signed values retain their
//! sign. Execution keeps the child's validity and allocates a wider values buffer, unless decimal
//! storage already has the output width.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::Widen;
use vortex::array::arrays::widen::WidenArraySlotsExt;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_integer_ptype;
use vortex::dtype::DType;
use vortex::dtype::NativePType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex_cuda_driver::DeviceRepr;
use vortex_cuda_driver::PushKernelArg;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::kernel::widen_decimal_values;

/// CUDA execution for integer and decimal arrays stored at a narrower width.
#[derive(Debug)]
pub(crate) struct WidenExecutor;

#[async_trait]
impl CudaExecute for WidenExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array = array.try_downcast::<Widen>().map_err(|array| {
            vortex_err!(
                "WidenExecutor requires a Widen array, got {}",
                array.encoding_id()
            )
        })?;
        let child = array.values().clone().execute_cuda(ctx).await?;

        match (array.dtype(), child) {
            (DType::Primitive(output_ptype, _), Canonical::Primitive(child)) => {
                let len = child.len();
                let parts = child.into_data_parts();
                let values = ctx.ensure_on_device(parts.buffer).await?;
                let output = match_each_integer_ptype!(parts.ptype, |S| {
                    match_each_integer_ptype!(*output_ptype, |D| {
                        widen_integers::<S, D>(values, len, ctx)?
                    })
                });

                Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
                    output,
                    *output_ptype,
                    parts.validity,
                )))
            }
            (DType::Decimal(decimal_dtype, _), Canonical::Decimal(child)) => {
                let len = child.len();
                let parts = child.into_data_parts();
                let target_type = decimal_dtype.canonical_values_type();
                vortex_ensure!(
                    parts.values_type <= target_type,
                    "CUDA Widen requires decimal storage no wider than {target_type}, got {}",
                    parts.values_type
                );

                let values = if parts.values_type == target_type {
                    ctx.ensure_on_device(parts.values).await?
                } else {
                    widen_decimal_values(parts.values, parts.values_type, target_type, len, ctx)
                        .await?
                };

                Ok(Canonical::Decimal(DecimalArray::try_new_handle(
                    values,
                    target_type,
                    *decimal_dtype,
                    parts.validity,
                )?))
            }
            (dtype, child) => vortex_bail!(
                "CUDA Widen requires an integer or decimal child matching {dtype}, got {}",
                child.dtype()
            ),
        }
    }
}

/// Widens `len` device values of type `S` into a new buffer of type `D`.
///
/// `D` must be strictly wider than `S` with the same signedness, as the [`Widen`] invariant
/// guarantees. The `widen.cu` kernels only instantiate those pairs. The other monomorphized pairs
/// cannot run because a valid `Widen` never dispatches to them.
fn widen_integers<S, D>(
    values: BufferHandle,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle>
where
    S: NativePType + DeviceRepr,
    D: NativePType + DeviceRepr,
{
    let input = values.cuda_view::<S>()?;
    let mut output = ctx.device_alloc::<D>(len)?;
    let len_u64 = len as u64;
    let function = ctx.load_function("widen", &[S::PTYPE, D::PTYPE])?;
    ctx.launch_kernel(&function, len, |args| {
        args.arg(&input).arg(&mut output).arg(&len_u64);
    })?;

    Ok(BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(
        output,
    )?)))
}

#[cfg(test)]
mod tests;
