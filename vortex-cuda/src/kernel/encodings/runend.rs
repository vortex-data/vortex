// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_native_ptype;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::array::validity::Validity;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::runend::RunEndArray;
use vortex::encodings::runend::RunEndArrayExt;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::scalar::Scalar;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA executor for run-end encoded arrays.
#[derive(Debug)]
pub(crate) struct RunEndExecutor;

impl RunEndExecutor {
    fn try_specialize(array: ArrayRef) -> Option<RunEndArray> {
        array.try_downcast::<RunEnd>().ok()
    }
}

#[async_trait]
impl CudaExecute for RunEndExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array =
            Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected RunEndArray"))?;

        if !array.dtype().is_primitive() {
            vortex_bail!("RunEndExecutor only supports primitive types")
        }

        let offset = array.offset();
        let output_len = array.len();
        let ends = array.ends().clone();
        let values = array.values().clone();

        let values_ptype = PType::try_from(values.dtype())?;
        let ends_ptype = PType::try_from(ends.dtype())?;

        if output_len == 0 {
            let nullability = values.dtype().nullability();
            return Ok(Canonical::Primitive(match_each_native_ptype!(
                values_ptype,
                |V| { PrimitiveArray::empty::<V>(nullability) }
            )));
        }

        if matches!(values.validity()?, Validity::AllInvalid) {
            return ConstantArray::new(Scalar::null(values.dtype().clone()), output_len)
                .into_array()
                .execute::<Canonical>(ctx.execution_ctx());
        }

        let ends = ends.execute_cuda(ctx).await?.into_primitive();
        let values = values.execute_cuda(ctx).await?.into_primitive();

        match_each_native_ptype!(values_ptype, |V| {
            match_each_unsigned_integer_ptype!(ends_ptype, |E| {
                decode_runend_typed::<V, E>(ends, values, offset, output_len, ctx).await
            })
        })
    }
}

async fn decode_runend_typed<V: DeviceRepr + NativePType, E: DeviceRepr + NativePType>(
    ends: PrimitiveArray,
    values: PrimitiveArray,
    offset: usize,
    output_len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let num_runs = ends.len();
    vortex_ensure!(num_runs > 0, "run-end array must have at least one run");
    vortex_ensure!(
        output_len > 0,
        "run-end output length must be greater than zero"
    );

    let PrimitiveDataParts {
        ptype: value_ptype,
        buffer: values_buffer,
        validity: values_validity,
        ..
    } = values.into_data_parts();

    let PrimitiveDataParts {
        buffer: ends_buffer,
        ..
    } = ends.into_data_parts();

    // Set up device buffers.
    let ends_device = ctx.ensure_on_device(ends_buffer).await?;
    let values_device = ctx.ensure_on_device(values_buffer).await?;

    let output_slice = ctx.device_alloc::<V>(output_len)?;
    let output_device = CudaDeviceBuffer::new(output_slice);

    let ends_view = ends_device.cuda_view::<E>()?;
    let values_view = values_device.cuda_view::<V>()?;
    let output_view = output_device.as_view::<V>();

    // Load kernel function
    let cuda_function = ctx.load_function("runend", &[value_ptype, E::PTYPE])?;

    ctx.launch_kernel(&cuda_function, output_len, |args| {
        args.arg(&ends_view)
            .arg(&num_runs)
            .arg(&values_view)
            .arg(&offset)
            .arg(&output_len)
            .arg(&output_view);
    })?;

    let output_validity = match values_validity {
        Validity::NonNullable => Validity::NonNullable,
        Validity::AllValid => Validity::AllValid,
        Validity::AllInvalid => {
            unreachable!("AllInvalid should be handled by RunEndExecutor::execute")
        }
        Validity::Array(_) => {
            vortex_bail!(
                "RunEnd GPU decoding does not yet support per-element validity in values; falling back to CPU"
            );
        }
    };

    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        BufferHandle::new_device(Arc::new(output_device)),
        value_ptype,
        output_validity,
    )))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::encodings::runend::RunEnd;
    use vortex::encodings::runend::RunEndArray;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::executor::CudaArrayExt;
    use crate::session::CudaSession;

    fn make_runend_array<V, E>(
        ends: Vec<E>,
        values: Vec<V>,
        ctx: &mut vortex::array::ExecutionCtx,
    ) -> RunEndArray
    where
        V: NativePType,
        E: NativePType,
    {
        let ends_array =
            PrimitiveArray::new(Buffer::from(ends), Validity::NonNullable).into_array();
        let values_array =
            PrimitiveArray::new(Buffer::from(values), Validity::NonNullable).into_array();
        RunEnd::new(ends_array, values_array, ctx)
    }

    type RunEndBuilder = fn(&mut vortex::array::ExecutionCtx) -> RunEndArray;

    #[rstest]
    #[case::u32_ends_u8_values(|ctx: &mut vortex::array::ExecutionCtx| make_runend_array(vec![3u32, 6, 10], vec![10u8, 20, 30], ctx))]
    #[case::u32_ends_u32_values(|ctx: &mut vortex::array::ExecutionCtx| make_runend_array(vec![2u32, 5, 10], vec![1u32, 2, 3], ctx))]
    #[case::u32_ends_f64_values(|ctx: &mut vortex::array::ExecutionCtx| make_runend_array(vec![2u32, 5, 8], vec![1.5f64, 2.5, 3.5], ctx))]
    #[case::u8_ends_i32_values(|ctx: &mut vortex::array::ExecutionCtx| make_runend_array(vec![2u8, 5, 10], vec![1i32, 2, 3], ctx))]
    #[case::u32_ends_i32_values(|ctx: &mut vortex::array::ExecutionCtx| make_runend_array(vec![2u32, 5, 10], vec![1i32, 2, 3], ctx))]
    #[case::u64_ends_i32_values(|ctx: &mut vortex::array::ExecutionCtx| make_runend_array(vec![2u64, 5, 10], vec![1i32, 2, 3], ctx))]
    #[crate::test]
    async fn test_cuda_runend_types(#[case] build: RunEndBuilder) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let runend_array = build(cuda_ctx.execution_ctx());
        let cpu_result = crate::canonicalize_cpu(runend_array.clone())?;

        let gpu_result = RunEndExecutor
            .execute(runend_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[crate::test]
    async fn test_cuda_runend_large_array() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let num_runs = 41;
        let run_length = 50;
        let total_len = num_runs * run_length;

        let ends: Vec<u64> = (1..=num_runs).map(|i| (i * run_length) as u64).collect();
        let values: Vec<i32> = (0..num_runs).map(|i| i32::try_from(i).unwrap()).collect();

        let runend_array = make_runend_array(ends, values, cuda_ctx.execution_ctx());
        assert_eq!(runend_array.len(), total_len);

        let cpu_result = crate::canonicalize_cpu(runend_array.clone())?;

        let gpu_result = RunEndExecutor
            .execute(runend_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[crate::test]
    async fn test_cuda_runend_single_run() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let runend_array = make_runend_array(vec![100u32], vec![42i32], cuda_ctx.execution_ctx());

        let cpu_result = crate::canonicalize_cpu(runend_array.clone())?;

        let gpu_result = RunEndExecutor
            .execute(runend_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[crate::test]
    async fn test_cuda_runend_many_small_runs() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create an array where each run has length 1.
        let num_elements = 2050;
        let ends: Vec<u32> = (1..=num_elements).collect();
        let values: Vec<i32> = (0..num_elements as i32).collect();

        let runend_array = make_runend_array(ends, values, cuda_ctx.execution_ctx());

        let cpu_result = crate::canonicalize_cpu(runend_array.clone())?;

        let gpu_result = RunEndExecutor
            .execute(runend_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[crate::test]
    async fn test_cuda_runend_nullable_values_falls_back_to_cpu() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Build a RunEnd array whose values have Validity::Array (some nulls).
        let ends_array =
            PrimitiveArray::new(Buffer::from(vec![3u32, 6, 10]), Validity::NonNullable)
                .into_array();
        let validity =
            Validity::Array(BoolArray::from_iter([true, false, true].into_iter()).into_array());
        let values_array =
            PrimitiveArray::new(Buffer::from(vec![10i32, 0, 30]), validity).into_array();
        let runend_array = RunEnd::new(ends_array, values_array, cuda_ctx.execution_ctx());

        let cpu_result = crate::canonicalize_cpu(runend_array.clone())?.into_array();

        // execute_cuda should fall back to CPU and still produce the correct result.
        let gpu_result = runend_array
            .into_array()
            .execute_cuda(&mut cuda_ctx)
            .await
            .vortex_expect("GPU/CPU fallback should succeed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }
}
