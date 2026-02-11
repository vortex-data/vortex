// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_cuda_macros::cuda_tests;

#[cuda_tests]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::DevicePtr;
    use cudarc::driver::LaunchConfig;
    use cudarc::driver::PushKernelArg;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPackedArray;
    use vortex_session::VortexSession;

    use crate::CudaBufferExt;
    use crate::CudaDeviceBuffer;
    use crate::CudaExecutionCtx;
    use crate::dynamic_dispatch_op::DynamicOp;
    use crate::dynamic_dispatch_op::DynamicOpCode_BITUNPACK;
    use crate::dynamic_dispatch_op::DynamicOpCode_FOR;
    use crate::session::CudaSession;

    fn make_bitpacked_array_u32(bit_width: u8, len: usize) -> BitPackedArray {
        let max_val = (1u64 << bit_width).saturating_sub(1);
        let values: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32)
            .collect();
        let primitive = PrimitiveArray::new(Buffer::from(values), NonNullable);
        BitPackedArray::encode(primitive.as_ref(), bit_width)
            .vortex_expect("failed to create BitPacked array")
    }

    fn run_dynamic_dispatch_u32(
        cuda_ctx: &CudaExecutionCtx,
        input_ptr: u64,
        output_len: usize,
        ops: &[DynamicOp],
    ) -> VortexResult<Vec<u32>> {
        let output_slice = cuda_ctx
            .device_alloc::<u32>(output_len.next_multiple_of(1024))
            .vortex_expect("alloc output");
        let output_buf = CudaDeviceBuffer::new(output_slice);
        let output_ptr = output_buf.as_view::<u32>().device_ptr(cuda_ctx.stream()).0;

        let device_ops = Arc::new(
            cuda_ctx
                .stream()
                .clone_htod(ops)
                .expect("copy ops to device"),
        );
        let ops_ptr = device_ops.device_ptr(cuda_ctx.stream()).0;
        let num_ops = ops.len() as u8;
        let array_len_u64 = output_len as u64;

        cuda_ctx.stream().synchronize().expect("sync");

        let cuda_function = cuda_ctx
            .load_function("dynamic_dispatch", &["u32"])
            .vortex_expect("load kernel");
        let mut launch_builder = cuda_ctx.launch_builder(&cuda_function);
        launch_builder.arg(&input_ptr);
        launch_builder.arg(&output_ptr);
        launch_builder.arg(&array_len_u64);
        launch_builder.arg(&ops_ptr);
        launch_builder.arg(&num_ops);

        let num_blocks = u32::try_from(output_len.div_ceil(2048))?;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            launch_builder.launch(config).expect("kernel launch");
        }

        let host_output: Vec<u32> = cuda_ctx
            .stream()
            .clone_dtoh(&output_buf.as_view::<u32>())
            .expect("copy back");

        Ok(host_output[..output_len].to_vec())
    }

    fn copy_to_device(
        cuda_ctx: &CudaExecutionCtx,
        bitpacked: &BitPackedArray,
    ) -> VortexResult<(u64, BufferHandle)> {
        let packed = bitpacked.packed().clone();
        let device_input = futures::executor::block_on(cuda_ctx.move_to_device(packed)?)
            .vortex_expect("move to device");
        let ptr = device_input
            .cuda_view::<u32>()
            .vortex_expect("input view")
            .device_ptr(cuda_ctx.stream())
            .0;
        Ok((ptr, device_input))
    }

    #[test]
    fn test_bitunpack() -> VortexResult<()> {
        let bit_width: u8 = 10;
        let len = 3000;

        let max_val = (1u64 << bit_width).saturating_sub(1);
        let expected: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32)
            .collect();

        let bitpacked = make_bitpacked_array_u32(bit_width, len);
        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (input_ptr, _device_input) = copy_to_device(&cuda_ctx, &bitpacked)?;

        let ops = [DynamicOp {
            op: DynamicOpCode_BITUNPACK,
            param: bit_width as u64,
        }];

        let result = run_dynamic_dispatch_u32(&cuda_ctx, input_ptr, len, &ops)?;
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_for() -> VortexResult<()> {
        let reference: u32 = 42;
        let len = 5000;

        let input: Vec<u32> = (0..len).map(|i| i as u32).collect();
        let expected: Vec<u32> = input.iter().map(|v| v + reference).collect();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let device_input = cuda_ctx
            .stream()
            .clone_htod(input.as_slice())
            .expect("copy input to device");
        let input_ptr = device_input.device_ptr(cuda_ctx.stream()).0;

        let ops = [DynamicOp {
            op: DynamicOpCode_FOR,
            param: reference as u64,
        }];

        let result = run_dynamic_dispatch_u32(&cuda_ctx, input_ptr, len, &ops)?;
        assert_eq!(result, expected);

        Ok(())
    }

    /// 1 bitunpack + 7 FoR
    #[test]
    fn test_max_ops_bitunpack_7for() -> VortexResult<()> {
        let bit_width: u8 = 6;
        let len = 2050;
        let references: [u32; 7] = [1, 2, 4, 8, 16, 32, 64];
        let total_reference: u32 = references.iter().sum();

        let max_val = (1u64 << bit_width).saturating_sub(1);
        let expected: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32 + total_reference)
            .collect();

        let bitpacked = make_bitpacked_array_u32(bit_width, len);
        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (input_ptr, _device_input) = copy_to_device(&cuda_ctx, &bitpacked)?;

        let mut ops = Vec::with_capacity(8);
        ops.push(DynamicOp {
            op: DynamicOpCode_BITUNPACK,
            param: bit_width as u64,
        });
        for &r in &references {
            ops.push(DynamicOp {
                op: DynamicOpCode_FOR,
                param: r as u64,
            });
        }
        assert_eq!(ops.len(), 8);

        let result = run_dynamic_dispatch_u32(&cuda_ctx, input_ptr, len, &ops)?;
        assert_eq!(result, expected);

        Ok(())
    }
}
