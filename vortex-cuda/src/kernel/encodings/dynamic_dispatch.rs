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
    use vortex_alp::ALPFloat;
    use vortex_alp::Exponents;
    use vortex_alp::alp_encode;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPackedArray;
    use vortex_fastlanes::FoRArray;
    use vortex_session::VortexSession;

    use crate::CudaBufferExt;
    use crate::CudaDeviceBuffer;
    use crate::CudaExecutionCtx;
    use crate::dynamic_dispatch_op::DynamicOp;
    use crate::dynamic_dispatch_op::DynamicOpCode_ALP;
    use crate::dynamic_dispatch_op::DynamicOpCode_BITUNPACK;
    use crate::dynamic_dispatch_op::DynamicOpCode_FOR;
    use crate::session::CudaSession;

    fn pack_alp_f32_param(f: f32, e: f32) -> u64 {
        (e.to_bits() as u64) << 32 | f.to_bits() as u64
    }

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

    fn run_dynamic_dispatch_f32(
        cuda_ctx: &CudaExecutionCtx,
        input_ptr: u64,
        output_len: usize,
        ops: &[DynamicOp],
    ) -> VortexResult<Vec<f32>> {
        let result = run_dynamic_dispatch_u32(cuda_ctx, input_ptr, output_len, ops)?;
        // SAFETY: f32 and u32 have identical size and alignment.
        Ok(unsafe { std::mem::transmute::<Vec<u32>, Vec<f32>>(result) })
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
        let len = 5000;

        let original: Vec<u32> = (0..len).map(|i| i as u32 + 42).collect();
        let primitive = PrimitiveArray::new(Buffer::from(original.clone()), NonNullable);

        let for_array = FoRArray::encode(primitive)?;
        let reference = u32::try_from(for_array.reference_scalar())?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let encoded_prim = for_array.encoded().to_primitive();
        let device_input = cuda_ctx
            .stream()
            .clone_htod(encoded_prim.as_slice::<u32>())
            .expect("copy input to device");
        let input_ptr = device_input.device_ptr(cuda_ctx.stream()).0;

        let ops = [DynamicOp {
            op: DynamicOpCode_FOR,
            param: reference as u64,
        }];

        // Kernel should reconstruct the original data.
        let result = run_dynamic_dispatch_u32(&cuda_ctx, input_ptr, len, &ops)?;
        assert_eq!(result, original);

        Ok(())
    }

    #[test]
    fn test_alp() -> VortexResult<()> {
        let len = 2050;

        // Start from f32 data that ALP-encodes cleanly - no patches.
        let exponents = Exponents { e: 2, f: 0 };
        let floats: Vec<f32> = (0..len)
            .map(|i| <f32 as ALPFloat>::decode_single(i as i32, exponents))
            .collect();
        let float_prim = PrimitiveArray::new(Buffer::from(floats.clone()), NonNullable);

        let alp_array = alp_encode(&float_prim, Some(exponents))?;
        assert!(alp_array.patches().is_none());

        let f = <f32 as ALPFloat>::F10[alp_array.exponents().f as usize];
        let e = <f32 as ALPFloat>::IF10[alp_array.exponents().e as usize];

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let encoded_prim = alp_array.encoded().to_primitive();
        let device_input = cuda_ctx
            .stream()
            .clone_htod(encoded_prim.as_slice::<i32>())
            .expect("copy input to device");
        let input_ptr = device_input.device_ptr(cuda_ctx.stream()).0;

        let ops = [DynamicOp {
            op: DynamicOpCode_ALP,
            param: pack_alp_f32_param(f, e),
        }];

        let result = run_dynamic_dispatch_f32(&cuda_ctx, input_ptr, len, &ops)?;
        assert_eq!(result, floats);

        Ok(())
    }

    #[test]
    fn test_alp_for_bitunpack() -> VortexResult<()> {
        let len = 2050;

        let exponents = Exponents { e: 2, f: 0 };
        let floats: Vec<f32> = (0..len)
            .map(|i| <f32 as ALPFloat>::decode_single(10 + (i as i32 % 64), exponents))
            .collect();
        let float_prim = PrimitiveArray::new(Buffer::from(floats.clone()), NonNullable);

        // ALP encode f32 → i32 encoded integers + exponents.
        let alp_array = alp_encode(&float_prim, Some(exponents))?;
        assert!(alp_array.patches().is_none());

        // FOR encode the ALP-encoded i32 integers.
        let for_array = FoRArray::encode(alp_array.encoded().to_primitive())?;
        let reference = i32::try_from(for_array.reference_scalar())? as u32;

        // BitPack the FOR-encoded values.
        let bit_width: u8 = 6;
        let bitpacked = BitPackedArray::encode(for_array.encoded(), bit_width)?;

        // Derive ALP decode factors from the actual exponents.
        let alp_f = <f32 as ALPFloat>::F10[alp_array.exponents().f as usize];
        let alp_e = <f32 as ALPFloat>::IF10[alp_array.exponents().e as usize];

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (input_ptr, _device_input) = copy_to_device(&cuda_ctx, &bitpacked)?;

        let ops = [
            DynamicOp {
                op: DynamicOpCode_BITUNPACK,
                param: bit_width as u64,
            },
            DynamicOp {
                op: DynamicOpCode_FOR,
                param: reference as u64,
            },
            DynamicOp {
                op: DynamicOpCode_ALP,
                param: pack_alp_f32_param(alp_f, alp_e),
            },
        ];

        let result = run_dynamic_dispatch_f32(&cuda_ctx, input_ptr, len, &ops)?;
        assert_eq!(result, floats);

        Ok(())
    }

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
