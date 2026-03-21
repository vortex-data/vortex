// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU comparison kernel: compares each element of a device buffer against a
//! scalar value and produces a packed bitmask on the device.
//!
//! The bitmask is LSB-first within each byte (same layout as Arrow/Vortex
//! validity bitmaps).  The output stays on the device as a `BufferHandle`
//! so it can be fed directly to CUB filter without a host round-trip.
//! The comparison operator is encoded as a u8 matching the `CompareOp` enum
//! in `compare.cu`.

use std::sync::Arc;

use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_native_ptype;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::scalar::Scalar;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaExecutionCtx;

/// Comparison operators, matching the `CompareOp` enum in `compare.cu`.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Eq = 0,
    NotEq = 1,
    Gt = 2,
    Gte = 3,
    Lt = 4,
    Lte = 5,
}

/// Compare each element of a device-resident primitive buffer against a scalar
/// value, producing a device-resident packed bitmask.
///
/// The input buffer must already be on the device.  The output bitmask stays
/// on the device as a `BufferHandle` so it can be passed directly to CUB
/// filter without any host round-trip.
///
/// The returned buffer contains `ceil(len / 32) * 4` bytes of packed bits
/// in LSB-first order.  `len` (the number of elements/bits) is also returned.
pub fn compare_on_device(
    input: &BufferHandle,
    ptype: PType,
    scalar: &Scalar,
    op: CompareOp,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle> {
    if len == 0 {
        return Ok(BufferHandle::new_host(vortex::buffer::ByteBuffer::empty()));
    }

    match_each_native_ptype!(ptype, |T| {
        compare_on_device_typed::<T>(input, scalar, op, len, ctx)
    })
}

fn compare_on_device_typed<T>(
    input: &BufferHandle,
    scalar: &Scalar,
    op: CompareOp,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle>
where
    T: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let scalar_value: T = scalar
        .as_primitive()
        .as_::<T>()
        .vortex_expect("scalar must not be null for comparison");

    let input_view = input.cuda_view::<T>()?;

    // The kernel writes one uint32_t (4 bytes) per 32 elements.
    let output_bytes = ((len + 31) / 32) * 4;
    let output_slice = ctx.device_alloc::<u8>(output_bytes)?;
    let output_buf = CudaDeviceBuffer::new(output_slice);
    let output_view = output_buf.as_view::<u8>();

    let cuda_function = ctx.load_function("compare", &[T::PTYPE])?;
    let array_len_u64 = len as u64;
    let op_u8 = op as u8;

    ctx.launch_kernel(&cuda_function, len, |args| {
        args.arg(&input_view)
            .arg(&output_view)
            .arg(&array_len_u64)
            .arg(&scalar_value)
            .arg(&op_u8);
    })?;

    Ok(BufferHandle::new_device(Arc::new(output_buf)))
}

#[cfg(test)]
mod tests {
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::primitive::PrimitiveArrayParts;
    use vortex::array::buffer::DeviceBuffer;
    use vortex::array::validity::Validity::NonNullable;
    use vortex::buffer::Alignment;
    use vortex::buffer::BitBuffer;
    use vortex::buffer::Buffer;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::mask::Mask;
    use vortex::scalar::Scalar;
    use vortex::session::VortexSession;

    use super::*;
    use crate::session::CudaSession;

    /// Copy the device bitmask to host and construct a Mask for assertion.
    async fn to_host_mask(bitmask: &BufferHandle, len: usize) -> VortexResult<Mask> {
        let host_buf = bitmask.as_device().copy_to_host(Alignment::new(1))?.await?;
        Ok(Mask::from_buffer(BitBuffer::new(host_buf, len)))
    }

    #[crate::test]
    async fn test_compare_gt() -> VortexResult<()> {
        let mut ctx =
            CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

        let data: Vec<i32> = (0..2048).map(|i| i as i32).collect();
        let prim = PrimitiveArray::new(Buffer::from(data.clone()), NonNullable);
        let handle = ctx.ensure_on_device(prim.into_parts().buffer).await?;

        let bitmask = compare_on_device(
            &handle,
            PType::I32,
            &Scalar::from(1000i32),
            CompareOp::Gt,
            2048,
            &mut ctx,
        )?;

        assert!(bitmask.is_on_device());
        let mask = to_host_mask(&bitmask, 2048).await?;
        assert_eq!(mask.true_count(), 1047); // 1001..=2047
        assert!(!mask.value(1000)); // 1000 is not > 1000
        assert!(mask.value(1001)); // 1001 is > 1000
        assert!(mask.value(2047));
        assert!(!mask.value(0));
        Ok(())
    }

    #[crate::test]
    async fn test_compare_eq() -> VortexResult<()> {
        let mut ctx =
            CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

        let data: Vec<u32> = (0..1024).map(|i| i % 4).collect();
        let prim = PrimitiveArray::new(Buffer::from(data), NonNullable);
        let handle = ctx.ensure_on_device(prim.into_parts().buffer).await?;

        let bitmask = compare_on_device(
            &handle,
            PType::U32,
            &Scalar::from(2u32),
            CompareOp::Eq,
            1024,
            &mut ctx,
        )?;

        assert!(bitmask.is_on_device());
        let mask = to_host_mask(&bitmask, 1024).await?;
        assert_eq!(mask.true_count(), 256); // every 4th element
        assert!(!mask.value(0));
        assert!(!mask.value(1));
        assert!(mask.value(2));
        assert!(!mask.value(3));
        assert!(mask.value(6));
        Ok(())
    }
}
