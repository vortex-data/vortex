// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decode into caller-owned storage without constructing an owning CUDA slice or stream.

use std::fmt;
use std::process;
use std::ptr;

use vortex::array::ArrayRef;
use vortex::array::arrays::Primitive;
use vortex::array::buffer::BufferHandle;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_ensure_eq;
use vortex::error::vortex_err;
use vortex_cuda_driver::CudaContext;
use vortex_cuda_driver::CudaEvent;
use vortex_cuda_driver::CudaStream;
use vortex_cuda_driver::DevicePtr;
use vortex_cuda_driver::DeviceSlice;
use vortex_cuda_driver::DriverError;
use vortex_cuda_driver::HostSlice;
use vortex_cuda_driver::sys;
use vortex_cuda_driver::sys::CUdeviceptr;
use vortex_cuda_driver::sys::CUstream;

use super::materialize_decode_into;
use crate::CudaBufferExt;
use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;

/// Whether a failed decode handed the destination back to its caller.
#[derive(Debug)]
pub enum ForeignDecodeError {
    /// No write was submitted, or all submitted writes precede subsequent caller-stream work.
    Ordered(VortexError),
    /// Neither the return fence nor recovery synchronization succeeded. Quarantine the allocation.
    OrderingLost(VortexError),
}

impl fmt::Display for ForeignDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ordered(error) => error.fmt(f),
            Self::OrderingLost(error) => {
                write!(
                    f,
                    "could not confirm completion of foreign CUDA write: {error}"
                )
            }
        }
    }
}

impl std::error::Error for ForeignDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Ordered(error) | Self::OrderingLost(error) => Some(error),
        }
    }
}

impl From<VortexError> for ForeignDecodeError {
    fn from(error: VortexError) -> Self {
        Self::Ordered(error)
    }
}

/// Decode one non-nullable primitive array into the first `len * ptype.byte_width()` bytes at
/// `dst`. Pass the address of the first output row, not necessarily the allocation base, and
/// `capacity_bytes` remaining **from that address**. The tail is untouched. An empty primitive
/// array is a no-op and accepts a null pointer. No pointer or stream ownership is transferred.
///
/// The destination must be device memory on the execution device: either owned by its CUDA
/// context or a device-local pool allocation without an owning context.
/// Canonical primitives copy directly from their host or device buffer. For encoded arrays,
/// a fully fused plan writes in place; otherwise the array is decoded and its values copied.
/// Prior caller-stream work (including allocation) precedes the write. After `Ok` or
/// [`ForeignDecodeError::Ordered`], consumption, reuse or stream-ordered free can be enqueued on
/// that stream without a host wait. This is not a device completion notification. On error the
/// values are unspecified. [`ForeignDecodeError::OrderingLost`] means that neither the handoff
/// nor synchronization succeeded. An FFI that cannot retain the allocation must abort rather
/// than return that error to a caller that may free it.
///
/// If submission unwinds, the same stream handoff (or a successful synchronization) precedes
/// propagation of the panic. If both fail during unwinding, this function aborts.
///
/// # Safety
///
/// - `dst` must designate an exclusively writable device allocation of the advertised capacity,
///   naturally aligned for the array's primitive type, with no overlap with input buffers. CUDA
///   cannot validate the size of an RMM suballocation. Keep it live through the ordered write.
/// - The caller stream must be live through the call, in `ctx`'s context, and ordered after all
///   earlier conflicting accesses. No concurrent conflicting submissions or graph capture on
///   either stream. Subsequent work on other streams needs a dependency on the caller stream.
/// - Default stream handles are interpreted in `ctx`'s context (null means legacy default).
///   For a per-thread default stream, poll the future on the originating host thread.
/// - Cancellation before submission does not access `dst`; submission and handoff never yield.
pub async unsafe fn decode_into_foreign(
    array: &ArrayRef,
    dst: CUdeviceptr,
    capacity_bytes: usize,
    caller_stream: CUstream,
    ctx: &mut CudaExecutionCtx,
) -> Result<(), ForeignDecodeError> {
    let (ptype, bytes) = destination_layout(array, dst, capacity_bytes)?;
    if bytes == 0 {
        return Ok(());
    }
    let stream = ctx.stream().clone();
    check_pointer_context(dst, stream.context())?;

    // Materialization can suspend. Do not expose the destination to any CUDA work until the
    // last await, so dropping the future never skips the handoff of an in-flight write.
    let decoded;
    let values = if let Some(values) = array.as_opt::<Primitive>() {
        values
    } else if let Some(plan) = materialize_decode_into(array, ctx).await? {
        // SAFETY: The caller lends the stream and disjoint destination. Layout validation and
        // the plan type check bound every typed store. The scope orders writes on both streams.
        return unsafe {
            ordered_write(caller_stream, &stream, || {
                vortex_ensure_eq!(plan.dispatch_plan.output_ptype(), ptype);
                plan.launch_into(dst, array.len(), ctx)
            })
        };
    } else {
        decoded = array.clone().execute_cuda(ctx).await?.into_primitive();
        decoded.as_view()
    };
    // SAFETY: As above; copy_into also checks the source byte count before submission.
    unsafe {
        ordered_write(caller_stream, &stream, || {
            vortex_ensure_eq!(values.ptype(), ptype);
            copy_into(values.buffer_handle(), dst, bytes, &stream)
        })
    }
}

/// `caller` must be a live stream. `write` may enqueue destination accesses only on `stream`.
unsafe fn ordered_write(
    caller: CUstream,
    stream: &CudaStream,
    write: impl FnOnce() -> VortexResult<()>,
) -> Result<(), ForeignDecodeError> {
    // cuEventRecord rejects a stream and event from different contexts (unlike
    // cuStreamWaitEvent, which allows cross-context waits). Re-record the same event for the
    // return trip: waits already queued observe only the recording they were submitted with.
    if caller == stream.cu_stream() {
        return write().map_err(ForeignDecodeError::Ordered);
    }
    let event = stream
        .context()
        .new_event(None)
        .map_err(VortexError::external)?;
    // SAFETY: The caller lends a live stream; event creation bound the execution context.
    unsafe { sys::cuEventRecord(event.cu_event(), caller) }
        .result()
        .map_err(VortexError::external)?;
    stream.wait(&event).map_err(VortexError::external)?;

    // Arm before calling user-configurable launch code: it can panic after queuing a write.
    let mut pending = ReturnFence {
        caller,
        stream,
        event,
        armed: true,
    };
    let result = write();
    let result = match pending.enqueue() {
        Ok(()) => result.map_err(ForeignDecodeError::Ordered),
        Err(fence) => match stream.synchronize() {
            Ok(()) => Err(ForeignDecodeError::Ordered(
                result.err().unwrap_or_else(|| VortexError::external(fence)),
            )),
            Err(sync) => Err(ForeignDecodeError::OrderingLost(
                VortexError::external(sync).with_context(format!(
                    "could not synchronize after caller-stream handoff failed: {fence}"
                )),
            )),
        },
    };
    pending.armed = false;
    result
}

struct ReturnFence<'a> {
    caller: CUstream,
    stream: &'a CudaStream,
    event: CudaEvent,
    armed: bool,
}

impl ReturnFence<'_> {
    fn enqueue(&self) -> Result<(), DriverError> {
        self.event.record(self.stream)?;
        // SAFETY: ordered_write borrows the caller stream for this scope. The entry event
        // verified its context, and record just bound that context on this thread.
        unsafe { sys::cuStreamWaitEvent(self.caller, self.event.cu_event(), 0) }.result()
    }
}

impl Drop for ReturnFence<'_> {
    fn drop(&mut self) {
        if self.armed && self.enqueue().is_err() && self.stream.synchronize().is_err() {
            // Unwind cannot deliver OrderingLost. Returning control to an FFI panic handler
            // would let the caller reclaim memory that the execution stream may still write.
            process::abort();
        }
    }
}

fn destination_layout(
    array: &ArrayRef,
    dst: CUdeviceptr,
    capacity: usize,
) -> VortexResult<(PType, usize)> {
    let DType::Primitive(ptype, Nullability::NonNullable) = array.dtype() else {
        vortex_bail!("foreign CUDA destination requires a non-nullable primitive array");
    };
    let width = ptype.byte_width();
    let bytes = array
        .len()
        .checked_mul(width)
        .ok_or_else(|| vortex_err!("destination byte count overflow"))?;
    vortex_ensure!(
        capacity >= bytes,
        "destination needs {bytes} bytes, has {capacity}"
    );
    if bytes != 0 {
        vortex_ensure!(dst != 0, "null CUDA destination for nonempty array");
        vortex_ensure!(
            dst.is_multiple_of(width as u64),
            "unaligned CUDA destination for {ptype}"
        );
        vortex_ensure!(
            dst.checked_add(bytes as u64).is_some(),
            "CUDA destination range overflows"
        );
    }
    Ok((*ptype, bytes))
}

fn check_pointer_context(dst: CUdeviceptr, ctx: &CudaContext) -> VortexResult<()> {
    // CUDA ABI attributes not named in the minimal driver wrapper.
    const CONTEXT: sys::CUpointer_attribute = sys::CUpointer_attribute(1);
    const DEVICE_ORDINAL: sys::CUpointer_attribute = sys::CUpointer_attribute(9);
    const MEMPOOL_HANDLE: sys::CUpointer_attribute = sys::CUpointer_attribute(17);
    const DEVICE_MEMORY: u32 = 2;
    let mut owner: sys::CUcontext = ptr::null_mut();
    let mut memory_type = 0u32;
    let mut ordinal = -1i32;
    let mut pool: sys::CUmemoryPool = ptr::null_mut();
    let mut attrs = [
        CONTEXT,
        sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        DEVICE_ORDINAL,
        MEMPOOL_HANDLE,
    ];
    let mut outputs = [
        (&raw mut owner).cast(),
        (&raw mut memory_type).cast(),
        (&raw mut ordinal).cast(),
        (&raw mut pool).cast(),
    ];
    ctx.bind_to_thread().map_err(VortexError::external)?;
    // SAFETY: Each attribute has a corresponding writable output of its CUDA ABI type.
    unsafe { sys::cuPointerGetAttributes(4, attrs.as_mut_ptr(), outputs.as_mut_ptr(), dst) }
        .result()
        .map_err(VortexError::external)?;
    vortex_ensure_eq!(memory_type, DEVICE_MEMORY);
    vortex_ensure_eq!(ordinal, ctx.cu_device());
    // Stream-ordered pool allocations are device-local but need not have an owning CUcontext.
    vortex_ensure!(
        owner == ctx.cu_ctx() || (owner.is_null() && !pool.is_null()),
        "CUDA destination belongs to a different or unknown context"
    );
    Ok(())
}

/// `dst` must hold `bytes` writable bytes, ordered for a non-overlapping copy on `stream`.
unsafe fn copy_into(
    values: &BufferHandle,
    dst: CUdeviceptr,
    bytes: usize,
    stream: &CudaStream,
) -> VortexResult<()> {
    vortex_ensure_eq!(values.len(), bytes);
    let view;
    let (status, guard) = if let Some(host) = values.as_host_opt() {
        // SAFETY: Finish the driver's host borrow below, including on copy failure. This waits
        // for pinned sources; pageable sources are staged before the copy returns.
        let (src, guard) = unsafe { host.as_slice().stream_synced_slice(stream) }
            .map_err(VortexError::external)?;
        // SAFETY: Both ranges cover `bytes` and are live and ordered on this stream.
        let status = unsafe {
            sys::cuMemcpyHtoDAsync_v2(dst, src.as_ptr().cast(), bytes, stream.cu_stream())
        };
        (status, guard)
    } else {
        view = values.cuda_view::<u8>()?;
        vortex_ensure!(
            view.stream().context().cu_ctx() == stream.context().cu_ctx(),
            "CUDA copy source is in another context"
        );
        let (src, guard) = view.device_ptr(stream).map_err(VortexError::external)?;
        // SAFETY: The source read is tracked through submission and both ranges cover `bytes`.
        let status = unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.cu_stream()) };
        (status, guard)
    };
    let finished = guard.finish();
    status.result().map_err(VortexError::external)?;
    finished.map_err(VortexError::external)
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;
    use std::panic::catch_unwind;
    use std::sync::Arc;

    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::validity::Validity;
    use vortex::encodings::sequence::Sequence;
    use vortex_cuda_driver::sys::CUevent_flags;

    use super::*;
    use crate::CudaDispatchMode;
    use crate::CudaKernelEvents;
    use crate::CudaSession;
    use crate::LaunchStrategy;
    use crate::cuda_session;

    #[test]
    fn test_destination_layout() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter([1u64, 2]).into_array();
        assert_eq!(destination_layout(&values, 8, 24)?, (PType::U64, 16));
        assert!(destination_layout(&values, 8, 15).is_err());
        assert!(destination_layout(&values, 4, 16).is_err());
        assert!(destination_layout(&values, 0, 16).is_err());
        assert!(destination_layout(&values, u64::MAX - 7, 16).is_err());
        let nullable = PrimitiveArray::from_option_iter([Some(1u64), None]).into_array();
        assert!(destination_layout(&nullable, 8, 16).is_err());
        let empty = PrimitiveArray::empty::<u64>(Nullability::NonNullable).into_array();
        assert_eq!(destination_layout(&empty, 0, 0)?, (PType::U64, 0));
        Ok(())
    }

    #[derive(Debug)]
    struct RejectKernelLaunch;

    impl LaunchStrategy for RejectKernelLaunch {
        fn event_flags(&self) -> Option<CUevent_flags> {
            Some(CUevent_flags::CU_EVENT_DISABLE_TIMING)
        }

        fn on_complete(&self, _: &CudaKernelEvents, _: usize) -> VortexResult<()> {
            vortex_bail!("canonical primitive copy launched a kernel")
        }
    }

    #[crate::test]
    async fn test_foreign_adjacent_writes() -> VortexResult<()> {
        let owner = CudaSession::create_execution_ctx(&cuda_session())?;
        let caller = owner.stream();
        let dst = caller
            .clone_htod(&[-1i32; 8])
            .map_err(VortexError::external)?;
        let encoded =
            Sequence::try_new_typed(5i32, -12i32, Nullability::NonNullable, 2)?.into_array();
        let values = PrimitiveArray::from_iter([5i32, -7]).into_array();
        let device_values = PrimitiveArray::from_buffer_handle(
            caller.copy_to_device(&[6i32, -8])?,
            PType::I32,
            Validity::NonNullable,
        )
        .into_array();
        let mut fused = owner
            .fork()?
            .with_dispatch_mode(CudaDispatchMode::DynDispatchOnly);
        let mut copies = fused
            .fork()?
            .with_launch_strategy(Arc::new(RejectKernelLaunch));
        for (row, array) in [(1, &encoded), (3, &values), (5, &device_values)] {
            let ctx = if array.is_canonical() {
                &mut copies
            } else {
                &mut fused
            };
            // SAFETY: The interior subrange and caller stream are live through readback. All
            // accesses to dst outside the bridge use its allocation stream.
            unsafe {
                decode_into_foreign(
                    array,
                    dst.device_address() + row * 4,
                    dst.num_bytes() - row as usize * 4,
                    caller.cu_stream(),
                    ctx,
                )
                .await
            }
            .map_err(VortexError::external)?;
        }
        assert_eq!(
            caller.clone_dtoh(&dst).map_err(VortexError::external)?,
            [-1, 5, -7, 5, -7, 6, -8, -1]
        );
        Ok(())
    }

    #[crate::test]
    fn test_foreign_stream_from_another_context_is_rejected_before_write() -> VortexResult<()> {
        let ctx = CudaSession::create_execution_ctx(&cuda_session())?;
        let other = CudaContext::new_non_primary(ctx.stream().context().ordinal(), 0)
            .map_err(VortexError::external)?;
        let caller = other.new_stream().map_err(VortexError::external)?;
        let mut submitted = false;
        // SAFETY: Both streams are live. The scope must reject the mismatched context.
        let result = unsafe {
            ordered_write(caller.cu_stream(), ctx.stream(), || {
                submitted = true;
                Ok(())
            })
        };
        assert!(matches!(result, Err(ForeignDecodeError::Ordered(_))));
        assert!(!submitted);
        Ok(())
    }

    #[rstest]
    #[case::error(false)]
    #[case::unwind(true)]
    #[crate::test]
    #[expect(clippy::panic, reason = "exercise unwinding after CUDA submission")]
    fn test_failure_after_submission_returns_destination_in_stream_order(
        #[case] unwind: bool,
    ) -> VortexResult<()> {
        let ctx = CudaSession::create_execution_ctx(&cuda_session())?;
        let caller = ctx
            .stream()
            .context()
            .new_stream()
            .map_err(VortexError::external)?;
        let dst = caller.clone_htod(&[0u8]).map_err(VortexError::external)?;
        // SAFETY: Submit exactly one byte on the ordered stream, then simulate a later error.
        let result = catch_unwind(AssertUnwindSafe(|| unsafe {
            ordered_write(caller.cu_stream(), ctx.stream(), || {
                sys::cuMemsetD8Async(dst.device_address(), 42, 1, ctx.stream().cu_stream())
                    .result()
                    .map_err(VortexError::external)?;
                if unwind {
                    panic!("post-submission panic");
                }
                vortex_bail!("post-submission failure")
            })
        }));
        if unwind {
            let payload = result.expect_err("submission panic should propagate");
            assert_eq!(
                payload.downcast_ref::<&str>(),
                Some(&"post-submission panic")
            );
        } else {
            assert!(matches!(result, Ok(Err(ForeignDecodeError::Ordered(_)))));
        }
        assert_eq!(
            caller.clone_dtoh(&dst).map_err(VortexError::external)?,
            [42]
        );
        Ok(())
    }
}
