// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use crate::CudaContext;
use crate::CudaStream;
use crate::DevicePtr;
use crate::DevicePtrMut;
use crate::DriverError;
use crate::HostSlice;
use crate::SyncOnDrop;
use crate::sys;

/// A context and two streams, or `None` without a CUDA device.
fn streams() -> Option<(Arc<CudaContext>, Arc<CudaStream>, Arc<CudaStream>)> {
    // SAFETY: Only probes for the driver library.
    if !unsafe { sys::is_culib_present() } {
        return None;
    }
    let context = CudaContext::new(0).ok()?;
    let first = context.new_stream().ok()?;
    let second = context.new_stream().ok()?;
    Some((context, first, second))
}

#[test]
fn host_device_round_trip() -> Result<(), DriverError> {
    let Some((context, stream, _)) = streams() else {
        return Ok(());
    };
    let values: Vec<u32> = (0..100_000).collect();
    let device = stream.clone_htod(&values)?;
    assert_eq!(stream.clone_dtoh(&device)?, values);
    assert_eq!(stream.clone_dtoh(&device.slice(10..20))?, &values[10..20]);
    context.check_err()
}

/// A write on one stream is visible to reads on another, through views as well.
#[test]
fn cross_stream_reads_follow_writes() -> Result<(), DriverError> {
    let Some((context, writer, reader)) = streams() else {
        return Ok(());
    };
    for round in 0..50u8 {
        let mut device = writer.alloc_zeros::<u8>(64 << 20)?;
        {
            let bytes = device.num_bytes();
            let (pointer, _record) = device.device_ptr_mut(&writer);
            // SAFETY: The allocation holds `bytes` bytes.
            unsafe { sys::cuMemsetD8Async(pointer, round, bytes, writer.cu_stream()) }.result()?;
        }
        let tail = reader.clone_dtoh(&device.slice((64 << 20) - 16..))?;
        assert!(tail.iter().all(|value| *value == round));
    }
    context.check_err()
}

/// A free after a read on another stream waits for that read, and the memory a later
/// allocation reuses is not overwritten under it.
#[test]
fn frees_follow_reads_on_other_streams() -> Result<(), DriverError> {
    let Some((context, owner, reader)) = streams() else {
        return Ok(());
    };
    let source = owner.clone_htod(&vec![7u8; 32 << 20])?;
    let mut copies = Vec::new();
    for _ in 0..8 {
        // SAFETY: Fully written by the copy below.
        let mut target = unsafe { reader.alloc::<u8>(32 << 20) }?;
        reader.memcpy_dtod(&source, &mut target)?;
        copies.push(target);
    }
    drop(source);
    let reused = owner.clone_htod(&vec![9u8; 32 << 20])?;
    for copy in &copies {
        assert!(reader.clone_dtoh(copy)?.iter().all(|value| *value == 7));
    }
    assert!(owner.clone_dtoh(&reused)?.iter().all(|value| *value == 9));
    context.check_err()
}

/// Same-stream accesses take no events; cross-stream ones reuse returned events.
#[test]
fn events_are_pooled() -> Result<(), DriverError> {
    let Some((context, first, second)) = streams() else {
        return Ok(());
    };
    for _ in 0..100 {
        let device = first.clone_htod(&[1u32; 1024])?;
        let (_pointer, record) = device.device_ptr(&second);
        drop(record);
    }
    // One write event and one read event, reused for every buffer.
    assert!(context.pooled_events() <= 2);
    context.check_err()
}

/// An error recorded elsewhere neither makes a free skip, nor is consumed by it.
#[test]
fn recorded_errors_do_not_leak_frees() -> Result<(), DriverError> {
    let Some((context, owner, reader)) = streams() else {
        return Ok(());
    };
    if !context.has_async_alloc() {
        return Ok(());
    }
    let bytes = 4 << 30;
    let used_before = pool_used_bytes(&context)?;
    let device = owner.alloc_zeros::<u8>(bytes)?;
    let (_pointer, record) = device.device_ptr(&reader);
    drop(record);
    let recorded = DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE);
    context.record_err::<()>(Err(recorded));
    drop(device);
    assert_eq!(context.check_err(), Err(recorded));
    owner.synchronize()?;
    // Other tests allocate concurrently, so only a leak of most of the allocation counts.
    assert!(pool_used_bytes(&context)? < used_before + bytes as u64 / 2);
    context.check_err()
}

/// The bytes this process uses from the device's default memory pool.
fn pool_used_bytes(context: &CudaContext) -> Result<u64, DriverError> {
    let ordinal = i32::try_from(context.ordinal())
        .map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))?;
    let (mut device, mut pool, mut used) = (0, std::ptr::null_mut(), 0u64);
    // SAFETY: The driver writes the device and its pool, and the attribute is a `u64`.
    unsafe {
        sys::cuDeviceGet(&raw mut device, ordinal).result()?;
        sys::cuDeviceGetMemPool(&raw mut pool, device).result()?;
        sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            (&raw mut used).cast(),
        )
        .result()?;
    }
    Ok(used)
}

/// The first recorded error is kept until checked, as later ones are likely its consequences.
#[test]
fn first_recorded_error_is_kept() -> Result<(), DriverError> {
    let Some((context, ..)) = streams() else {
        return Ok(());
    };
    let first = DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE);
    let second = DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY);
    context.record_err::<()>(Err(first));
    context.record_err::<()>(Err(second));
    assert_eq!(context.check_err(), Err(first));
    context.check_err()
}

/// Memory of one context cannot be accessed from another context's stream.
#[test]
fn rejects_access_from_another_context() -> Result<(), DriverError> {
    let Some((context, owner, _)) = streams() else {
        return Ok(());
    };
    let other = CudaContext::new_non_primary(0, 0)?;
    let stream = other.new_stream()?;
    let device = owner.clone_htod(&[1u32; 16])?;
    let access = std::panic::catch_unwind(AssertUnwindSafe(|| drop(device.device_ptr(&stream))));
    assert!(access.is_err());
    drop(device);
    context.check_err()
}

/// Reports fewer elements than the slice it copies.
struct ShortHostSlice([u32; 4]);

impl HostSlice<u32> for ShortHostSlice {
    fn len(&self) -> usize {
        1
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [u32], SyncOnDrop<'a>) {
        (&self.0, SyncOnDrop::none())
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [u32], SyncOnDrop<'a>) {
        (&mut self.0, SyncOnDrop::none())
    }
}

/// Host copies are bounded by the slices they copy, not by the reported lengths.
#[test]
fn host_copies_check_copied_slices() -> Result<(), DriverError> {
    let Some((context, stream, _)) = streams() else {
        return Ok(());
    };
    let mut host = ShortHostSlice([7; 4]);
    let mut device = stream.alloc_zeros::<u32>(1)?;
    let upload =
        std::panic::catch_unwind(AssertUnwindSafe(|| stream.memcpy_htod(&host, &mut device)));
    assert!(upload.is_err());
    let download =
        std::panic::catch_unwind(AssertUnwindSafe(|| stream.memcpy_dtoh(&device, &mut host)));
    assert!(download.is_ok());
    assert_eq!(host.0, [0, 7, 7, 7]);
    context.check_err()
}
