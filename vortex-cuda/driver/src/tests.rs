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
use crate::mem_pool;
use crate::memory;
use crate::sys;
use crate::sys::CUmemPool_attribute as PoolAttr;

/// A context and two streams, or `None` without a CUDA driver or device.
///
/// Any other failure fails the test, rather than passing it without running it.
fn streams() -> Option<(Arc<CudaContext>, Arc<CudaStream>, Arc<CudaStream>)> {
    if !driver_available(sys::driver_load_error()) {
        return None;
    }
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(DriverError(sys::CUresult::CUDA_ERROR_NO_DEVICE)) => return None,
        Err(error) => panic!("CUDA initialization failed: {error}"),
    };
    let first = context.new_stream().expect("stream");
    let second = context.new_stream().expect("stream");
    Some((context, first, second))
}

fn driver_available(error: Option<&sys::DriverLoadError>) -> bool {
    match error {
        None => true,
        Some(sys::DriverLoadError::LibraryUnavailable) => false,
        Some(error) => panic!("CUDA driver loading failed: {error}"),
    }
}

#[test]
fn missing_driver_skips_tests() {
    assert!(!driver_available(Some(
        &sys::DriverLoadError::LibraryUnavailable
    )));
}

#[test]
fn loaded_driver_runs_tests() {
    assert!(driver_available(None));
}

#[test]
#[should_panic(expected = "CUDA driver loading failed: cuMemAllocAsync: undefined symbol")]
fn missing_driver_symbol_fails_tests() {
    driver_available(Some(&sys::DriverLoadError::MissingSymbol(
        "cuMemAllocAsync: undefined symbol".to_owned(),
    )));
}

#[test]
fn host_device_round_trip() -> Result<(), DriverError> {
    let Some((_context, stream, _)) = streams() else {
        return Ok(());
    };
    let values: Vec<u32> = (0..100_000).collect();
    let device = stream.clone_htod(&values)?;
    assert_eq!(stream.clone_dtoh(&device)?, values);
    assert_eq!(stream.clone_dtoh(&device.slice(10..20))?, &values[10..20]);
    Ok(())
}

/// A write on one stream is visible to reads on another, through views as well.
#[test]
fn cross_stream_reads_follow_writes() -> Result<(), DriverError> {
    let Some((_context, writer, reader)) = streams() else {
        return Ok(());
    };
    for round in 0..50u8 {
        let mut device = writer.alloc_zeros::<u8>(64 << 20)?;
        {
            let bytes = device.num_bytes();
            let (pointer, record) = device.device_ptr_mut(&writer)?;
            // SAFETY: The allocation holds `bytes` bytes.
            let result =
                unsafe { sys::cuMemsetD8Async(pointer, round, bytes, writer.cu_stream()) }.result();
            let finished = record.finish();
            result?;
            finished?;
        }
        let tail = reader.clone_dtoh(&device.slice((64 << 20) - 16..))?;
        assert!(tail.iter().all(|value| *value == round));
    }
    Ok(())
}

/// A free after a read on another stream waits for that read, and the memory a later
/// allocation reuses is not overwritten under it.
#[test]
fn frees_follow_reads_on_other_streams() -> Result<(), DriverError> {
    let Some((_context, owner, reader)) = streams() else {
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
    Ok(())
}

/// Writes and cross-stream reads reuse events returned by earlier allocations.
#[test]
fn events_are_pooled() -> Result<(), DriverError> {
    let Some((context, first, second)) = streams() else {
        return Ok(());
    };
    for _ in 0..100 {
        let device = first.clone_htod(&[1u32; 1024])?;
        let (_pointer, record) = device.device_ptr(&second)?;
        record.finish()?;
    }
    // One write event and one read event, reused for every buffer.
    assert!(context.pooled_events() <= 2);
    Ok(())
}

/// A host slice whose acquisition fails before a copy can be enqueued.
struct FailedHostSlice(DriverError);

impl HostSlice<u32> for FailedHostSlice {
    fn len(&self) -> usize {
        4
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> Result<(&'a [u32], SyncOnDrop<'a>), DriverError> {
        Err(self.0)
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> Result<(&'a mut [u32], SyncOnDrop<'a>), DriverError> {
        Err(self.0)
    }
}

#[test]
fn host_acquisition_errors_return_directly_without_poisoning_context() -> Result<(), DriverError> {
    let Some((context, stream, _)) = streams() else {
        return Ok(());
    };
    let error = DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE);
    let mut host = FailedHostSlice(error);
    let mut device = stream.clone_htod(&[7u32; 4])?;
    assert_eq!(stream.memcpy_htod(&host, &mut device), Err(error));
    assert_eq!(stream.memcpy_dtoh(&device, &mut host), Err(error));
    assert_eq!(stream.clone_htod(&host).err(), Some(error));
    context.bind_to_thread()?;
    assert_eq!(stream.clone_dtoh(&device)?, [7; 4]);
    Ok(())
}

/// Memory of one context cannot be accessed from another context's stream.
#[test]
fn rejects_access_from_another_context() -> Result<(), DriverError> {
    let Some((_context, owner, _)) = streams() else {
        return Ok(());
    };
    let other = CudaContext::new_non_primary(0, 0)?;
    let stream = other.new_stream()?;
    let device = owner.clone_htod(&[1u32; 16])?;
    let access = std::panic::catch_unwind(AssertUnwindSafe(|| drop(device.device_ptr(&stream))));
    assert!(access.is_err());
    drop(device);
    Ok(())
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
    ) -> Result<(&'a [u32], SyncOnDrop<'a>), DriverError> {
        Ok((&self.0, SyncOnDrop::none()))
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> Result<(&'a mut [u32], SyncOnDrop<'a>), DriverError> {
        Ok((&mut self.0, SyncOnDrop::none()))
    }
}

/// Host copies are bounded by the slices they copy, not by the reported lengths.
#[test]
fn host_copies_check_copied_slices() -> Result<(), DriverError> {
    let Some((_context, stream, _)) = streams() else {
        return Ok(());
    };
    let mut host = ShortHostSlice([7; 4]);
    let mut device = stream.alloc_zeros::<u32>(1)?;
    let upload =
        std::panic::catch_unwind(AssertUnwindSafe(|| stream.memcpy_htod(&host, &mut device)));
    assert!(upload.is_err());
    let download =
        std::panic::catch_unwind(AssertUnwindSafe(|| stream.memcpy_dtoh(&device, &mut host)));
    download.expect("download must not panic")?;
    assert_eq!(host.0, [0, 7, 7, 7]);
    Ok(())
}

/// Reinterpreting memory as more elements than fit fails, even if their size overflows.
#[test]
fn transmute_rejects_overflowing_lengths() -> Result<(), DriverError> {
    let Some((_context, stream, _)) = streams() else {
        return Ok(());
    };
    let device = stream.alloc_zeros::<u8>(16)?;
    // SAFETY: Zeroed bytes are valid `u32`s, and allocations are aligned for them.
    unsafe {
        assert!(device.transmute::<u32>(4).is_some());
        assert!(device.transmute::<u32>(5).is_none());
        assert!(device.transmute::<u32>(usize::MAX / 2 + 1).is_none());
    }
    Ok(())
}

/// Empty allocations succeed with either allocator and support every access.
#[test]
fn empty_allocations() -> Result<(), DriverError> {
    let Some((_context, owner, other)) = streams() else {
        return Ok(());
    };
    let device = owner.clone_htod(&Vec::<u32>::new())?;
    let mut copy = other.alloc_zeros::<u32>(0)?;
    other.memcpy_dtod(&device, &mut copy)?;
    assert!(owner.clone_dtoh(&copy)?.is_empty());
    drop(device);
    drop(copy);
    Ok(())
}

#[test]
fn borrowed_host_slice_binds_context() -> Result<(), DriverError> {
    let Some((context, stream, _)) = streams() else {
        return Ok(());
    };
    // SAFETY: Clearing the calling thread's current context does not destroy it.
    unsafe { sys::cuCtxSetCurrent(std::ptr::null_mut()) }.result()?;

    let host = [1u8; 16];
    // SAFETY: No work uses the slice; the guard is dropped below.
    let (_, guard) = unsafe { host.stream_synced_slice(&stream) }?;
    let mut current = std::ptr::null_mut();
    // SAFETY: The driver writes the calling thread's current context.
    unsafe { sys::cuCtxGetCurrent(&raw mut current) }.result()?;
    assert_eq!(current, context.cu_ctx());
    guard.finish()
}

/// A copy from a borrowed page-locked slice no longer reads it once the copy returns, so the
/// slice can be reused right away.
#[test]
fn page_locked_slices_are_read_before_copies_return() -> Result<(), DriverError> {
    let Some((_context, stream, _)) = streams() else {
        return Ok(());
    };
    let len = 1 << 20;
    // SAFETY: The context is current after `streams`, and the memory is freed below.
    let pointer = unsafe { memory::malloc_host(len, 0) }?;
    // SAFETY: The allocation holds `len` bytes, initialized below.
    let host = unsafe { std::slice::from_raw_parts_mut(pointer.cast::<u8>(), len) };
    host.fill(1);
    let mut device = stream.alloc_zeros::<u8>(len)?;
    stream.memcpy_htod(&*host, &mut device)?;
    host.fill(2);
    assert!(stream.clone_dtoh(&device)?.iter().all(|byte| *byte == 1));
    stream.memcpy_dtoh(&device, host)?;
    assert!(host.iter().all(|byte| *byte == 1));
    // SAFETY: The copies accessing the memory completed.
    unsafe { memory::free_host(pointer) }?;
    Ok(())
}

fn pooled_stream() -> Option<Arc<CudaStream>> {
    let (context, stream, _) = streams()?;
    if !context.has_async_alloc() {
        return None;
    }
    assert!(!context.mem_pool().is_null());
    Some(stream)
}

fn pool_attribute(
    pool: sys::CUmemoryPool,
    attr: sys::CUmemPool_attribute,
) -> Result<u64, DriverError> {
    let mut value = 0u64;
    // SAFETY: These tests pass live pools and attributes with u64 values.
    unsafe { mem_pool::get_attribute(pool, attr, (&raw mut value).cast()) }?;
    Ok(value)
}

#[test]
fn retaining_pool_threshold_is_half_device_memory() -> Result<(), DriverError> {
    let Some(stream) = pooled_stream() else {
        return Ok(());
    };
    let context = stream.context();
    let threshold = pool_attribute(
        context.mem_pool(),
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
    )?;
    let mut total = 0;
    // SAFETY: The context keeps the device valid; the driver writes its memory size.
    unsafe { sys::cuDeviceTotalMem_v2(&raw mut total, context.cu_device()) }.result()?;
    assert_eq!(threshold, total as u64 / 2);
    Ok(())
}

#[test]
fn retaining_pool_keeps_freed_memory() -> Result<(), DriverError> {
    let Some(stream) = pooled_stream() else {
        return Ok(());
    };
    let pool = stream.context().mem_pool();
    let used = PoolAttr::CU_MEMPOOL_ATTR_USED_MEM_CURRENT;
    let reserved = PoolAttr::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT;
    let device = stream.alloc_zeros::<u32>(1024)?;
    stream.synchronize()?;
    assert!(pool_attribute(pool, used)? > 0);

    drop(device);
    stream.synchronize()?;
    assert_eq!(pool_attribute(pool, used)?, 0);
    assert!(pool_attribute(pool, reserved)? > 0);
    Ok(())
}

#[test]
fn borrowed_pool_is_isolated_and_not_destroyed_with_context() -> Result<(), DriverError> {
    let Some(default_stream) = pooled_stream() else {
        return Ok(());
    };
    let default_context = default_stream.context();
    let default_pool = default_context.mem_pool();
    let host = CudaContext::new(0)?;
    let host_pool = host.mem_pool();
    assert!(!host_pool.is_null());
    assert_ne!(host_pool, default_pool);
    // SAFETY: `host` owns a pool for device 0 and outlives the borrower and its synchronized frees.
    let borrowed = unsafe { CudaContext::new_with_mem_pool(0, host_pool) }?;
    assert_eq!(borrowed.cu_ctx(), default_context.cu_ctx());
    assert_eq!(borrowed.mem_pool(), host_pool);
    assert_eq!(default_context.mem_pool(), default_pool);

    let stream = borrowed.new_stream()?;
    let device = stream.alloc_zeros::<u32>(1024)?;
    stream.synchronize()?;
    let used = PoolAttr::CU_MEMPOOL_ATTR_USED_MEM_CURRENT;
    assert!(pool_attribute(host_pool, used)? > 0);
    assert_eq!(pool_attribute(default_pool, used)?, 0);

    drop(device);
    stream.synchronize()?;
    drop(stream);
    drop(borrowed);
    assert_eq!(pool_attribute(host_pool, used)?, 0);
    assert!(pool_attribute(host_pool, PoolAttr::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)? > 0);

    let device = default_stream.alloc_zeros::<u32>(1024)?;
    default_stream.synchronize()?;
    assert!(pool_attribute(default_pool, used)? > 0);
    assert_eq!(pool_attribute(host_pool, used)?, 0);
    drop(device);
    default_stream.synchronize()
}

#[test]
fn null_pool_uses_device_pool_without_changing_other_contexts() -> Result<(), DriverError> {
    let Some(default_stream) = pooled_stream() else {
        return Ok(());
    };
    let default_pool = default_stream.context().mem_pool();
    // SAFETY: Null selects the device-managed pool, with no borrowed handle to keep alive.
    let context = unsafe { CudaContext::new_with_mem_pool(0, std::ptr::null_mut()) }?;
    assert!(context.mem_pool().is_null());
    assert_eq!(default_stream.context().mem_pool(), default_pool);

    let stream = context.new_stream()?;
    let device = stream.alloc_zeros::<u32>(1024)?;
    stream.synchronize()?;
    assert_eq!(
        pool_attribute(default_pool, PoolAttr::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
        0
    );
    drop(device);
    stream.synchronize()
}
