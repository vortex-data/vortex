// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaStream;
use cudarc::driver::HostSlice;
use cudarc::driver::PinnedHostSlice;
use cudarc::driver::SyncOnDrop;
use parking_lot::Mutex;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;
use vortex::utils::aliases::hash_map::HashMap;

use crate::CudaDeviceBuffer;
use crate::stream::VortexCudaStream;

/// A page-locked host buffer allocated by CUDA.
///
/// This is intended as a staging buffer for H2D transfers. Contents are uninitialized after
/// allocation.
pub(crate) struct PinnedByteBuffer {
    inner: PinnedHostSlice<u8>,
    logical_len: usize,
}

#[expect(clippy::same_name_method)]
impl PinnedByteBuffer {
    /// Allocate a pinned host buffer with a given capacity and logical length.
    ///
    /// # Safety
    /// The returned buffer's contents are uninitialized. The caller must initialize before read.
    pub(crate) unsafe fn uninit_with_capacity(
        ctx: &Arc<CudaContext>,
        capacity: usize,
        logical_len: usize,
    ) -> VortexResult<Self> {
        // alloc_pinned uses CU_MEMHOSTALLOC_WRITECOMBINED: fast for host writes
        // and H2D transfers, but very slow for host-side reads.
        let inner = unsafe {
            ctx.alloc_pinned::<u8>(capacity)
                .map_err(|e| vortex_err!("failed to allocate pinned host buffer: {e}"))?
        };
        Ok(Self { inner, logical_len })
    }

    /// Returns the length of the buffer in bytes.
    pub(crate) fn len(&self) -> usize {
        self.logical_len
    }

    pub(crate) fn capacity(&self) -> usize {
        self.inner.len()
    }

    /// Returns the buffer as a mutable slice.
    pub(crate) fn as_mut_slice(&mut self) -> VortexResult<&mut [u8]> {
        self.inner
            .as_mut_slice()
            .map(|slice| &mut slice[..self.logical_len])
            .map_err(|e| vortex_err!("failed to access pinned host buffer: {e}"))
    }

    fn set_logical_len(&mut self, len: usize) {
        assert!(len <= self.inner.len());
        self.logical_len = len;
    }
}

impl HostSlice<u8> for PinnedByteBuffer {
    fn len(&self) -> usize {
        self.len()
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [u8], SyncOnDrop<'a>) {
        let (slice, sync) = unsafe {
            <PinnedHostSlice<u8> as HostSlice<u8>>::stream_synced_slice(&self.inner, stream)
        };
        (&slice[..self.logical_len], sync)
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [u8], SyncOnDrop<'a>) {
        let (slice, sync) = unsafe {
            <PinnedHostSlice<u8> as HostSlice<u8>>::stream_synced_mut_slice(&mut self.inner, stream)
        };
        (&mut slice[..self.logical_len], sync)
    }
}

/// A simple pinned host buffer pool keyed by allocation size.
///
/// Requested sizes are rounded up to the next power of two so that buffers are reusable across
/// similar-but-not-identical request sizes.
pub struct PinnedByteBufferPool {
    ctx: Arc<CudaContext>,
    max_keep_per_size: usize,
    buckets: Mutex<HashMap<usize, Vec<PinnedByteBuffer>>>,
    inflight: Mutex<Vec<InflightPinnedBuffer>>,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
    allocs: std::sync::atomic::AtomicU64,
    puts: std::sync::atomic::AtomicU64,
}

struct InflightPinnedBuffer {
    event: Arc<CudaEvent>,
    buffer: PinnedByteBuffer,
}

impl PinnedByteBufferPool {
    /// Create a new pool with default limits.
    pub fn new(ctx: Arc<CudaContext>) -> Self {
        Self::with_limits(ctx, 256)
    }

    /// Create a new pool with a maximum number of cached buffers per size class.
    pub fn with_limits(ctx: Arc<CudaContext>, max_keep_per_size: usize) -> Self {
        Self {
            ctx,
            max_keep_per_size: max_keep_per_size.max(1),
            buckets: Mutex::new(HashMap::new()),
            inflight: Mutex::new(Vec::new()),
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
            allocs: std::sync::atomic::AtomicU64::new(0),
            puts: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Try to get a pooled pinned buffer without performing expensive allocation.
    ///
    /// Returns `Ok(None)` if no buffer is available in the pool for the requested size class.
    /// Unlike `get`, this will never call `cuMemAllocHost`.
    pub fn try_get(self: &Arc<Self>, len: usize) -> VortexResult<Option<PooledPinnedBuffer>> {
        match self.try_get_inner(len)? {
            Some(inner) => Ok(Some(PooledPinnedBuffer::new(inner, Arc::clone(self)))),
            None => Ok(None),
        }
    }

    /// Acquire a pooled pinned buffer of the given size in bytes.
    ///
    /// The buffer is returned to the pool when the [`PooledPinnedBuffer`] is dropped.
    pub(crate) fn get(self: &Arc<Self>, len: usize) -> VortexResult<PooledPinnedBuffer> {
        let inner = self.get_inner(len)?;
        Ok(PooledPinnedBuffer::new(inner, Arc::clone(self)))
    }

    /// Defer returning a pinned buffer to the pool until the CUDA event completes.
    pub(crate) fn put_inflight(
        &self,
        event: Arc<CudaEvent>,
        buffer: PinnedByteBuffer,
    ) -> VortexResult<()> {
        let mut inflight = self.inflight.lock();
        inflight.push(InflightPinnedBuffer { event, buffer });
        Ok(())
    }

    /// Snapshot pool reuse statistics.
    pub fn stats(&self) -> PinnedPoolStats {
        PinnedPoolStats {
            hits: self.hits.load(std::sync::atomic::Ordering::Relaxed),
            misses: self.misses.load(std::sync::atomic::Ordering::Relaxed),
            allocs: self.allocs.load(std::sync::atomic::Ordering::Relaxed),
            puts: self.puts.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    fn size_class_len(&self, len: usize) -> usize {
        if len == 0 { 0 } else { len.next_power_of_two() }
    }

    /// Reclaim inflight pinned buffers whose CUDA completion events have fired.
    ///
    /// Completed buffers are moved back into size-class buckets for reuse.
    fn reclaim_completed(&self) -> VortexResult<()> {
        let mut inflight = self.inflight.lock();
        if inflight.is_empty() {
            return Ok(());
        }
        self.ctx
            .bind_to_thread()
            .map_err(|e| vortex_err!("Failed to bind CUDA context: {e}"))?;
        let mut idx = 0usize;
        while idx < inflight.len() {
            if !inflight[idx].event.is_complete() {
                idx += 1;
                continue;
            }
            let completed = inflight.swap_remove(idx);
            self.put(completed.buffer);
        }
        Ok(())
    }

    fn get_inner(&self, len: usize) -> VortexResult<PinnedByteBuffer> {
        self.reclaim_completed()?;
        let key_len = self.size_class_len(len);
        {
            let mut buckets = self.buckets.lock();
            if let Some(bucket) = buckets.get_mut(&key_len)
                && let Some(buf) = bucket.pop()
            {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let mut buf = buf;
                buf.set_logical_len(len);
                return Ok(buf);
            }
        }
        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.allocs
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { PinnedByteBuffer::uninit_with_capacity(&self.ctx, key_len, len) }
    }

    fn put(&self, buf: PinnedByteBuffer) {
        let len = buf.capacity();
        let overflow = {
            let mut buckets = self.buckets.lock();
            let bucket = buckets.entry(len).or_default();
            if bucket.len() < self.max_keep_per_size {
                bucket.push(buf);
                None
            } else {
                Some(buf)
            }
        };
        // If the pool is full, the buffer (cuMemFreeHost) is dropped outside the lock.
        drop(overflow);
        self.puts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn try_get_inner(&self, len: usize) -> VortexResult<Option<PinnedByteBuffer>> {
        self.reclaim_completed()?;
        let key_len = self.size_class_len(len);
        let mut buckets = self.buckets.lock();
        if let Some(bucket) = buckets.get_mut(&key_len)
            && let Some(mut buf) = bucket.pop()
        {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            buf.set_logical_len(len);
            Ok(Some(buf))
        } else {
            Ok(None)
        }
    }
}

/// Reuse counters for a pinned buffer pool.
#[derive(Clone, Copy, Debug, Default)]
pub struct PinnedPoolStats {
    pub hits: u64,
    pub misses: u64,
    pub allocs: u64,
    pub puts: u64,
}

/// A pinned buffer that is returned to its pool when dropped.
///
/// This wrapper owns a pinned byte buffer and ensures it gets returned to the
/// [`PinnedByteBufferPool`] when the buffer is no longer needed. This enables efficient
/// buffer reuse for I/O operations.
pub struct PooledPinnedBuffer {
    inner: Option<PinnedByteBuffer>,
    pool: Arc<PinnedByteBufferPool>,
}

impl PooledPinnedBuffer {
    /// Create a new pooled buffer.
    pub(crate) fn new(inner: PinnedByteBuffer, pool: Arc<PinnedByteBufferPool>) -> Self {
        Self {
            inner: Some(inner),
            pool,
        }
    }

    /// Returns the buffer as a mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if the buffer has already been consumed or if the CUDA context is invalid.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let inner = self
            .inner
            .as_mut()
            .unwrap_or_else(|| vortex_panic!("buffer already consumed"));
        inner
            .as_mut_slice()
            .unwrap_or_else(|e| vortex_panic!("failed to access pinned host buffer: {e}"))
    }

    /// Submits a non-blocking H2D DMA transfer and returns a device buffer.
    ///
    /// The pinned buffer is placed in the pool's inflight queue, gated on a `CudaEvent` marking
    /// the transfer completion. The pool reclaims it once the event fires.
    pub fn transfer_to_device(
        mut self,
        stream: &VortexCudaStream,
    ) -> VortexResult<CudaDeviceBuffer> {
        let pinned = self
            .inner
            .as_ref()
            .unwrap_or_else(|| vortex_panic!("buffer already consumed"));
        let len = pinned.len();

        let mut cuda_slice = stream.device_alloc::<u8>(len)?;

        // Async because the pinned buffer is page-locked: memcpy_htod returns a
        // SyncOnDrop::Record (non-blocking) rather than SyncOnDrop::Sync.
        stream
            .memcpy_htod(pinned, &mut cuda_slice)
            .map_err(|e| vortex_err!("Failed to schedule H2D copy: {}", e))?;

        let event = Arc::new(
            stream
                .record_event(None)
                .map_err(|e| vortex_err!("Failed to record CUDA event: {}", e))?,
        );

        // Take ownership only after all fallible ops succeed. Before this point,
        // errors cause `self` to drop, returning the buffer to the pool.
        let inner = self
            .inner
            .take()
            .unwrap_or_else(|| vortex_panic!("buffer already consumed"));
        self.pool.put_inflight(event, inner)?;

        Ok(CudaDeviceBuffer::new(cuda_slice))
    }
}

impl Drop for PooledPinnedBuffer {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            self.pool.put(inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::CudaContext;
    use vortex::array::buffer::DeviceBuffer;
    use vortex::buffer::Alignment;
    use vortex::error::VortexResult;

    use super::*;

    fn setup() -> VortexResult<(Arc<PinnedByteBufferPool>, VortexCudaStream)> {
        let ctx = CudaContext::new(0).map_err(|e| vortex_err!("Failed to initialize CUDA: {e}"))?;
        let pool = Arc::new(PinnedByteBufferPool::new(Arc::clone(&ctx)));
        let stream = VortexCudaStream(
            ctx.new_stream()
                .map_err(|e| vortex_err!("Failed to create stream: {e}"))?,
        );
        Ok((pool, stream))
    }

    #[crate::test]
    fn transfer_to_device_round_trip() -> VortexResult<()> {
        let (pool, stream) = setup()?;
        let data: Vec<u8> = (0..=255u8).collect();

        let mut pinned = pool.get(data.len())?;
        pinned.as_mut_slice().copy_from_slice(&data);

        let device_buf = pinned.transfer_to_device(&stream)?;

        let host_buf = device_buf.copy_to_host_sync(Alignment::of::<u8>())?;
        assert_eq!(host_buf.as_ref(), &data[..]);
        Ok(())
    }

    #[crate::test]
    fn transfer_puts_buffer_inflight() -> VortexResult<()> {
        let (pool, stream) = setup()?;

        let mut pinned = pool.get(1024)?;
        pinned.as_mut_slice().fill(0xAB);

        let stats_before = pool.stats();
        assert_eq!(stats_before.allocs, 1);
        assert_eq!(stats_before.puts, 0);
        assert_eq!(stats_before.hits, 0);

        let _device_buf = pinned.transfer_to_device(&stream)?;

        // put_inflight does not increment the puts counter
        let stats_after = pool.stats();
        assert_eq!(stats_after.puts, 0);

        // The buffer is in the inflight queue, not yet in a bucket
        assert_eq!(pool.inflight.lock().len(), 1);

        Ok(())
    }

    #[crate::test]
    fn pool_reclaims_after_transfer_completes() -> VortexResult<()> {
        let (pool, stream) = setup()?;

        let mut pinned = pool.get(1024)?;
        pinned.as_mut_slice().fill(0xCD);

        let _device_buf = pinned.transfer_to_device(&stream)?;

        // Sync the stream so the recorded event completes.
        stream
            .synchronize()
            .map_err(|e| vortex_err!("Failed to sync stream: {e}"))?;

        assert_eq!(pool.stats().hits, 0);
        assert_eq!(pool.stats().allocs, 1);

        // get_inner calls reclaim_completed, which finds the completed event
        // and moves the buffer back into a bucket before serving the request.
        let _pinned2 = pool.get(1024)?;

        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        // reclaim_completed called put once for the completed buffer
        assert_eq!(stats.puts, 1);
        // No additional allocation was needed
        assert_eq!(stats.allocs, 1);
        // Inflight queue is now drained
        assert_eq!(pool.inflight.lock().len(), 0);

        Ok(())
    }

    #[crate::test]
    fn drop_returns_buffer_to_pool() -> VortexResult<()> {
        let (pool, _stream) = setup()?;

        {
            let mut pinned = pool.get(512)?;
            pinned.as_mut_slice().fill(0);
        }

        let stats = pool.stats();
        assert_eq!(stats.puts, 1);
        assert_eq!(stats.allocs, 1);

        // Getting again should be a pool hit, not a new allocation.
        let _pinned2 = pool.get(512)?;
        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.allocs, 1);

        Ok(())
    }

    #[crate::test]
    fn transfer_consumes_inner_so_drop_is_noop() -> VortexResult<()> {
        let (pool, stream) = setup()?;

        let mut pinned = pool.get(256)?;
        pinned.as_mut_slice().fill(0xFF);

        // transfer_to_device takes self and moves inner into inflight.
        // The PooledPinnedBuffer's Drop should not double-return the buffer.
        let _device_buf = pinned.transfer_to_device(&stream)?;

        // Sync and reclaim so the single inflight buffer returns.
        stream
            .synchronize()
            .map_err(|e| vortex_err!("Failed to sync stream: {e}"))?;
        pool.reclaim_completed()?;

        // Exactly one put from reclaim, zero from Drop.
        assert_eq!(pool.stats().puts, 1);

        Ok(())
    }
}
