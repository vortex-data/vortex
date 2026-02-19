// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use bytes::Bytes;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaStream;
use cudarc::driver::HostSlice;
use cudarc::driver::PinnedHostSlice;
use cudarc::driver::SyncOnDrop;
use parking_lot::Mutex;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_utils::aliases::hash_map::HashMap;

/// A page-locked host buffer allocated by CUDA.
///
/// This is intended as a staging buffer for H2D transfers. Contents are uninitialized after
/// allocation.
pub struct PinnedByteBuffer {
    inner: PinnedHostSlice<u8>,
    logical_len: usize,
}

#[allow(clippy::same_name_method)]
impl PinnedByteBuffer {
    /// Allocate a pinned host buffer with uninitialized contents.
    ///
    /// # Safety
    /// The returned buffer's contents are uninitialized. The caller must initialize before read.
    pub unsafe fn uninit(ctx: &Arc<CudaContext>, len: usize) -> VortexResult<Self> {
        let inner = unsafe {
            ctx.alloc_pinned::<u8>(len)
                .map_err(|e| vortex_err!("failed to allocate pinned host buffer: {e}"))?
        };
        Ok(Self {
            inner,
            logical_len: len,
        })
    }

    /// Allocate a pinned host buffer with a given capacity and logical length.
    ///
    /// # Safety
    /// The returned buffer's contents are uninitialized. The caller must initialize before read.
    pub unsafe fn uninit_with_capacity(
        ctx: &Arc<CudaContext>,
        capacity: usize,
        logical_len: usize,
    ) -> VortexResult<Self> {
        let inner = unsafe {
            ctx.alloc_pinned::<u8>(capacity)
                .map_err(|e| vortex_err!("failed to allocate pinned host buffer: {e}"))?
        };
        Ok(Self { inner, logical_len })
    }

    /// Returns the length of the buffer in bytes.
    pub fn len(&self) -> usize {
        self.logical_len
    }

    pub fn capacity(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.logical_len == 0
    }

    /// Returns the buffer as an immutable slice.
    pub fn as_slice(&self) -> VortexResult<&[u8]> {
        self.inner
            .as_slice()
            .map(|slice| &slice[..self.logical_len])
            .map_err(|e| vortex_err!("failed to access pinned host buffer: {e}"))
    }

    /// Returns the buffer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> VortexResult<&mut [u8]> {
        self.inner
            .as_mut_slice()
            .map(|slice| &mut slice[..self.logical_len])
            .map_err(|e| vortex_err!("failed to access pinned host buffer: {e}"))
    }

    /// Returns a raw pointer to the buffer.
    pub fn as_ptr(&self) -> VortexResult<*const u8> {
        self.inner
            .as_ptr()
            .map_err(|e| vortex_err!("failed to access pinned host buffer: {e}"))
    }

    /// Returns a mutable raw pointer to the buffer.
    pub fn as_mut_ptr(&mut self) -> VortexResult<*mut u8> {
        self.inner
            .as_mut_ptr()
            .map_err(|e| vortex_err!("failed to access pinned host buffer: {e}"))
    }

    fn set_logical_len(&mut self, len: usize) {
        debug_assert!(len <= self.inner.len());
        self.logical_len = len;
    }

    /// Returns the CUDA context that owns this allocation.
    pub fn context(&self) -> &Arc<CudaContext> {
        self.inner.context()
    }
}

#[allow(clippy::same_name_method)]
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
pub struct PinnedByteBufferPool {
    ctx: Arc<CudaContext>,
    max_keep_per_size: usize,
    buckets: Mutex<HashMap<usize, Vec<PinnedByteBuffer>>>,
    deferred: Mutex<Vec<DeferredPinnedBuffer>>,
    round_len_pow2: bool,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
    allocs: std::sync::atomic::AtomicU64,
    puts: std::sync::atomic::AtomicU64,
}

struct DeferredPinnedBuffer {
    event: Arc<CudaEvent>,
    buffer: PinnedByteBuffer,
}

impl PinnedByteBufferPool {
    /// Create a new pool with default limits.
    pub fn new(ctx: Arc<CudaContext>) -> Self {
        Self::with_limits_pow2(ctx, 256)
    }

    /// Create a new pool with a maximum number of cached buffers per size.
    pub fn with_limits(ctx: Arc<CudaContext>, max_keep_per_size: usize) -> Self {
        Self {
            ctx,
            max_keep_per_size: max_keep_per_size.max(1),
            buckets: Mutex::new(HashMap::new()),
            deferred: Mutex::new(Vec::new()),
            round_len_pow2: false,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
            allocs: std::sync::atomic::AtomicU64::new(0),
            puts: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Create a new pool with a maximum number of cached buffers per size and power-of-two buckets.
    pub fn with_limits_pow2(ctx: Arc<CudaContext>, max_keep_per_size: usize) -> Self {
        let mut pool = Self::with_limits(ctx, max_keep_per_size);
        pool.round_len_pow2 = true;
        pool
    }

    fn size_class_len(&self, len: usize) -> usize {
        if !self.round_len_pow2 {
            return len;
        }
        if len == 0 { 0 } else { len.next_power_of_two() }
    }

    fn reclaim_deferred(&self) -> VortexResult<()> {
        let mut deferred = self.deferred.lock();
        if deferred.is_empty() {
            return Ok(());
        }
        self.ctx
            .bind_to_thread()
            .map_err(|e| vortex_err!("Failed to bind CUDA context: {e}"))?;
        let mut buckets = self.buckets.lock();
        let mut idx = 0usize;
        while idx < deferred.len() {
            if deferred[idx].event.is_complete() {
                let deferred_item = deferred.swap_remove(idx);
                let len = deferred_item.buffer.capacity();
                let bucket = buckets.entry(len).or_default();
                if bucket.len() < self.max_keep_per_size {
                    bucket.push(deferred_item.buffer);
                }
                self.puts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                idx += 1;
            }
        }
        Ok(())
    }

    /// Acquire a pinned buffer of the given size in bytes.
    pub fn get(&self, len: usize) -> VortexResult<PinnedByteBuffer> {
        self.reclaim_deferred()?;
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
        // Allocate outside the lock — cuMemAllocHost is an expensive syscall.
        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.allocs
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { PinnedByteBuffer::uninit_with_capacity(&self.ctx, key_len, len) }
    }

    /// Return a buffer to the pool.
    pub fn put(&self, buf: PinnedByteBuffer) -> VortexResult<()> {
        self.reclaim_deferred()?;
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
        Ok(())
    }

    /// Try to get a buffer from the pool without performing expensive allocation.
    ///
    /// Returns `Ok(None)` if no buffer is available in the pool for the requested size class.
    /// Unlike [`get`][Self::get], this will never call `cuMemAllocHost`.
    pub fn try_get(&self, len: usize) -> VortexResult<Option<PinnedByteBuffer>> {
        self.reclaim_deferred()?;
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

    /// Try to get a pooled pinned buffer without performing expensive allocation.
    ///
    /// Returns `Ok(None)` if no buffer is available in the pool.
    pub fn try_get_pooled(
        self: &Arc<Self>,
        len: usize,
    ) -> VortexResult<Option<PooledPinnedBuffer>> {
        match self.try_get(len)? {
            Some(inner) => Ok(Some(PooledPinnedBuffer {
                inner: Some(inner),
                pool: self.clone(),
            })),
            None => Ok(None),
        }
    }

    /// Get a pooled pinned buffer that will be returned to the pool on drop.
    pub fn get_pooled(self: &Arc<Self>, len: usize) -> VortexResult<PooledPinnedBuffer> {
        let inner = self.get(len)?;
        Ok(PooledPinnedBuffer {
            inner: Some(inner),
            pool: self.clone(),
        })
    }

    /// Defer returning a pinned buffer to the pool until the CUDA event completes.
    pub fn put_deferred(
        &self,
        event: Arc<CudaEvent>,
        buffer: PinnedByteBuffer,
    ) -> VortexResult<()> {
        let mut deferred = self.deferred.lock();
        deferred.push(DeferredPinnedBuffer { event, buffer });
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

    /// Reset pool reuse statistics.
    pub fn reset_stats(&self) {
        self.hits.store(0, std::sync::atomic::Ordering::Relaxed);
        self.misses.store(0, std::sync::atomic::Ordering::Relaxed);
        self.allocs.store(0, std::sync::atomic::Ordering::Relaxed);
        self.puts.store(0, std::sync::atomic::Ordering::Relaxed);
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
/// This wrapper owns a [`PinnedByteBuffer`] and ensures it gets returned to the
/// [`PinnedByteBufferPool`] when the buffer is no longer needed. This enables efficient
/// buffer reuse for I/O operations.
pub struct PooledPinnedBuffer {
    inner: Option<PinnedByteBuffer>,
    pool: Arc<PinnedByteBufferPool>,
}

#[allow(clippy::same_name_method)]
impl PooledPinnedBuffer {
    /// Create a new pooled buffer.
    pub fn new(inner: PinnedByteBuffer, pool: Arc<PinnedByteBufferPool>) -> Self {
        Self {
            inner: Some(inner),
            pool,
        }
    }

    /// Returns the length of the buffer in bytes.
    pub fn len(&self) -> usize {
        self.inner
            .as_ref()
            .map(|b| b.len())
            .unwrap_or_else(|| vortex_panic!("buffer already consumed"))
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Convert this pooled buffer into a [`ByteBuffer`].
    ///
    /// The returned buffer will return the underlying pinned memory to the pool when dropped.
    /// This enables zero-copy conversion to the standard Vortex buffer type while maintaining
    /// pool-based memory reuse.
    pub fn into_byte_buffer(mut self) -> ByteBuffer {
        let inner = self
            .inner
            .take()
            .unwrap_or_else(|| vortex_panic!("buffer already consumed"));
        let len = inner.len();
        let pool = self.pool.clone();

        // Create a wrapper that will return the buffer to the pool on drop
        let wrapper = PooledPinnedBufferOwner::new(inner, pool);

        // Use Bytes::from_owner to create a Bytes that owns the wrapper
        let bytes = Bytes::from_owner(wrapper);

        // The ByteBuffer should have the full length
        assert_eq!(bytes.len(), len);

        ByteBuffer::from(bytes)
    }

    pub fn into_inner(mut self) -> (PinnedByteBuffer, Arc<PinnedByteBufferPool>) {
        let inner = self
            .inner
            .take()
            .unwrap_or_else(|| vortex_panic!("buffer already consumed"));
        (inner, self.pool.clone())
    }
}

#[allow(clippy::same_name_method)]
impl HostSlice<u8> for PooledPinnedBuffer {
    fn len(&self) -> usize {
        self.len()
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [u8], SyncOnDrop<'a>) {
        let inner = self
            .inner
            .as_ref()
            .unwrap_or_else(|| vortex_panic!("buffer already consumed"));
        let (slice, sync) = unsafe { HostSlice::stream_synced_slice(inner, stream) };
        (&slice[..inner.len()], sync)
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [u8], SyncOnDrop<'a>) {
        let inner = self
            .inner
            .as_mut()
            .unwrap_or_else(|| vortex_panic!("buffer already consumed"));
        let len = inner.len();
        let (slice, sync) = unsafe { HostSlice::stream_synced_mut_slice(inner, stream) };
        (&mut slice[..len], sync)
    }
}

impl Drop for PooledPinnedBuffer {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            // Return the buffer to the pool, ignoring errors
            drop(self.pool.put(inner));
        }
    }
}

/// Internal wrapper that owns a PinnedByteBuffer and returns it to the pool on drop.
///
/// This is used by `Bytes::from_owner` to manage the lifecycle of pooled pinned buffers.
struct PooledPinnedBufferOwner {
    // We use Option so we can take the buffer out in Drop
    inner: Option<PinnedByteBuffer>,
    // Cached pointer and length for AsRef implementation
    ptr: *const u8,
    len: usize,
    pool: Arc<PinnedByteBufferPool>,
}

// SAFETY: The pinned buffer is allocated by CUDA and is safe to send across threads.
// The pointer is derived from the buffer and remains valid as long as the buffer exists.
unsafe impl Send for PooledPinnedBufferOwner {}
unsafe impl Sync for PooledPinnedBufferOwner {}

impl PooledPinnedBufferOwner {
    fn new(inner: PinnedByteBuffer, pool: Arc<PinnedByteBufferPool>) -> Self {
        let ptr = inner
            .as_ptr()
            .unwrap_or_else(|e| vortex_panic!("failed to get pointer to pinned buffer: {e}"));
        let len = inner.len();
        Self {
            inner: Some(inner),
            ptr,
            len,
            pool,
        }
    }
}

impl AsRef<[u8]> for PooledPinnedBufferOwner {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: The pointer and length were captured when the buffer was created
        // and remain valid as long as this struct exists (buffer is in the Mutex).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for PooledPinnedBufferOwner {
    fn drop(&mut self) {
        // Take the buffer out and return it to the pool
        if let Some(buffer) = self.inner.take() {
            drop(self.pool.put(buffer));
        }
    }
}
