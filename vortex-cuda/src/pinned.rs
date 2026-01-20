// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use bytes::Bytes;
use cudarc::driver::CudaContext;
use cudarc::driver::PinnedHostSlice;
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
}

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
        Ok(Self { inner })
    }

    /// Returns the length of the buffer in bytes.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the buffer as an immutable slice.
    pub fn as_slice(&self) -> VortexResult<&[u8]> {
        self.inner
            .as_slice()
            .map_err(|e| vortex_err!("failed to access pinned host buffer: {e}"))
    }

    /// Returns the buffer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> VortexResult<&mut [u8]> {
        self.inner
            .as_mut_slice()
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

    /// Returns the CUDA context that owns this allocation.
    pub fn context(&self) -> &Arc<CudaContext> {
        self.inner.context()
    }
}

/// A simple pinned host buffer pool keyed by allocation size.
pub struct PinnedByteBufferPool {
    ctx: Arc<CudaContext>,
    max_keep_per_size: usize,
    buckets: Mutex<HashMap<usize, Vec<PinnedByteBuffer>>>,
}

impl PinnedByteBufferPool {
    /// Create a new pool with default limits.
    pub fn new(ctx: Arc<CudaContext>) -> Self {
        Self::with_limits(ctx, 4)
    }

    /// Create a new pool with a maximum number of cached buffers per size.
    pub fn with_limits(ctx: Arc<CudaContext>, max_keep_per_size: usize) -> Self {
        Self {
            ctx,
            max_keep_per_size: max_keep_per_size.max(1),
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Acquire a pinned buffer of the given size in bytes.
    pub fn get(&self, len: usize) -> VortexResult<PinnedByteBuffer> {
        let mut buckets = self.buckets.lock();
        if let Some(bucket) = buckets.get_mut(&len)
            && let Some(buf) = bucket.pop()
        {
            return Ok(buf);
        }
        unsafe { PinnedByteBuffer::uninit(&self.ctx, len) }
    }

    /// Return a buffer to the pool.
    pub fn put(&self, buf: PinnedByteBuffer) -> VortexResult<()> {
        let len = buf.len();
        let mut buckets = self.buckets.lock();
        let bucket = buckets.entry(len).or_default();
        if bucket.len() < self.max_keep_per_size {
            bucket.push(buf);
        }
        Ok(())
    }

    /// Get a pooled pinned buffer that will be returned to the pool on drop.
    pub fn get_pooled(self: &Arc<Self>, len: usize) -> VortexResult<PooledPinnedBuffer> {
        let inner = self.get(len)?;
        Ok(PooledPinnedBuffer {
            inner: Some(inner),
            pool: self.clone(),
        })
    }
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
        self.inner.as_ref().map(|b| b.len()).unwrap_or(0)
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
    inner: Mutex<Option<PinnedByteBuffer>>,
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
            inner: Mutex::new(Some(inner)),
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
        if let Some(buffer) = self.inner.lock().take() {
            drop(self.pool.put(buffer));
        }
    }
}
