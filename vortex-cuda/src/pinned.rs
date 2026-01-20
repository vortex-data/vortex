// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::PinnedHostSlice;
use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
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
}
