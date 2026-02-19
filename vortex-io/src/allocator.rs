// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_utils::aliases::hash_map::HashMap;

use crate::WriteTarget;

/// Allocates buffers for I/O reads.
pub trait BufferAllocator: Send + Sync + 'static {
    /// Allocate a buffer for the requested length and alignment.
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>>;

    /// Try to allocate without potentially blocking on expensive operations.
    ///
    /// Returns `Ok(None)` if the allocation would require a blocking operation
    /// (e.g., CUDA pinned memory allocation via `cuMemAllocHost`). In that case,
    /// the caller may choose to pipeline the allocation with I/O.
    ///
    /// The default implementation always succeeds by delegating to [`allocate`][Self::allocate].
    fn try_allocate(
        &self,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<Option<Box<dyn WriteTarget>>> {
        self.allocate(len, alignment).map(Some)
    }
}

/// The default allocator that uses `ByteBufferMut`.
pub struct DefaultAllocator;

/// Allocation counters for the default allocator.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultAllocStats {
    pub count: u64,
    pub bytes: u64,
}

static DEFAULT_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static DEFAULT_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

pub fn default_alloc_stats() -> DefaultAllocStats {
    DefaultAllocStats {
        count: DEFAULT_ALLOC_COUNT.load(Ordering::Relaxed),
        bytes: DEFAULT_ALLOC_BYTES.load(Ordering::Relaxed),
    }
}

pub fn reset_default_alloc_stats() {
    DEFAULT_ALLOC_COUNT.store(0, Ordering::Relaxed);
    DEFAULT_ALLOC_BYTES.store(0, Ordering::Relaxed);
}

impl BufferAllocator for DefaultAllocator {
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>> {
        DEFAULT_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        DEFAULT_ALLOC_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
        unsafe { buffer.set_len(len) };
        Ok(Box::new(buffer))
    }
}

/// Allocation counters for the pooled host allocator.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostPoolStats {
    pub hits: u64,
    pub misses: u64,
    pub allocs: u64,
    pub puts: u64,
}

/// A simple host buffer pool keyed by (length, alignment).
pub struct HostByteBufferPool {
    max_keep_per_key: usize,
    buckets: Mutex<HashMap<(usize, Alignment), Vec<ByteBufferMut>>>,
    fixed_alignment: Option<Alignment>,
    round_len_pow2: bool,
    hits: AtomicU64,
    misses: AtomicU64,
    allocs: AtomicU64,
    puts: AtomicU64,
}

impl HostByteBufferPool {
    /// Create a new pool with default limits.
    pub fn new() -> Self {
        Self::with_limits(4)
    }

    /// Create a new pool with a maximum number of cached buffers per key.
    pub fn with_limits(max_keep_per_key: usize) -> Self {
        Self {
            max_keep_per_key: max_keep_per_key.max(1),
            buckets: Mutex::new(HashMap::new()),
            fixed_alignment: None,
            round_len_pow2: false,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            allocs: AtomicU64::new(0),
            puts: AtomicU64::new(0),
        }
    }

    /// Create a new pool with a fixed alignment and maximum buffers per key.
    pub fn with_fixed_alignment(alignment: Alignment, max_keep_per_key: usize) -> Self {
        let mut pool = Self::with_limits(max_keep_per_key);
        pool.fixed_alignment = Some(alignment);
        pool
    }

    /// Create a new pool with fixed alignment and power-of-two length bucketing.
    pub fn with_fixed_alignment_pow2(alignment: Alignment, max_keep_per_key: usize) -> Self {
        let mut pool = Self::with_limits(max_keep_per_key);
        pool.fixed_alignment = Some(alignment);
        pool.round_len_pow2 = true;
        pool
    }

    fn size_class_len(&self, len: usize) -> usize {
        if !self.round_len_pow2 {
            return len;
        }
        if len == 0 { 0 } else { len.next_power_of_two() }
    }

    fn get(&self, len: usize, alignment: Alignment) -> VortexResult<ByteBufferMut> {
        let key_len = self.size_class_len(len);
        let key_alignment = match self.fixed_alignment {
            Some(fixed) if fixed.is_aligned_to(alignment) => fixed,
            Some(_) => alignment,
            None => alignment,
        };
        let mut buckets = self.buckets.lock();
        if let Some(bucket) = buckets.get_mut(&(key_len, key_alignment))
            && let Some(mut buf) = bucket.pop()
        {
            debug_assert!(buf.capacity() >= len);
            unsafe { buf.set_len(len) };
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(buf);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.allocs.fetch_add(1, Ordering::Relaxed);
        let mut buf = ByteBufferMut::with_capacity_aligned(key_len, key_alignment);
        unsafe { buf.set_len(len) };
        Ok(buf)
    }

    fn put(&self, buf: ByteBufferMut) -> VortexResult<()> {
        let len = self.size_class_len(buf.len());
        let alignment = match self.fixed_alignment {
            Some(fixed) if fixed.is_aligned_to(buf.alignment()) => fixed,
            Some(_) => buf.alignment(),
            None => buf.alignment(),
        };
        let mut buckets = self.buckets.lock();
        let bucket = buckets.entry((len, alignment)).or_default();
        if bucket.len() < self.max_keep_per_key {
            bucket.push(buf);
        }
        self.puts.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get a pooled host buffer that returns to the pool on drop.
    pub fn get_pooled(
        self: &Arc<Self>,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<PooledHostBuffer> {
        let inner = self.get(len, alignment)?;
        Ok(PooledHostBuffer {
            inner: Some(inner),
            pool: self.clone(),
        })
    }

    /// Snapshot pool reuse statistics.
    pub fn stats(&self) -> HostPoolStats {
        HostPoolStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            allocs: self.allocs.load(Ordering::Relaxed),
            puts: self.puts.load(Ordering::Relaxed),
        }
    }

    /// Reset pool reuse statistics.
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.allocs.store(0, Ordering::Relaxed);
        self.puts.store(0, Ordering::Relaxed);
    }
}

impl Default for HostByteBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// A pooled host buffer that returns to its pool on drop.
pub struct PooledHostBuffer {
    inner: Option<ByteBufferMut>,
    pool: Arc<HostByteBufferPool>,
}

impl PooledHostBuffer {
    fn inner_len(&self) -> usize {
        self.inner.as_ref().map(|b| b.len()).unwrap_or(0)
    }
}

impl WriteTarget for PooledHostBuffer {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.inner
            .as_mut()
            .unwrap_or_else(|| vortex_panic!("pooled buffer already consumed"))
            .as_mut_slice()
    }

    fn len(&self) -> usize {
        self.inner_len()
    }

    fn into_handle(self: Box<Self>) -> VortexResult<BufferHandle> {
        let mut this = *self;
        let inner = this
            .inner
            .take()
            .ok_or_else(|| vortex_err!("pooled buffer already consumed"))?;
        let len = inner.len();
        let pool = this.pool.clone();

        let owner = PooledHostBufferOwner::new(inner, len, pool);
        let bytes = Bytes::from_owner(owner);
        Ok(BufferHandle::new_host(bytes.into()))
    }
}

struct PooledHostBufferOwner {
    inner: Option<ByteBufferMut>,
    ptr: *const u8,
    len: usize,
    pool: Arc<HostByteBufferPool>,
}

// SAFETY: host buffers are safe to send across threads.
unsafe impl Send for PooledHostBufferOwner {}
unsafe impl Sync for PooledHostBufferOwner {}

impl PooledHostBufferOwner {
    fn new(mut inner: ByteBufferMut, len: usize, pool: Arc<HostByteBufferPool>) -> Self {
        let ptr = inner.as_mut_slice().as_ptr();
        inner.as_mut_slice(); // keep length initialized
        Self {
            inner: Some(inner),
            ptr,
            len,
            pool,
        }
    }
}

impl AsRef<[u8]> for PooledHostBufferOwner {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for PooledHostBufferOwner {
    fn drop(&mut self) {
        if let Some(mut buffer) = self.inner.take() {
            if buffer.len() != self.len {
                unsafe { buffer.set_len(self.len) };
            }
            drop(self.pool.put(buffer));
        }
    }
}

/// Allocator backed by a pooled host buffer pool.
pub struct PooledHostAllocator {
    pool: Arc<HostByteBufferPool>,
}

impl PooledHostAllocator {
    pub fn new(pool: Arc<HostByteBufferPool>) -> Self {
        Self { pool }
    }
}

impl BufferAllocator for PooledHostAllocator {
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>> {
        let buffer = self.pool.get_pooled(len, alignment)?;
        Ok(Box::new(buffer))
    }
}
