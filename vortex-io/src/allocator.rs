// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_utils::aliases::hash_map::HashMap;

use crate::WriteTarget;

/// Page size used as the minimum allocation size and fixed alignment for pooled buffers.
const PAGE_SIZE: usize = 4096;
const PAGE_ALIGNMENT: Alignment = Alignment::new(PAGE_SIZE);

pub trait BufferAllocator: Send + Sync + 'static {
    /// Allocate a buffer for the requested length and alignment.
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>>;
}

/// A simple allocator that uses `ByteBufferMut`.
pub struct DefaultAllocator;

impl BufferAllocator for DefaultAllocator {
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<Box<dyn WriteTarget>> {
        let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
        unsafe { buffer.set_len(len) };
        Ok(Box::new(buffer))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HostPoolStats {
    /// Number of allocation requests satisfied from the pool.
    pub hits: u64,
    /// Number of allocation requests that required a new allocation.
    pub misses: u64,
    /// Number of buffers returned to the pool.
    pub returns: u64,
}

/// A host buffer pool with power of two size classes and page alignment.
///
/// All buffers are page-aligned (4096 bytes). Requested lengths are rounded up to the next
/// power of two (minimum one page).
pub struct HostByteBufferPool {
    max_buffers_per_bucket: usize,
    buckets: Mutex<HashMap<usize, Vec<ByteBufferMut>>>,
    hits: AtomicU64,
    misses: AtomicU64,
    returns: AtomicU64,
}

impl HostByteBufferPool {
    /// Create a new pool with a maximum number of cached buffers per size class.
    pub fn with_max_buffers_per_bucket(max_buffers_per_bucket: usize) -> Self {
        Self {
            max_buffers_per_bucket: max_buffers_per_bucket.max(1),
            buckets: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            returns: AtomicU64::new(0),
        }
    }

    pub fn stats(&self) -> HostPoolStats {
        HostPoolStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            returns: self.returns.load(Ordering::Relaxed),
        }
    }

    /// Get a pooled buffer that returns to the pool on drop.
    pub fn get(self: &Arc<Self>, len: usize) -> VortexResult<PooledHostBuffer> {
        let size_class = len.next_power_of_two().max(PAGE_SIZE);
        let inner = self.get_inner(size_class, len)?;
        Ok(PooledHostBuffer {
            inner,
            size_class,
            pool: self.clone(),
        })
    }

    fn get_inner(&self, size_class: usize, len: usize) -> VortexResult<ByteBufferMut> {
        let mut buckets = self.buckets.lock();
        if let Some(bucket) = buckets.get_mut(&size_class)
            && let Some(mut buf) = bucket.pop()
        {
            // Has to hold, capacity is size_class, which is larger than len>
            assert!(buf.capacity() >= len);
            unsafe { buf.set_len(len) };
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(buf);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let mut buf = ByteBufferMut::with_capacity_aligned(size_class, PAGE_ALIGNMENT);
        assert!(buf.capacity() >= len);
        unsafe { buf.set_len(len) };
        Ok(buf)
    }

    fn put(&self, size_class: usize, buf: ByteBufferMut) {
        let mut buckets = self.buckets.lock();
        let bucket = buckets.entry(size_class).or_default();
        if bucket.len() < self.max_buffers_per_bucket {
            bucket.push(buf);
        }
        self.returns.fetch_add(1, Ordering::Relaxed);
    }
}

impl Default for HostByteBufferPool {
    fn default() -> Self {
        Self::with_max_buffers_per_bucket(4)
    }
}

/// A pooled host buffer that returns to its pool on drop.
pub struct PooledHostBuffer {
    inner: ByteBufferMut,
    size_class: usize,
    pool: Arc<HostByteBufferPool>,
}

impl WriteTarget for PooledHostBuffer {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.inner.as_mut_slice()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn into_handle(self: Box<Self>) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        async move {
            let this = *self;
            let owner = PooledHostBufferHandle {
                inner: Some(this.inner),
                size_class: this.size_class,
                pool: this.pool,
            };
            let bytes = Bytes::from_owner(owner);
            Ok(BufferHandle::new_host(bytes.into()))
        }
        .boxed()
    }
}

// Read only view of a pooled host buffer, returns the underlying
// buffer into the pool on drop.
struct PooledHostBufferHandle {
    inner: Option<ByteBufferMut>,
    size_class: usize,
    pool: Arc<HostByteBufferPool>,
}

impl AsRef<[u8]> for PooledHostBufferHandle {
    fn as_ref(&self) -> &[u8] {
        self.inner
            .as_ref()
            .unwrap_or_else(|| vortex_panic!("pooled buffer already consumed"))
            .as_slice()
    }
}

impl Drop for PooledHostBufferHandle {
    fn drop(&mut self) {
        if let Some(mut buffer) = self.inner.take() {
            // Restore to size-class capacity so the pool bucket key matches.
            unsafe { buffer.set_len(self.size_class) };
            self.pool.put(self.size_class, buffer);
        }
    }
}

/// Allocator backed by a [`HostByteBufferPool`].
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
        assert!(
            PAGE_ALIGNMENT.is_aligned_to(alignment),
            "PooledHostAllocator uses page alignment ({PAGE_SIZE}), \
             which does not satisfy requested alignment {alignment}"
        );
        let buffer = self.pool.get(len)?;
        Ok(Box::new(buffer))
    }
}
