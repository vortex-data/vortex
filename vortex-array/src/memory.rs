// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped memory allocation for host-side buffers.

use std::cell::RefCell;
use std::fmt::Debug;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_metrics::Counter;
use vortex_metrics::DefaultMetricsRegistry;
use vortex_metrics::Gauge;
use vortex_metrics::Histogram;
use vortex_metrics::Label;
use vortex_metrics::Metric;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;
use vortex_session::Ref;
use vortex_session::RefMut;
use vortex_session::SessionExt;
use vortex_utils::aliases::hash_map::HashMap;

const DEFAULT_MAX_BYTES_PER_THREAD: usize = 64 * 1024 * 1024;
const FALLBACK_POOL_ALIGNMENT_BYTES: usize = 4 * 1024;

// (bucket_size_bytes, max_entries_per_thread)
const POOL_BUCKETS: &[(usize, usize)] = &[
    (4 * 1024, 256),
    (16 * 1024, 128),
    (64 * 1024, 64),
    (128 * 1024, 32),
    (256 * 1024, 16),
    (512 * 1024, 8),
    (1024 * 1024, 8),
    (2 * 1024 * 1024, 4),
    (4 * 1024 * 1024, 2),
];

static NEXT_POOLED_ALLOCATOR_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static POOLED_HOST_ALLOCATOR_POOLS: RefCell<HashMap<u64, ThreadLocalAllocatorPool>> =
        RefCell::new(HashMap::new());
}

fn default_pooled_metrics_registry() -> DefaultMetricsRegistry {
    static REGISTRY: OnceLock<DefaultMetricsRegistry> = OnceLock::new();
    REGISTRY
        .get_or_init(DefaultMetricsRegistry::default)
        .clone()
}

/// Returns a snapshot of metrics recorded by default-constructed pooled host allocators.
pub fn default_pooled_allocator_metrics_snapshot() -> Vec<Metric> {
    default_pooled_metrics_registry().snapshot()
}

/// Mutable host buffer contract used by [`WritableHostBuffer`].
pub trait HostBufferMut: Send + 'static {
    /// Returns the logical byte length of the buffer.
    fn len(&self) -> usize;

    /// Whether the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alignment of the buffer.
    fn alignment(&self) -> Alignment;

    /// Returns mutable access to the writable byte range.
    fn as_mut_slice(&mut self) -> &mut [u8];

    /// Freeze the buffer into an immutable [`ByteBuffer`].
    fn freeze(self: Box<Self>) -> ByteBuffer;
}

/// Exact-size writable host buffer returned by a [`HostAllocator`].
pub struct WritableHostBuffer {
    inner: Box<dyn HostBufferMut>,
}

impl WritableHostBuffer {
    /// Create a writable host buffer from an implementation of [`HostBufferMut`].
    pub fn new(inner: Box<dyn HostBufferMut>) -> Self {
        Self { inner }
    }

    /// Returns the logical byte length of the buffer.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true when the buffer has zero bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alignment of the buffer.
    pub fn alignment(&self) -> Alignment {
        self.inner.alignment()
    }

    /// Returns mutable access to the writable byte range.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.inner.as_mut_slice()
    }

    /// Returns mutable access to the buffer as a typed slice.
    pub fn as_mut_slice_typed<T>(&mut self) -> VortexResult<&mut [T]> {
        vortex_ensure!(
            size_of::<T>() != 0,
            InvalidArgument: "Cannot create typed mutable slice for zero-sized type {}",
            std::any::type_name::<T>()
        );
        vortex_ensure!(
            self.alignment().is_aligned_to(Alignment::of::<T>()),
            InvalidArgument: "Buffer is not sufficiently aligned for type {}",
            std::any::type_name::<T>()
        );

        let bytes = self.as_mut_slice();
        let byte_len = bytes.len();
        let ptr = bytes.as_mut_ptr();
        let type_size = size_of::<T>();

        vortex_ensure!(
            byte_len.is_multiple_of(type_size),
            InvalidArgument: "Buffer length {byte_len} is not a multiple of {} for {}",
            type_size,
            std::any::type_name::<T>()
        );

        // SAFETY: We checked size divisibility and pointer alignment for `T`,
        // and we have exclusive mutable access to the underlying bytes.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr.cast::<T>(), byte_len / type_size) })
    }

    /// Freeze the writable buffer into an immutable [`ByteBuffer`].
    pub fn freeze(self) -> ByteBuffer {
        self.inner.freeze()
    }

    /// Freeze the writable buffer into a typed immutable [`Buffer<T>`].
    pub fn freeze_typed<T>(self) -> VortexResult<Buffer<T>> {
        vortex_ensure!(
            size_of::<T>() != 0,
            InvalidArgument: "Cannot freeze typed buffer for zero-sized type {}",
            std::any::type_name::<T>()
        );

        let buffer = self.freeze();
        let byte_len = buffer.len();
        let type_size = size_of::<T>();
        let type_align = Alignment::of::<T>();

        vortex_ensure!(
            byte_len.is_multiple_of(type_size),
            InvalidArgument: "Buffer length {byte_len} is not a multiple of {} for {}",
            type_size,
            std::any::type_name::<T>()
        );
        vortex_ensure!(
            buffer.is_aligned(type_align),
            InvalidArgument: "Buffer pointer is not aligned to {} for {}",
            type_align,
            std::any::type_name::<T>()
        );

        Ok(Buffer::from_byte_buffer(buffer))
    }
}

impl Debug for WritableHostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WritableHostBuffer")
            .field("len", &self.len())
            .field("alignment", &self.alignment())
            .finish()
    }
}

/// Allocator for exact-size writable host buffers.
pub trait HostAllocator: Debug + Send + Sync + 'static {
    /// Allocate a writable host buffer with the requested byte length and alignment.
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer>;
}

/// Shared allocator reference used throughout session-scoped memory APIs.
pub type HostAllocatorRef = Arc<dyn HostAllocator>;

/// Extension methods for [`HostAllocator`]s.
pub trait HostAllocatorExt: HostAllocator {
    /// Allocate host memory for `len` elements of `T` using `Alignment::of::<T>()`.
    fn allocate_typed<T>(&self, len: usize) -> VortexResult<WritableHostBuffer> {
        let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            vortex_err!(
                "Typed host allocation overflow for type {} and len {}",
                std::any::type_name::<T>(),
                len
            )
        })?;
        self.allocate(bytes, Alignment::of::<T>())
    }
}

impl<A: HostAllocator + ?Sized> HostAllocatorExt for A {}

/// Session-scoped memory configuration for Vortex arrays.
#[derive(Debug)]
pub struct MemorySession {
    allocator: HostAllocatorRef,
}

impl MemorySession {
    /// Creates a new session memory configuration using the provided allocator.
    pub fn new(allocator: HostAllocatorRef) -> Self {
        Self { allocator }
    }

    /// Returns the configured allocator.
    pub fn allocator(&self) -> HostAllocatorRef {
        Arc::clone(&self.allocator)
    }

    /// Updates the configured allocator.
    pub fn set_allocator(&mut self, allocator: HostAllocatorRef) {
        self.allocator = allocator;
    }
}

impl Default for MemorySession {
    fn default() -> Self {
        Self::new(Arc::new(PooledHostAllocator::default()))
    }
}

/// Extension trait for accessing session-scoped memory configuration.
pub trait MemorySessionExt: SessionExt {
    /// Returns the memory session for this execution/session context.
    fn memory(&self) -> Ref<'_, MemorySession> {
        self.get::<MemorySession>()
    }

    /// Returns the configured host allocator for this execution/session context.
    fn allocator(&self) -> HostAllocatorRef {
        self.memory().allocator()
    }

    /// Returns mutable access to the memory session.
    fn memory_mut(&self) -> RefMut<'_, MemorySession> {
        self.get_mut::<MemorySession>()
    }
}

impl<S: SessionExt> MemorySessionExt for S {}

/// Default host allocator.
#[derive(Debug, Default)]
pub struct DefaultHostAllocator;

impl HostAllocator for DefaultHostAllocator {
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer> {
        allocate_unpooled(len, alignment)
    }
}

/// A pooled host allocator with thread-local buckets and metric instrumentation.
#[derive(Debug)]
pub struct PooledHostAllocator {
    id: u64,
    max_bytes_per_thread: usize,
    metrics: Arc<PooledAllocatorMetrics>,
}

impl PooledHostAllocator {
    /// Create a pooled allocator.
    ///
    /// `max_bytes_per_thread` controls the maximum total capacity retained in the thread-local
    /// pool for this allocator. Set it to `0` to disable pooling while still recording metrics.
    pub fn new(max_bytes_per_thread: usize, metrics_registry: Arc<dyn MetricsRegistry>) -> Self {
        let id = NEXT_POOLED_ALLOCATOR_ID.fetch_add(1, Ordering::Relaxed);
        let labels = vec![
            Label::new("allocator", "pooled_host"),
            Label::new("allocator_id", id.to_string()),
        ];

        Self {
            id,
            max_bytes_per_thread,
            metrics: Arc::new(PooledAllocatorMetrics::new(
                metrics_registry.as_ref(),
                labels,
            )),
        }
    }

    /// Maximum retained bytes per thread for this allocator.
    pub fn max_bytes_per_thread(&self) -> usize {
        self.max_bytes_per_thread
    }
}

impl Default for PooledHostAllocator {
    fn default() -> Self {
        Self::new(
            DEFAULT_MAX_BYTES_PER_THREAD,
            Arc::new(default_pooled_metrics_registry()),
        )
    }
}

impl HostAllocator for PooledHostAllocator {
    fn allocate(
        &self,
        len: usize,
        requested_alignment: Alignment,
    ) -> VortexResult<WritableHostBuffer> {
        self.metrics.alloc_requests.add(1);
        self.metrics.request_bytes.update(len as f64);

        if self.max_bytes_per_thread == 0 {
            self.metrics.bypass_disabled.add(1);
            return allocate_unpooled(len, requested_alignment);
        }

        let pool_alignment = pooled_alignment();
        if !pool_alignment.is_aligned_to(requested_alignment) {
            self.metrics.bypass_alignment.add(1);
            return allocate_unpooled(len, requested_alignment);
        }

        let Some(bucket_idx) = bucket_index_for_len(len) else {
            self.metrics.bypass_size.add(1);
            return allocate_unpooled(len, requested_alignment);
        };

        let (bucket_size, _) = POOL_BUCKETS[bucket_idx];
        if bucket_size > self.max_bytes_per_thread {
            self.metrics.bypass_size.add(1);
            return allocate_unpooled(len, requested_alignment);
        }

        let (pooled, retained_bytes, retained_buffers) = with_allocator_pool(self.id, |pool| {
            let pooled = pool.take_buffer(bucket_idx);
            (pooled, pool.retained_bytes, pool.buffer_count())
        });

        self.metrics.retained_bytes.set(retained_bytes as f64);
        self.metrics.retained_buffers.set(retained_buffers as f64);
        self.metrics.bucket_bytes.update(bucket_size as f64);

        let mut buffer = if let Some(buffer) = pooled {
            self.metrics.hits.add(1);
            buffer
        } else {
            self.metrics.misses.add(1);
            ByteBufferMut::with_capacity_aligned(bucket_size, pool_alignment)
        };

        // SAFETY: We fully initialize this slice before freezing it.
        unsafe { buffer.set_len(len) };

        Ok(WritableHostBuffer::new(Box::new(
            DefaultWritableHostBuffer::with_pool(
                buffer,
                requested_alignment,
                PooledReturn {
                    allocator_id: self.id,
                    bucket_idx,
                    max_bytes_per_thread: self.max_bytes_per_thread,
                    metrics: Arc::clone(&self.metrics),
                },
            ),
        )))
    }
}

fn allocate_unpooled(len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer> {
    let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
    // SAFETY: We fully initialize this slice before freezing it.
    unsafe { buffer.set_len(len) };
    Ok(WritableHostBuffer::new(Box::new(
        DefaultWritableHostBuffer::new(buffer, alignment),
    )))
}

#[derive(Debug)]
struct PooledAllocatorMetrics {
    alloc_requests: Counter,
    hits: Counter,
    misses: Counter,
    puts: Counter,
    drops: Counter,
    bypass_alignment: Counter,
    bypass_size: Counter,
    bypass_disabled: Counter,
    request_bytes: Histogram,
    bucket_bytes: Histogram,
    retained_bytes: Gauge,
    retained_buffers: Gauge,
}

impl PooledAllocatorMetrics {
    fn new(metrics_registry: &dyn MetricsRegistry, labels: Vec<Label>) -> Self {
        Self {
            alloc_requests: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("memory.host_pool.alloc_requests"),
            hits: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("memory.host_pool.hits"),
            misses: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("memory.host_pool.misses"),
            puts: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("memory.host_pool.puts"),
            drops: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("memory.host_pool.drops"),
            bypass_alignment: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("memory.host_pool.bypass_alignment"),
            bypass_size: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("memory.host_pool.bypass_size"),
            bypass_disabled: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("memory.host_pool.bypass_disabled"),
            request_bytes: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .histogram("memory.host_pool.request_bytes"),
            bucket_bytes: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .histogram("memory.host_pool.bucket_bytes"),
            retained_bytes: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .gauge("memory.host_pool.retained_bytes"),
            retained_buffers: MetricBuilder::new(metrics_registry)
                .add_labels(labels)
                .gauge("memory.host_pool.retained_buffers"),
        }
    }
}

#[derive(Debug)]
struct PooledReturn {
    allocator_id: u64,
    bucket_idx: usize,
    max_bytes_per_thread: usize,
    metrics: Arc<PooledAllocatorMetrics>,
}

#[derive(Debug, Default)]
struct ThreadLocalAllocatorPool {
    retained_bytes: usize,
    buckets: Vec<Vec<ByteBufferMut>>,
}

impl ThreadLocalAllocatorPool {
    fn new() -> Self {
        Self {
            retained_bytes: 0,
            buckets: (0..POOL_BUCKETS.len()).map(|_| Vec::new()).collect(),
        }
    }

    fn take_buffer(&mut self, bucket_idx: usize) -> Option<ByteBufferMut> {
        let buffer = self.buckets.get_mut(bucket_idx)?.pop()?;
        self.retained_bytes = self.retained_bytes.saturating_sub(buffer.capacity());
        Some(buffer)
    }

    fn try_put_buffer(
        &mut self,
        bucket_idx: usize,
        mut buffer: ByteBufferMut,
        max_bytes_per_thread: usize,
    ) -> bool {
        if bucket_idx >= self.buckets.len() {
            return false;
        }

        let (_, max_entries) = POOL_BUCKETS[bucket_idx];
        if self.buckets[bucket_idx].len() >= max_entries {
            return false;
        }

        let capacity = buffer.capacity();
        if self.retained_bytes.saturating_add(capacity) > max_bytes_per_thread {
            return false;
        }

        buffer.clear();
        self.retained_bytes = self.retained_bytes.saturating_add(capacity);
        self.buckets[bucket_idx].push(buffer);
        true
    }

    fn buffer_count(&self) -> usize {
        self.buckets.iter().map(Vec::len).sum()
    }

    #[cfg(test)]
    fn bucket_len(&self, bucket_idx: usize) -> usize {
        self.buckets[bucket_idx].len()
    }
}

fn with_allocator_pool<R>(
    allocator_id: u64,
    f: impl FnOnce(&mut ThreadLocalAllocatorPool) -> R,
) -> R {
    POOLED_HOST_ALLOCATOR_POOLS.with(|pools| {
        let mut pools = pools.borrow_mut();
        let pool = pools
            .entry(allocator_id)
            .or_insert_with(ThreadLocalAllocatorPool::new);
        f(pool)
    })
}

fn return_buffer_to_pool(buffer: ByteBufferMut, pooled: PooledReturn) {
    let (stored, retained_bytes, retained_buffers) =
        with_allocator_pool(pooled.allocator_id, |pool| {
            let stored =
                pool.try_put_buffer(pooled.bucket_idx, buffer, pooled.max_bytes_per_thread);
            (stored, pool.retained_bytes, pool.buffer_count())
        });

    pooled.metrics.retained_bytes.set(retained_bytes as f64);
    pooled.metrics.retained_buffers.set(retained_buffers as f64);

    if stored {
        pooled.metrics.puts.add(1);
    } else {
        pooled.metrics.drops.add(1);
    }
}

fn bucket_index_for_len(len: usize) -> Option<usize> {
    POOL_BUCKETS
        .iter()
        .position(|(bucket_size, _)| len <= *bucket_size)
}

fn pooled_alignment() -> Alignment {
    static CACHED: OnceLock<Alignment> = OnceLock::new();

    *CACHED.get_or_init(|| {
        let page_size = os_page_size().unwrap_or(FALLBACK_POOL_ALIGNMENT_BYTES);
        if page_size != 0 && page_size.is_power_of_two() && page_size <= (u16::MAX as usize) {
            Alignment::new(page_size)
        } else {
            Alignment::new(FALLBACK_POOL_ALIGNMENT_BYTES)
        }
    })
}

#[cfg(unix)]
fn os_page_size() -> Option<usize> {
    // SAFETY: `sysconf` is thread-safe for `_SC_PAGESIZE` and has no aliasing requirements.
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if value <= 0 {
        return None;
    }

    usize::try_from(value).ok()
}

#[cfg(not(unix))]
fn os_page_size() -> Option<usize> {
    None
}

#[derive(Debug)]
struct DefaultWritableHostBuffer {
    buffer: Option<ByteBufferMut>,
    alignment: Alignment,
    pooled: Option<PooledReturn>,
}

impl DefaultWritableHostBuffer {
    fn new(buffer: ByteBufferMut, alignment: Alignment) -> Self {
        Self {
            buffer: Some(buffer),
            alignment,
            pooled: None,
        }
    }

    fn with_pool(buffer: ByteBufferMut, alignment: Alignment, pooled: PooledReturn) -> Self {
        Self {
            buffer: Some(buffer),
            alignment,
            pooled: Some(pooled),
        }
    }

    fn take_parts(&mut self) -> (ByteBufferMut, Option<PooledReturn>) {
        (
            self.buffer
                .take()
                .vortex_expect("buffer must exist until freeze/drop"),
            self.pooled.take(),
        )
    }
}

impl Drop for DefaultWritableHostBuffer {
    fn drop(&mut self) {
        let Some(pooled) = self.pooled.take() else {
            return;
        };

        let Some(buffer) = self.buffer.take() else {
            return;
        };

        return_buffer_to_pool(buffer, pooled);
    }
}

#[derive(Debug)]
struct HostBufferOwner {
    buffer: Option<ByteBufferMut>,
    pooled: Option<PooledReturn>,
}

impl AsRef<[u8]> for HostBufferOwner {
    fn as_ref(&self) -> &[u8] {
        self.buffer
            .as_ref()
            .vortex_expect("buffer must exist while bytes owner is alive")
            .as_slice()
    }
}

impl Drop for HostBufferOwner {
    fn drop(&mut self) {
        let Some(pooled) = self.pooled.take() else {
            return;
        };

        let Some(buffer) = self.buffer.take() else {
            return;
        };

        return_buffer_to_pool(buffer, pooled);
    }
}

impl HostBufferMut for DefaultWritableHostBuffer {
    fn len(&self) -> usize {
        self.buffer
            .as_ref()
            .vortex_expect("buffer must exist until freeze/drop")
            .len()
    }

    fn alignment(&self) -> Alignment {
        self.alignment
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer
            .as_mut()
            .vortex_expect("buffer must exist until freeze/drop")
            .as_mut_slice()
    }

    fn freeze(mut self: Box<Self>) -> ByteBuffer {
        let alignment = self.alignment;
        let (buffer, pooled) = self.take_parts();
        let bytes = Bytes::from_owner(HostBufferOwner {
            buffer: Some(buffer),
            pooled,
        });
        ByteBuffer::from_bytes_aligned(bytes, alignment)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use super::*;

    #[derive(Debug)]
    struct CountingAllocator {
        allocations: Arc<AtomicUsize>,
    }

    impl HostAllocator for CountingAllocator {
        fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer> {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            DefaultHostAllocator.allocate(len, alignment)
        }
    }

    fn clear_allocator_pool(allocator_id: u64) {
        POOLED_HOST_ALLOCATOR_POOLS.with(|pools| {
            pools.borrow_mut().remove(&allocator_id);
        });
    }

    fn allocator_pool_bucket_len(allocator_id: u64, bucket_idx: usize) -> usize {
        with_allocator_pool(allocator_id, |pool| pool.bucket_len(bucket_idx))
    }

    #[test]
    fn writable_host_buffer_freeze_round_trip() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate(16, Alignment::new(8)).unwrap();
        for (idx, byte) in writable.as_mut_slice().iter_mut().enumerate() {
            *byte = u8::try_from(idx).unwrap();
        }

        let host = writable.freeze();
        assert_eq!(host.len(), 16);
        assert!(host.is_aligned(Alignment::new(8)));
        assert_eq!(host.as_slice(), (0u8..16).collect::<Vec<_>>().as_slice());
    }

    #[test]
    fn memory_session_replaces_allocator() {
        let allocations = Arc::new(AtomicUsize::new(0));
        let allocator = Arc::new(CountingAllocator {
            allocations: Arc::clone(&allocations),
        });
        let mut session = MemorySession::default();
        session.set_allocator(allocator);
        drop(session.allocator().allocate(4, Alignment::none()).unwrap());
        assert_eq!(allocations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn typed_allocation_uses_type_alignment() {
        let allocator = DefaultHostAllocator;
        let writable = allocator.allocate_typed::<u64>(4).unwrap();
        assert_eq!(writable.len(), 4 * size_of::<u64>());
        assert_eq!(writable.alignment(), Alignment::of::<u64>());
    }

    #[test]
    fn typed_mut_slice_round_trip() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate_typed::<u64>(4).unwrap();
        writable
            .as_mut_slice_typed::<u64>()
            .unwrap()
            .copy_from_slice(&[10, 20, 30, 40]);

        let frozen = writable.freeze();
        let values = unsafe {
            std::slice::from_raw_parts(
                frozen.as_slice().as_ptr().cast::<u64>(),
                frozen.len() / size_of::<u64>(),
            )
        };
        assert_eq!(values, [10, 20, 30, 40]);
    }

    #[test]
    fn typed_mut_slice_rejects_length_mismatch() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate(7, Alignment::none()).unwrap();
        assert!(writable.as_mut_slice_typed::<u32>().is_err());
    }

    #[test]
    fn freeze_typed_round_trip() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate_typed::<u64>(4).unwrap();
        writable
            .as_mut_slice_typed::<u64>()
            .unwrap()
            .copy_from_slice(&[1, 3, 5, 7]);

        let frozen = writable.freeze_typed::<u64>().unwrap();
        assert_eq!(frozen.as_slice(), [1, 3, 5, 7]);
    }

    #[test]
    fn freeze_typed_rejects_length_mismatch() {
        let allocator = DefaultHostAllocator;
        let writable = allocator.allocate(7, Alignment::none()).unwrap();
        let err = writable.freeze_typed::<u32>().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not a multiple of"));
    }

    #[test]
    fn pooled_allocator_reuses_bucket() {
        let allocator =
            PooledHostAllocator::new(8 * 1024 * 1024, Arc::new(DefaultMetricsRegistry::default()));
        clear_allocator_pool(allocator.id);

        let bucket_idx = bucket_index_for_len(100).unwrap();
        {
            let writable = allocator.allocate(100, Alignment::new(8)).unwrap();
            drop(writable);
        }

        assert_eq!(allocator_pool_bucket_len(allocator.id, bucket_idx), 1);

        {
            let writable = allocator.allocate(100, Alignment::new(8)).unwrap();
            // Reuse pops from pool.
            assert_eq!(allocator_pool_bucket_len(allocator.id, bucket_idx), 0);
            drop(writable);
        }

        assert_eq!(allocator_pool_bucket_len(allocator.id, bucket_idx), 1);
    }

    #[test]
    fn pooled_allocator_bypasses_large_requests() {
        let allocator =
            PooledHostAllocator::new(8 * 1024 * 1024, Arc::new(DefaultMetricsRegistry::default()));
        clear_allocator_pool(allocator.id);

        let too_large = POOL_BUCKETS.last().unwrap().0 + 1;
        let writable = allocator.allocate(too_large, Alignment::new(8)).unwrap();
        drop(writable);

        let pooled_count = with_allocator_pool(allocator.id, |pool| pool.buffer_count());
        assert_eq!(pooled_count, 0);
    }
}
