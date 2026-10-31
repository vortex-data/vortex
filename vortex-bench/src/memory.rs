// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Memory measurement and reclamation utilities for benchmarks

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::AtomicIsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use parking_lot::Mutex;
use sysinfo::ProcessRefreshKind;
use sysinfo::ProcessesToUpdate;
use sysinfo::RefreshKind;
use sysinfo::System;

const PEAK_SAMPLE_INTERVAL: Duration = Duration::from_millis(5);

const HEAP_STRIPES: usize = 64;

#[repr(align(128))]
struct HeapStripe(AtomicIsize);

/// Live heap bytes, striped so allocating threads rarely share a cache line.
static LIVE_HEAP: [HeapStripe; HEAP_STRIPES] =
    [const { HeapStripe(AtomicIsize::new(0)) }; HEAP_STRIPES];

/// A [`GlobalAlloc`] wrapper that counts live heap bytes, unlike RSS, which an allocator that
/// never purges only ever grows.
pub struct CountingAlloc<A>(pub A);

#[inline]
fn heap_stripe() -> &'static AtomicIsize {
    // Thread stacks are megabytes apart, so a stack address picks a stripe per thread without TLS,
    // which an allocator cannot safely use.
    let marker = 0u8;
    let index = (&raw const marker as usize >> 20) % HEAP_STRIPES;
    &LIVE_HEAP[index].0
}

#[allow(clippy::cast_possible_wrap)]
// SAFETY: forwards every call to the wrapped allocator unchanged, only counting sizes.
unsafe impl<A: GlobalAlloc> GlobalAlloc for CountingAlloc<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `alloc`'s contract.
        let ptr = unsafe { self.0.alloc(layout) };
        if !ptr.is_null() {
            heap_stripe().fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `alloc_zeroed`'s contract.
        let ptr = unsafe { self.0.alloc_zeroed(layout) };
        if !ptr.is_null() {
            heap_stripe().fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller upholds `dealloc`'s contract.
        unsafe { self.0.dealloc(ptr, layout) };
        heap_stripe().fetch_sub(layout.size() as isize, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller upholds `realloc`'s contract.
        let new_ptr = unsafe { self.0.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            heap_stripe().fetch_add(
                new_size as isize - layout.size() as isize,
                Ordering::Relaxed,
            );
        }
        new_ptr
    }
}

/// Live heap bytes allocated through [`CountingAlloc`], or zero without it.
pub fn live_heap_bytes() -> i64 {
    LIVE_HEAP
        .iter()
        .map(|stripe| stripe.0.load(Ordering::Relaxed) as i64)
        .sum()
}

/// Memory statistics for a process
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    /// Physical memory usage in bytes
    pub physical_memory: u64,
    /// Virtual memory usage in bytes  
    pub virtual_memory: u64,
}

impl MemoryStats {
    pub fn new(physical_memory: u64, virtual_memory: u64) -> Self {
        Self {
            physical_memory,
            virtual_memory,
        }
    }

    /// Calculate the difference between two memory measurements
    pub fn diff(&self, other: &MemoryStats) -> MemoryStatsDiff {
        MemoryStatsDiff {
            physical_memory_delta: self.physical_memory as i64 - other.physical_memory as i64,
            virtual_memory_delta: self.virtual_memory as i64 - other.virtual_memory as i64,
        }
    }
}

/// Memory usage difference between two measurements
#[derive(Debug, Clone, Copy)]
pub struct MemoryStatsDiff {
    /// Change in physical memory usage in bytes (can be negative)
    pub physical_memory_delta: i64,
    /// Change in virtual memory usage in bytes (can be negative)
    pub virtual_memory_delta: i64,
}

/// Thread-safe memory tracker using sysinfo
#[derive(Default)]
pub struct MemoryTracker {
    system: Mutex<System>,
    pid: u32,
    peak_physical_memory: Mutex<u64>,
    peak_virtual_memory: Mutex<u64>,
}

impl MemoryTracker {
    /// Create a new memory tracker for the current process
    pub fn new() -> Self {
        let mut system = System::new_with_specifics(
            RefreshKind::default().with_processes(ProcessRefreshKind::nothing().with_memory()),
        );
        let pid = std::process::id();

        // Initial refresh to populate process info
        system.refresh_processes(
            ProcessesToUpdate::Some(&[sysinfo::Pid::from_u32(pid)]),
            true,
        );

        // Get initial memory usage as baseline for peak tracking
        let initial_memory = if let Some(process) = system.process(sysinfo::Pid::from_u32(pid)) {
            MemoryStats::new(process.memory(), process.virtual_memory())
        } else {
            MemoryStats::new(0, 0)
        };

        Self {
            system: Mutex::new(system),
            pid,
            peak_physical_memory: Mutex::new(initial_memory.physical_memory),
            peak_virtual_memory: Mutex::new(initial_memory.virtual_memory),
        }
    }

    /// Get current memory usage for the tracked process and update peak tracking
    pub fn current_memory(&self) -> Option<MemoryStats> {
        let mut system = self.system.lock();
        system.refresh_processes(
            ProcessesToUpdate::Some(&[sysinfo::Pid::from_u32(self.pid)]),
            true,
        );

        if let Some(process) = system.process(sysinfo::Pid::from_u32(self.pid)) {
            let current_stats = MemoryStats::new(process.memory(), process.virtual_memory());

            // Update peak memory if current usage is higher
            {
                let mut peak_physical = self.peak_physical_memory.lock();
                if current_stats.physical_memory > *peak_physical {
                    *peak_physical = current_stats.physical_memory;
                }
            }

            {
                let mut peak_virtual = self.peak_virtual_memory.lock();
                if current_stats.virtual_memory > *peak_virtual {
                    *peak_virtual = current_stats.virtual_memory;
                }
            }

            Some(current_stats)
        } else {
            None
        }
    }

    /// Get the peak memory usage recorded so far
    pub fn peak_memory(&self) -> MemoryStats {
        let peak_physical = *self.peak_physical_memory.lock();
        let peak_virtual = *self.peak_virtual_memory.lock();
        MemoryStats::new(peak_physical, peak_virtual)
    }

    /// Reset peak memory tracking to current usage
    pub fn reset_peak(&self) {
        if let Some(current) = self.current_memory() {
            *self.peak_physical_memory.lock() = current.physical_memory;
            *self.peak_virtual_memory.lock() = current.virtual_memory;
        }
    }
}

/// Memory measurement guard that tracks memory usage before and after an operation
pub struct MemoryMeasurement {
    tracker: MemoryTracker,
    before: Option<MemoryStats>,
}

/// Comprehensive memory measurement result containing all memory metrics
#[derive(Debug, Clone)]
pub struct MemoryMeasurementResult {
    /// Memory change during the operation (can be negative)
    pub physical_memory_delta: i64,
    pub virtual_memory_delta: i64,
    /// Peak memory usage reached by the global tracker
    pub peak_physical_memory: u64,
    pub peak_virtual_memory: u64,
}

/// Simplified interface for memory tracking with a global tracker
pub struct BenchmarkMemoryTracker {
    global_tracker: Arc<MemoryTracker>,
    baseline_memory: Option<MemoryStats>,
    stop_sampler: Arc<AtomicBool>,
    baseline_heap: i64,
    peak_heap: Arc<AtomicI64>,
}

impl Default for BenchmarkMemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkMemoryTracker {
    /// Create a new benchmark memory tracker
    pub fn new() -> Self {
        let global_tracker = Arc::new(MemoryTracker::new());
        let stop_sampler = Arc::new(AtomicBool::new(false));
        let peak_heap = Arc::new(AtomicI64::new(live_heap_bytes()));
        // Peak memory is only as good as its sampling: query boundaries alone miss it.
        {
            let tracker = Arc::clone(&global_tracker);
            let stop = Arc::clone(&stop_sampler);
            let peak_heap = Arc::clone(&peak_heap);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    tracker.current_memory();
                    peak_heap.fetch_max(live_heap_bytes(), Ordering::Relaxed);
                    std::thread::sleep(PEAK_SAMPLE_INTERVAL);
                }
            });
        }
        Self {
            global_tracker,
            baseline_memory: None,
            stop_sampler,
            baseline_heap: 0,
            peak_heap,
        }
    }

    /// Mark the start of a query execution - should be called before running a query
    pub fn start_query(&mut self) {
        self.global_tracker.reset_peak();
        self.baseline_memory = self.global_tracker.current_memory();
        self.baseline_heap = live_heap_bytes();
        self.peak_heap.store(self.baseline_heap, Ordering::Relaxed);
    }

    /// Heap bytes above the query's starting point: at peak, and still live after it finished.
    pub fn heap_usage(&self) -> (i64, i64) {
        let after = live_heap_bytes();
        let peak = self.peak_heap.load(Ordering::Relaxed).max(after);
        (peak - self.baseline_heap, after - self.baseline_heap)
    }

    /// Mark the end of a query execution and collect memory measurements
    /// Returns None if no baseline was set or memory tracking failed
    pub fn end_query(&self) -> Option<MemoryMeasurementResult> {
        let baseline = self.baseline_memory?;
        let after_memory = self.global_tracker.current_memory()?;
        let usage_diff = baseline.diff(&after_memory);

        // Get peak memory from global tracker
        let peak_memory = self.global_tracker.peak_memory();

        Some(MemoryMeasurementResult {
            physical_memory_delta: usage_diff.physical_memory_delta,
            virtual_memory_delta: usage_diff.virtual_memory_delta,
            peak_physical_memory: peak_memory.physical_memory,
            peak_virtual_memory: peak_memory.virtual_memory,
        })
    }

    /// Get the current peak memory without ending a query
    pub fn peak_memory(&self) -> MemoryStats {
        self.global_tracker.peak_memory()
    }
}

impl Drop for BenchmarkMemoryTracker {
    fn drop(&mut self) {
        self.stop_sampler.store(true, Ordering::Relaxed);
    }
}

impl MemoryMeasurement {
    /// Start a memory measurement
    pub fn start() -> Self {
        let tracker = MemoryTracker::new();
        let before = tracker.current_memory();

        Self { tracker, before }
    }

    /// End the measurement and return the memory usage difference
    pub fn end(self) -> Option<MemoryStatsDiff> {
        let after = self.tracker.current_memory()?;
        let before = self.before?;

        Some(after.diff(&before))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new();
        let memory = tracker.current_memory();
        assert!(memory.is_some());

        if let Some(stats) = memory {
            assert!(stats.physical_memory > 0);
            assert!(stats.virtual_memory > 0);
        }
    }

    #[test]
    fn test_memory_measurement() {
        let measurement = MemoryMeasurement::start();

        // Allocate some memory
        let _data: Vec<u8> = vec![0; 1024 * 1024]; // 1MB

        let diff = measurement.end();
        assert!(diff.is_some());
    }
}
