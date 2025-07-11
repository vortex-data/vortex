// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Memory measurement and reclamation utilities for benchmarks

use parking_lot::Mutex;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

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
            RefreshKind::default().with_processes(ProcessRefreshKind::everything()),
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

/// Force memory reclamation using platform-specific methods
pub fn force_memory_reclaim() {
    #[cfg(target_os = "linux")]
    {
        // Use malloc_trim on Linux (glibc)
        unsafe {
            libc::malloc_trim(0);
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Use malloc_zone_pressure_relief on macOS
        unsafe extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
            fn malloc_default_zone() -> *mut std::ffi::c_void;
        }

        unsafe {
            malloc_zone_pressure_relief(malloc_default_zone(), 0);
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Use _heapmin on Windows
        extern "C" {
            fn _heapmin() -> i32;
        }

        unsafe {
            _heapmin();
        }
    }

    // Force Rust garbage collection by running GC
    // This is a hint to the allocator to release unused memory
    std::hint::black_box(Vec::<u8>::with_capacity(1));
}

/// Memory measurement guard that tracks memory usage before and after an operation
pub struct MemoryMeasurement {
    tracker: MemoryTracker,
    before: Option<MemoryStats>,
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

        Some(before.diff(&after))
    }

    /// End the measurement, force memory reclamation, and return both measurements
    pub fn end_with_reclaim(self) -> Option<(MemoryStatsDiff, MemoryStatsDiff)> {
        let after = self.tracker.current_memory()?;
        let before = self.before?;

        let usage_diff = before.diff(&after);

        // Force memory reclamation
        force_memory_reclaim();

        // Measure memory after reclamation
        let after_reclaim = self.tracker.current_memory()?;
        let reclaim_diff = after.diff(&after_reclaim);

        Some((usage_diff, reclaim_diff))
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

    #[test]
    fn test_force_memory_reclaim() {
        // This should not panic
        force_memory_reclaim();
    }
}
