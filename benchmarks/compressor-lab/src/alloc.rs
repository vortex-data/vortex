// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A counting global allocator, for peak-memory measures.
//!
//! The `vx-lab` binary installs [`Counting`] as its global allocator. Library code reads it
//! through [`Region`], which reports the peak bytes live above the region's start and the number
//! of allocations made inside it. Timing is single-threaded, so the counts belong to the measured
//! operation. When the allocator is not installed (unit tests, other binaries), regions report
//! `None`.

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

static INSTALLED: AtomicBool = AtomicBool::new(false);
static CURRENT: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);

/// The system allocator, counting live bytes, their high-water mark and allocations.
pub struct Counting;

impl Counting {
    fn grow(bytes: u64) {
        let now = CURRENT.fetch_add(bytes, Ordering::Relaxed) + bytes;
        PEAK.fetch_max(now, Ordering::Relaxed);
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        if !INSTALLED.load(Ordering::Relaxed) {
            INSTALLED.store(true, Ordering::Relaxed);
        }
    }

    fn shrink(bytes: u64) {
        CURRENT.fetch_sub(bytes, Ordering::Relaxed);
    }
}

// SAFETY: every call is forwarded to `System`; the counters have no effect on the memory handed out.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            Self::grow(layout.size() as u64);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            Self::grow(layout.size() as u64);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) };
        Self::shrink(layout.size() as u64);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            Self::shrink(layout.size() as u64);
            Self::grow(new_size as u64);
        }
        new
    }
}

/// Memory used inside a [`Region`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryUse {
    /// Peak live bytes above the region's starting level.
    pub peak_bytes: u64,
    /// Allocations made inside the region.
    pub allocations: u64,
}

/// A span of code whose memory use is measured.
pub struct Region {
    start_bytes: u64,
    start_allocs: u64,
}

impl Region {
    /// Starts a region: resets the high-water mark to the current level.
    pub fn start() -> Self {
        let now = CURRENT.load(Ordering::Relaxed);
        PEAK.store(now, Ordering::Relaxed);
        Self {
            start_bytes: now,
            start_allocs: ALLOCS.load(Ordering::Relaxed),
        }
    }

    /// Ends the region. `None` when the counting allocator is not installed.
    pub fn finish(self) -> Option<MemoryUse> {
        if !INSTALLED.load(Ordering::Relaxed) {
            return None;
        }
        Some(MemoryUse {
            peak_bytes: PEAK
                .load(Ordering::Relaxed)
                .saturating_sub(self.start_bytes),
            allocations: ALLOCS
                .load(Ordering::Relaxed)
                .saturating_sub(self.start_allocs),
        })
    }
}
