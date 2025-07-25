// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[derive(Debug, Clone)]
pub struct PerformanceHint {
    coalescing_window: usize,
    max_read: Option<usize>,
}

impl PerformanceHint {
    pub fn new(coalescing_window: usize, max_read: Option<usize>) -> Self {
        Self {
            coalescing_window,
            max_read,
        }
    }

    /// Creates a new instance with a profile appropriate for in-memory reads.
    pub fn in_memory() -> Self {
        Self::new(0, None)
    }

    /// Creates a new instance with a profile appropriate for fast local storage, like memory or files on NVMe devices.
    pub fn local() -> Self {
        // Coalesce ~8K page size, also ensures we span padding for adjacent segments.
        Self::new(8192, Some(8192))
    }

    pub fn object_storage() -> Self {
        Self::new(
            1 << 20,       // 1MB,
            Some(8 << 20), // 8MB,
        )
    }

    /// The maximum distance between two reads that should coalesce into a single operation.
    pub fn coalescing_window(&self) -> usize {
        self.coalescing_window
    }

    /// Maximum number of bytes in a coalesced read.
    pub fn max_read(&self) -> Option<usize> {
        self.max_read
    }
}
