// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(not(feature = "dhat-heap"))]
use anyhow::bail;

/// Peak heap statistics collected by dhat.
#[derive(Debug, Clone, Copy)]
pub struct HeapStats {
    pub max_bytes: u64,
}

impl HeapStats {
    pub fn max_mib(self) -> f64 {
        self.max_bytes as f64 / 1024.0 / 1024.0
    }
}

/// Guard for a dhat heap profile.
pub struct HeapProfiler {
    #[cfg(feature = "dhat-heap")]
    profiler: ::dhat::Profiler,
}

/// Start a heap profile for the current thread.
pub fn start_heap_profiling() -> anyhow::Result<HeapProfiler> {
    #[cfg(feature = "dhat-heap")]
    {
        Ok(HeapProfiler {
            profiler: ::dhat::Profiler::builder().testing().build(),
        })
    }

    #[cfg(not(feature = "dhat-heap"))]
    {
        bail!("dhat heap profiling is disabled; rebuild with the `dhat-heap` feature")
    }
}

impl HeapProfiler {
    pub fn finish(self) -> HeapStats {
        #[cfg(feature = "dhat-heap")]
        {
            let stats = ::dhat::HeapStats::get();
            drop(self.profiler);
            HeapStats {
                max_bytes: u64::try_from(stats.max_bytes).unwrap_or(u64::MAX),
            }
        }

        #[cfg(not(feature = "dhat-heap"))]
        {
            let _ = self;
            unreachable!("HeapProfiler can only be constructed with the `dhat-heap` feature");
        }
    }
}
