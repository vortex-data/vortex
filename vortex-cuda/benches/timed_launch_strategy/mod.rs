// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex::error::VortexResult;
use vortex_cuda::CudaKernelEvents;
use vortex_cuda::LaunchStrategy;
use vortex_cuda_driver::sys::CUevent_flags;

#[derive(Debug, Default)]
pub struct TimedLaunchStrategy {
    pub total_time_ns: Arc<AtomicU64>,
}

impl TimedLaunchStrategy {
    /// Returns a shared handle to the accumulated kernel time, for reading after launches complete.
    pub fn timer(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.total_time_ns)
    }
}

impl LaunchStrategy for TimedLaunchStrategy {
    fn event_flags(&self) -> Option<CUevent_flags> {
        // Let the benchmark thread block rather than busy-wait while `duration()`
        // waits for the CUDA events recorded just before and just after the kernel.
        Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC)
    }

    fn on_complete(&self, events: &CudaKernelEvents, _len: usize) -> VortexResult<()> {
        // NOTE: as long as the duration < 584 years this cast is safe.
        #[allow(clippy::cast_possible_truncation)]
        let elapsed_nanos = events.duration()?.as_nanos() as u64;
        self.total_time_ns
            .fetch_add(elapsed_nanos, Ordering::Relaxed);

        Ok(())
    }
}
