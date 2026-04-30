// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC;
use vortex::error::VortexResult;
use vortex_cuda::CudaKernelEvents;
use vortex_cuda::LaunchStrategy;

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
    fn event_flags(&self) -> CUevent_flags {
        // using blocking_sync to make sure all events flush before we complete.
        CU_EVENT_BLOCKING_SYNC
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
