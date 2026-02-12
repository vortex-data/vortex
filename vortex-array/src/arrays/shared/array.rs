// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::sync::Arc;

use async_lock::Mutex;
use async_lock::MutexGuard;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::stats::ArrayStats;

#[derive(Debug, Clone)]
pub struct SharedArray {
    pub(super) state: Arc<Mutex<SharedState>>,
    pub(super) dtype: DType,
    pub(super) stats: ArrayStats,
}

#[derive(Debug, Clone)]
pub(super) enum SharedState {
    Source(ArrayRef),
    Cached(Canonical),
}

impl SharedArray {
    /// Creates a new `SharedArray` wrapping the given source array.
    pub fn new(source: ArrayRef) -> Self {
        Self {
            dtype: source.dtype().clone(),
            state: Arc::new(Mutex::new(SharedState::Source(source))),
            stats: ArrayStats::default(),
        }
    }

    #[cfg(not(target_family = "wasm"))]
    fn lock_sync(&self) -> MutexGuard<'_, SharedState> {
        self.state.lock_blocking()
    }

    #[cfg(target_family = "wasm")]
    fn lock_sync(&self) -> MutexGuard<'_, SharedState> {
        // this should mirror how parking_lot compiles to wasm
        self.state
            .try_lock()
            .expect("SharedArray: mutex contention on single-threaded wasm target")
    }

    pub fn get_or_compute(
        &self,
        f: impl FnOnce(&ArrayRef) -> VortexResult<Canonical>,
    ) -> VortexResult<Canonical> {
        let mut state = self.lock_sync();
        match &*state {
            SharedState::Cached(canonical) => Ok(canonical.clone()),
            SharedState::Source(source) => {
                let canonical = f(source)?;
                *state = SharedState::Cached(canonical.clone());
                Ok(canonical)
            }
        }
    }

    pub async fn get_or_compute_async<F, Fut>(&self, f: F) -> VortexResult<Canonical>
    where
        F: FnOnce(ArrayRef) -> Fut,
        Fut: Future<Output = VortexResult<Canonical>>,
    {
        let mut state = self.state.lock().await;
        match &*state {
            SharedState::Cached(canonical) => Ok(canonical.clone()),
            SharedState::Source(source) => {
                let source = source.clone();
                let canonical = f(source).await?;
                *state = SharedState::Cached(canonical.clone());
                Ok(canonical)
            }
        }
    }

    pub(super) fn current_array_ref(&self) -> ArrayRef {
        let state = self.lock_sync();
        match &*state {
            SharedState::Source(source) => source.clone(),
            SharedState::Cached(canonical) => canonical.clone().into_array(),
        }
    }

    pub(super) fn set_source(&mut self, source: ArrayRef) {
        self.dtype = source.dtype().clone();
        *self.lock_sync() = SharedState::Source(source);
    }
}
