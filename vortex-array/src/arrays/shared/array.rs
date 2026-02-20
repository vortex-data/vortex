// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::backtrace::Backtrace;
use std::future::Future;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use async_lock::Mutex;
use async_lock::MutexGuard;
use vortex_error::{VortexError, VortexResult};

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::dtype::DType;
use crate::stats::ArrayStats;

#[derive(Debug, Clone)]
pub struct SharedArray {
    pub(super) state: Arc<RwLock<SharedState>>,
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
            state: Arc::new(RwLock::new(SharedState::Source(source))),
            stats: ArrayStats::default(),
        }
    }

    #[cfg(not(target_family = "wasm"))]
    fn wlock_sync(&self) -> VortexResult<RwLockWriteGuard<'_, SharedState>> {
        let result = self.state.write();
        match result {
            Ok(guard) => Ok(guard),
            Err(_) => Err(VortexError::Other(
                "SharedArray: RwLock poisoned".into(),
                Box::new(Backtrace::capture()),
            )),
        }
    }

    #[cfg(not(target_family = "wasm"))]
    fn rlock_sync(&self) -> VortexResult<RwLockReadGuard<'_, SharedState>> {
        let result = self.state.read();
        match result {
            Ok(guard) => Ok(guard),
            Err(_) => Err(VortexError::Other(
                "SharedArray: RwLock poisoned".into(),
                Box::new(Backtrace::capture()),
            )),
        }
    }

    #[cfg(target_family = "wasm")]
    fn wlock_sync(&self) -> VortexResult<RwLockWriteGuard<'_, SharedState>> {
        // this should mirror how parking_lot compiles to wasm
        self.state
            .try_write()
            .map_err(|_| VortexError::Other(
                "SharedArray: mutex contention on single-threaded wasm target".into(),
                Box::new(Backtrace::capture()),
            ))
    }

    #[cfg(target_family = "wasm")]
    fn rlock_sync(&self) -> VortexResult<RwLockReadGuard<'_, SharedState>> {
        // this should mirror how parking_lot compiles to wasm
        self.state.try_read()
            .map_err(|_| VortexError::Other(
                "SharedArray: mutex contention on single-threaded wasm target".into(),
                Box::new(Backtrace::capture()),
            ))
    }


    pub fn get_or_compute(
        &self,
        f: impl FnOnce(&ArrayRef) -> VortexResult<Canonical>,
    ) -> VortexResult<Canonical> {
        let mut state = self.rlock_sync();
        match &*state {
            SharedState::Cached(canonical) => Ok(canonical.clone()),
            SharedState::Source(source) => {
                drop(state);
                let canonical = f(source)?;
                let state = self.wlock_sync();
                match &*state {
                    SharedState::Cached(canonical) => Ok(canonical.clone()),
                    SharedState::Source(_) => {
                        *state = SharedState::Cached(canonical.clone());
                        Ok(canonical)
                    }
                }
            }
        }
    }

    pub async fn get_or_compute_async<F, Fut>(&self, f: F) -> VortexResult<Canonical>
    where
        F: FnOnce(ArrayRef) -> Fut,
        Fut: Future<Output=VortexResult<Canonical>>,
    {
        let mut state = self.state.read().await;
        match &*state {
            SharedState::Cached(canonical) => Ok(canonical.clone()),
            SharedState::Source(source) => {
                let source = source.clone();
                drop(state);
                let canonical = f(source).await?;
                let mut state = self.state.write().await;
                match &*state {
                    SharedState::Cached(canonical) => Ok(canonical.clone()),
                    SharedState::Source(_) => {
                        *state = SharedState::Cached(canonical.clone());
                        Ok(canonical)
                    }
                }
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
