// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::sync::Arc;
use std::sync::OnceLock;

use async_lock::Mutex as AsyncMutex;
use vortex_error::SharedVortexResult;
use vortex_error::VortexResult;

use crate::ArrayCommon;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;

/// A lazily-executing array wrapper with a one-way transition from source to cached form.
///
/// Before materialization, operations delegate to the source array.
/// After materialization (via `get_or_compute`), operations delegate to the cached result.
#[derive(Debug, Clone)]
pub struct SharedArray {
    source: ArrayRef,
    cached: Arc<OnceLock<SharedVortexResult<ArrayRef>>>,
    async_compute_lock: Arc<AsyncMutex<()>>,
    pub(super) common: ArrayCommon,
}

impl SharedArray {
    pub fn new(source: ArrayRef) -> Self {
        let common = ArrayCommon::new(source.len(), source.dtype().clone());
        Self {
            source,
            cached: Arc::new(OnceLock::new()),
            async_compute_lock: Arc::new(AsyncMutex::new(())),
            common,
        }
    }

    /// Returns the current array reference.
    ///
    /// After materialization, returns the cached result. Otherwise, returns the source.
    /// If materialization failed, falls back to the source.
    pub(super) fn current_array_ref(&self) -> &ArrayRef {
        match self.cached.get() {
            Some(Ok(arr)) => arr,
            _ => &self.source,
        }
    }

    /// Compute and cache the result. The computation runs exactly once via `OnceLock`.
    ///
    /// If the computation fails, the error is cached and returned on all subsequent calls.
    pub fn get_or_compute(
        &self,
        f: impl FnOnce(&ArrayRef) -> VortexResult<Canonical>,
    ) -> VortexResult<ArrayRef> {
        let result = self
            .cached
            .get_or_init(|| f(&self.source).map(|c| c.into_array()).map_err(Arc::new));
        result.clone().map_err(Into::into)
    }

    /// Async version of `get_or_compute`.
    pub async fn get_or_compute_async<F, Fut>(&self, f: F) -> VortexResult<ArrayRef>
    where
        F: FnOnce(ArrayRef) -> Fut,
        Fut: Future<Output = VortexResult<Canonical>>,
    {
        // Fast path: already computed.
        if let Some(result) = self.cached.get() {
            return result.clone().map_err(Into::into);
        }

        // Serialize async computation to prevent redundant work.
        let _guard = self.async_compute_lock.lock().await;

        // Double-check after acquiring the lock.
        if let Some(result) = self.cached.get() {
            return result.clone().map_err(Into::into);
        }

        let computed = f(self.source.clone())
            .await
            .map(|c| c.into_array())
            .map_err(Arc::new);

        let result = self.cached.get_or_init(|| computed);
        result.clone().map_err(Into::into)
    }

    pub(super) fn set_source(&mut self, source: ArrayRef) {
        self.common = ArrayCommon::new(source.len(), source.dtype().clone());
        self.source = source;
        self.cached = Arc::new(OnceLock::new());
        self.async_compute_lock = Arc::new(AsyncMutex::new(()));
    }
}
