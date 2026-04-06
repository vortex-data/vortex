// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::sync::Arc;
use std::sync::OnceLock;

use async_lock::Mutex as AsyncMutex;
use vortex_error::SharedVortexResult;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::array::Array;
use crate::arrays::Shared;
use crate::dtype::DType;
use crate::stats::ArrayStats;

/// The source array that is shared and lazily computed.
pub(super) const SOURCE_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["source"];

/// A lazily-executing array wrapper with a one-way transition from source to cached form.
///
/// Before materialization, operations delegate to the source array.
/// After materialization (via `get_or_compute`), operations delegate to the cached result.
#[derive(Debug, Clone)]
pub struct SharedData {
    pub(super) slots: Vec<Option<ArrayRef>>,
    cached: Arc<OnceLock<SharedVortexResult<ArrayRef>>>,
    async_compute_lock: Arc<AsyncMutex<()>>,
    pub(super) dtype: DType,
    pub(super) stats: ArrayStats,
}

impl SharedData {
    pub fn new(source: ArrayRef) -> Self {
        Self {
            dtype: source.dtype().clone(),
            slots: vec![Some(source)],
            cached: Arc::new(OnceLock::new()),
            async_compute_lock: Arc::new(AsyncMutex::new(())),
            stats: ArrayStats::default(),
        }
    }

    /// Returns the source array reference.
    pub(super) fn source(&self) -> &ArrayRef {
        self.slots[SOURCE_SLOT]
            .as_ref()
            .vortex_expect("SharedArray source slot")
    }

    /// Returns the current array reference.
    ///
    /// After materialization, returns the cached result. Otherwise, returns the source.
    /// If materialization failed, falls back to the source.
    pub(super) fn current_array_ref(&self) -> &ArrayRef {
        match self.cached.get() {
            Some(Ok(arr)) => arr,
            _ => self.source(),
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
            .get_or_init(|| f(self.source()).map(|c| c.into_array()).map_err(Arc::new));
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

        let computed = f(self.source().clone())
            .await
            .map(|c| c.into_array())
            .map_err(Arc::new);

        let result = self.cached.get_or_init(|| computed);
        result.clone().map_err(Into::into)
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.current_array_ref().len()
    }

    /// Returns the [`DType`] of this array.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Array<Shared> {
    /// Creates a new `SharedArray`.
    pub fn new(source: ArrayRef) -> Self {
        Array::try_from_data(SharedData::new(source)).vortex_expect("SharedData is always valid")
    }
}

impl SharedData {
    pub(super) fn set_source(&mut self, source: Option<ArrayRef>) {
        if let Some(ref s) = source {
            self.dtype = s.dtype().clone();
        }
        self.slots = vec![source];
        self.cached = Arc::new(OnceLock::new());
        self.async_compute_lock = Arc::new(AsyncMutex::new(()));
    }
}
