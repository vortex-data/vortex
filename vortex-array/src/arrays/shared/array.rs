// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
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
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::Shared;

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
    cached: Arc<OnceLock<SharedVortexResult<ArrayRef>>>,
    async_compute_lock: Arc<AsyncMutex<()>>,
}

impl Display for SharedData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

#[expect(async_fn_in_trait)]
pub trait SharedArrayExt: TypedArrayRef<Shared> {
    fn source(&self) -> &ArrayRef {
        self.as_ref().slots()[SOURCE_SLOT]
            .as_ref()
            .vortex_expect("validated shared source slot")
    }

    fn current_array_ref(&self) -> &ArrayRef {
        match self.cached.get() {
            Some(Ok(arr)) => arr,
            _ => self.source(),
        }
    }

    fn get_or_compute(
        &self,
        f: impl FnOnce(&ArrayRef) -> VortexResult<Canonical>,
    ) -> VortexResult<ArrayRef> {
        let result = self
            .cached
            .get_or_init(|| f(self.source()).map(|c| c.into_array()).map_err(Arc::new));
        result.clone().map_err(Into::into)
    }

    async fn get_or_compute_async<F, Fut>(&self, f: F) -> VortexResult<ArrayRef>
    where
        F: FnOnce(ArrayRef) -> Fut,
        Fut: Future<Output = VortexResult<Canonical>>,
    {
        if let Some(result) = self.cached.get() {
            return result.clone().map_err(Into::into);
        }

        let _guard = self.async_compute_lock.lock().await;

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
}
impl<T: TypedArrayRef<Shared>> SharedArrayExt for T {}

impl SharedData {
    pub fn new() -> Self {
        Self {
            cached: Arc::new(OnceLock::new()),
            async_compute_lock: Arc::new(AsyncMutex::new(())),
        }
    }
}

impl Default for SharedData {
    fn default() -> Self {
        Self::new()
    }
}

impl Array<Shared> {
    /// Creates a new `SharedArray`.
    pub fn new(source: ArrayRef) -> Self {
        let dtype = source.dtype().clone();
        let len = source.len();
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Shared, dtype, len, SharedData::new())
                    .with_slots(vec![Some(source)]),
            )
        }
    }
}
