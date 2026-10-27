// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::future::Future;
use std::sync::Arc;
use std::sync::OnceLock;

use async_lock::Mutex as AsyncMutex;
use vortex_array::Array;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_error::SharedVortexResult;
use vortex_error::VortexResult;

use crate::arrays::shared::Shared;
use crate::arrays::shared::SharedArray;

#[array_slots(Shared)]
pub struct SharedSlots {
    /// The source array that is shared and lazily computed.
    #[slot(0)]
    pub source: ArrayRef,
}

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
pub trait SharedArrayExt: TypedArrayRef<Shared> + SharedArraySlotsExt {
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

impl Shared {
    /// Creates a new [`SharedArray`] wrapping `source`.
    pub fn new(source: ArrayRef) -> SharedArray {
        let dtype = source.dtype().clone();
        let len = source.len();
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(
                Shared,
                dtype,
                len,
                SharedData::new(),
                SharedSlots { source }.into_slots(),
            ))
        }
    }
}
