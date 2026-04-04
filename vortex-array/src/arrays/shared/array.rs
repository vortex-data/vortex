// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::sync::Arc;
use std::sync::OnceLock;

use async_lock::Mutex as AsyncMutex;
use vortex_error::SharedVortexResult;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::arrays::Shared;
use crate::dtype::DType;

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

#[allow(async_fn_in_trait)]
pub trait SharedArrayExt {
    fn shared_data(&self) -> &SharedData;
    fn shared_dtype(&self) -> &DType;
    fn shared_len(&self) -> usize;

    fn source(&self) -> &ArrayRef {
        self.as_slots()[SOURCE_SLOT]
            .as_ref()
            .expect("validated shared source slot")
    }

    fn current_array_ref(&self) -> &ArrayRef {
        match self.shared_data().cached.get() {
            Some(Ok(arr)) => arr,
            _ => self.source(),
        }
    }

    fn get_or_compute(
        &self,
        f: impl FnOnce(&ArrayRef) -> VortexResult<Canonical>,
    ) -> VortexResult<ArrayRef> {
        let result = self.shared_data().cached.get_or_init(|| {
            f(self.source()).map(|c| c.into_array()).map_err(Arc::new)
        });
        result.clone().map_err(Into::into)
    }

    async fn get_or_compute_async<F, Fut>(&self, f: F) -> VortexResult<ArrayRef>
    where
        F: FnOnce(ArrayRef) -> Fut,
        Fut: Future<Output = VortexResult<Canonical>>,
    {
        if let Some(result) = self.shared_data().cached.get() {
            return result.clone().map_err(Into::into);
        }

        let _guard = self.shared_data().async_compute_lock.lock().await;

        if let Some(result) = self.shared_data().cached.get() {
            return result.clone().map_err(Into::into);
        }

        let computed = f(self.source().clone())
            .await
            .map(|c| c.into_array())
            .map_err(Arc::new);

        let result = self.shared_data().cached.get_or_init(|| computed);
        result.clone().map_err(Into::into)
    }

    fn as_slots(&self) -> &[Option<ArrayRef>];
}

impl SharedArrayExt for Array<Shared> {
    fn shared_data(&self) -> &SharedData {
        self.data()
    }

    fn shared_dtype(&self) -> &DType {
        self.dtype()
    }

    fn shared_len(&self) -> usize {
        self.len()
    }

    fn as_slots(&self) -> &[Option<ArrayRef>] {
        self.slots()
    }
}

impl SharedArrayExt for ArrayView<'_, Shared> {
    fn shared_data(&self) -> &SharedData {
        self.data()
    }

    fn shared_dtype(&self) -> &DType {
        self.dtype()
    }

    fn shared_len(&self) -> usize {
        self.len()
    }

    fn as_slots(&self) -> &[Option<ArrayRef>] {
        self.slots()
    }
}

impl SharedData {
    pub fn new() -> Self {
        Self {
            cached: Arc::new(OnceLock::new()),
            async_compute_lock: Arc::new(AsyncMutex::new(())),
        }
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
