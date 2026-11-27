// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A collection of built-in layouts for Vortex

use futures::future::BoxFuture;
use futures::future::Shared;
use parking_lot::RwLock;
use vortex_array::ArrayRef;
use vortex_error::SharedVortexResult;

pub mod buffered;
pub mod chunked;
pub mod collect;
pub mod compressed;
pub mod constant;
pub mod dict;
pub mod file_stats;
pub mod flat;
pub(crate) mod foreign;
pub mod indexed;
pub mod list;
pub(crate) mod partitioned;
pub mod repartition;
pub mod row_idx;
pub mod struct_;
pub mod table;
pub mod variant;
pub mod zoned;

pub type SharedArrayFuture = Shared<BoxFuture<'static, SharedVortexResult<ArrayRef>>>;

/// Returns `true` if a shared future is still in-flight or resolved to `Ok`.
pub(crate) fn is_shared_future_usable<T: Clone>(
    fut: &Shared<BoxFuture<'static, SharedVortexResult<T>>>,
) -> bool {
    !matches!(fut.peek(), Some(Err(_)))
}

/// Caches one shared future while it is in-flight or resolved to `Ok`.
///
/// Once the cached future resolves to `Err`, a later caller replaces it with a fresh attempt.
pub(crate) struct SharedFutureSlot<T> {
    slot: RwLock<Option<Shared<BoxFuture<'static, SharedVortexResult<T>>>>>,
}

impl<T> Default for SharedFutureSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SharedFutureSlot<T> {
    pub(crate) fn new() -> Self {
        Self {
            slot: RwLock::new(None),
        }
    }
}

impl<T: Clone> SharedFutureSlot<T> {
    /// Returns the cached future if it is in-flight or resolved to `Ok`.
    pub(crate) fn get(&self) -> Option<Shared<BoxFuture<'static, SharedVortexResult<T>>>> {
        let guard = self.slot.read();
        guard
            .as_ref()
            .filter(|fut| is_shared_future_usable(fut))
            .cloned()
    }

    /// Returns the cached future if it is in-flight or resolved to `Ok`, otherwise initializes
    /// and caches a new future using `init`.
    pub(crate) fn get_or_init(
        &self,
        init: impl FnOnce() -> Shared<BoxFuture<'static, SharedVortexResult<T>>>,
    ) -> Shared<BoxFuture<'static, SharedVortexResult<T>>> {
        {
            let guard = self.slot.read();
            if let Some(fut) = guard.as_ref()
                && is_shared_future_usable(fut)
            {
                return fut.clone();
            }
        }

        let mut guard = self.slot.write();
        if let Some(fut) = guard.as_ref()
            && is_shared_future_usable(fut)
        {
            return fut.clone();
        }

        let fut = init();
        *guard = Some(fut.clone());
        fut
    }
}
