// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_session::registry::Id;
use vortex_utils::aliases::hash_map::HashMap;

/// Per-reader-tree dependency context, threaded through [`crate::VTable::new_reader`].
///
/// Holds an [`Id`]-keyed registry of `Arc<dyn Any>` values. Ancestors publish via
/// [`Self::with`]; descendants retrieve via [`Self::get`]. This is a *read-only* channel
/// from the descendant's perspective — they can only consume what an ancestor chose to
/// publish.
///
/// [`Self::with`] returns a derived context that copies the existing map and inserts or
/// replaces one entry — original unchanged, so concurrent reader-tree constructions each
/// derive their own context without races. Contexts are cheap to clone via internal `Arc`
/// and can be captured by lazy children helpers.
#[derive(Clone, Default)]
pub struct LayoutReaderContext {
    values: Arc<HashMap<Id, Arc<dyn Any + Send + Sync>>>,
}

impl std::fmt::Debug for LayoutReaderContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayoutReaderContext")
            .field("ids", &self.values.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl LayoutReaderContext {
    /// Creates a new, empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a new context that publishes `value` under `id`.
    ///
    /// The original context is unchanged. If a value was already published under the
    /// same `id`, the new one replaces it in the returned context — so two ancestors
    /// using the same well-known static id give descendants "nearest ancestor wins".
    pub fn with<T: Any + Send + Sync>(&self, id: Id, value: Arc<T>) -> Self {
        let mut values = HashMap::clone(&self.values);
        values.insert(id, value);
        Self {
            values: Arc::new(values),
        }
    }

    /// Returns the value published under `id`, downcast to `T`. Returns `None` if no
    /// ancestor published under that id, or if the published value is not a `T`.
    pub fn get<T: Any + Send + Sync>(&self, id: Id) -> Option<Arc<T>> {
        self.values
            .get(&id)
            .and_then(|v| Arc::clone(v).downcast::<T>().ok())
    }
}
