// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-export options, including options defined by external Arrow export plugins.

use std::any::Any;
use std::any::TypeId;
use std::fmt;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

/// Typed options propagated to every nested array in an Arrow export.
///
/// Export plugins can define their own option types without modifying this crate. Each type has
/// at most one value; absent options leave the exporter's default behavior unchanged.
///
/// ```
/// use vortex_arrow::{ArrowExportOptions, CompactBuffers};
///
/// let options = ArrowExportOptions::default().with(CompactBuffers(false));
/// assert_eq!(options.get::<CompactBuffers>(), Some(&CompactBuffers(false)));
/// ```
#[derive(Clone, Default)]
pub struct ArrowExportOptions {
    values: Vec<(TypeId, Arc<dyn Any + Send + Sync>)>,
}

impl ArrowExportOptions {
    /// Set an option, replacing any previous value of the same type.
    pub fn with<T: Any + Send + Sync>(mut self, option: T) -> Self {
        let id = TypeId::of::<T>();
        if let Some((_, value)) = self.values.iter_mut().find(|(key, _)| *key == id) {
            *value = Arc::new(option);
        } else {
            self.values.push((id, Arc::new(option)));
        }
        self
    }

    /// Look up an option defined by this crate or an external exporter.
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.values
            .iter()
            .find(|(key, _)| *key == TypeId::of::<T>())
            .and_then(|(_, value)| value.downcast_ref())
    }

    /// Look up an option, falling back to its default when unset.
    pub fn get_or_default<T: Any + Send + Sync + Clone + Default>(&self) -> T {
        self.get::<T>().cloned().unwrap_or_default()
    }
}

impl fmt::Debug for ArrowExportOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrowExportOptions")
            .field("option_count", &self.values.len())
            .finish_non_exhaustive()
    }
}

/// Whether to compact string and binary view buffers during Arrow export.
///
/// Enabled by default. Disabling compaction avoids scans and copies, but the exported array may
/// retain backing data outside the selected rows. Values and the Arrow type are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactBuffers(pub bool);

impl Default for CompactBuffers {
    fn default() -> Self {
        Self(true)
    }
}

/// Reuses the Arrow export of dictionary values that several Vortex dictionaries share.
///
/// A scan emits one Vortex dictionary per batch, and the batches of one dictionary chunk share
/// the same values array. With this option set, a dictionary exported as an Arrow `Dictionary`
/// exports those values once, and every batch then shares one Arrow values array. Engines that
/// cache work per values array, such as hashes of the dictionary values, can therefore reuse it
/// across batches.
///
/// Clones share one cache. The cache keeps the most recently exported values, at most
/// [`DictionaryValuesCache::CAPACITY`] of them, and keeps each Vortex values array it holds alive,
/// so a cached address can never be reused by another array.
#[derive(Clone, Default)]
pub struct DictionaryValuesCache(Arc<Mutex<Vec<CachedDictionaryValues>>>);

struct CachedDictionaryValues {
    values: ArrayRef,
    data_type: DataType,
    arrow: ArrowArrayRef,
}

impl DictionaryValuesCache {
    /// The most exported values arrays the cache holds at once.
    pub const CAPACITY: usize = 64;

    /// The cached Arrow export of `values` as `data_type`, or the result of `export`, which is
    /// then cached.
    pub(crate) fn get_or_try_insert(
        &self,
        values: &ArrayRef,
        data_type: &DataType,
        export: impl FnOnce() -> VortexResult<ArrowArrayRef>,
    ) -> VortexResult<ArrowArrayRef> {
        if let Some(arrow) = self.get(values, data_type) {
            return Ok(arrow);
        }
        // The lock is not held while exporting: a concurrent export of the same values only
        // repeats the work, and the first result to arrive is the one every later batch shares.
        let arrow = export()?;
        let mut entries = self.0.lock();
        if let Some(entry) = entries
            .iter()
            .find(|entry| entry.matches(values, data_type))
        {
            return Ok(Arc::clone(&entry.arrow));
        }
        if entries.len() >= Self::CAPACITY {
            entries.remove(0);
        }
        entries.push(CachedDictionaryValues {
            values: values.clone(),
            data_type: data_type.clone(),
            arrow: Arc::clone(&arrow),
        });
        Ok(arrow)
    }

    fn get(&self, values: &ArrayRef, data_type: &DataType) -> Option<ArrowArrayRef> {
        self.0
            .lock()
            .iter()
            .find(|entry| entry.matches(values, data_type))
            .map(|entry| Arc::clone(&entry.arrow))
    }

    /// The number of exported values arrays the cache holds.
    pub fn len(&self) -> usize {
        self.0.lock().len()
    }

    /// Whether the cache holds no exported values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl CachedDictionaryValues {
    fn matches(&self, values: &ArrayRef, data_type: &DataType) -> bool {
        ArrayRef::ptr_eq(&self.values, values) && &self.data_type == data_type
    }
}

impl fmt::Debug for DictionaryValuesCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DictionaryValuesCache")
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests;
