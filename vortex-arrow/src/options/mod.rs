// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-export options, including options defined by external Arrow export plugins.

use std::any::Any;
use std::any::TypeId;
use std::fmt;
use std::sync::Arc;

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

#[cfg(test)]
mod tests;
