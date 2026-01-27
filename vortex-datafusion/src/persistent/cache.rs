// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;
use std::sync::Arc;

use datafusion_common::ScalarValue;
use datafusion_execution::cache::cache_manager::FileMetadata;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::expr::stats::Precision;
use vortex::expr::stats::Stat;
use vortex::file::VortexFile;

/// Cached Vortex file metadata for use with DataFusion's [`FileMetadataCache`].
pub struct CachedVortexMetadata {
    dtype: DType,
    file_stats: Option<Arc<[StatsSet]>>,
    row_count: u64,
}

impl CachedVortexMetadata {
    /// Create a new cached metadata entry from a VortexFile.
    pub fn new(vortex_file: &VortexFile) -> Self {
        Self {
            dtype: vortex_file.dtype().clone(),
            file_stats: vortex_file.file_stats().cloned(),
            row_count: vortex_file.row_count(),
        }
    }

    /// Get the cached dtype.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Get the cached file stats.
    pub fn file_stats(&self) -> Option<&Arc<[StatsSet]>> {
        self.file_stats.as_ref()
    }

    /// Get the cached row count.
    pub fn row_count(&self) -> u64 {
        self.row_count
    }
}

impl FileMetadata for CachedVortexMetadata {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn memory_size(&self) -> usize {
        // Estimate based on dtype size + stats size
        let stats_size = self
            .file_stats
            .as_ref()
            .map(|stats| {
                stats
                    .iter()
                    .map(|s| {
                        s.iter().count() * (size_of::<Stat>() + size_of::<Precision<ScalarValue>>())
                    })
                    .sum::<usize>()
            })
            .unwrap_or(0);
        size_of::<DType>() + stats_size + size_of::<u64>()
    }

    #[allow(clippy::disallowed_types)]
    fn extra_info(&self) -> std::collections::HashMap<String, String> {
        Default::default()
    }
}
