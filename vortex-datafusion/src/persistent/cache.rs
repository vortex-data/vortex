// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use datafusion_common::ScalarValue;
use datafusion_execution::cache::cache_manager::FileMetadata;
use vortex::expr::stats::Precision;
use vortex::expr::stats::Stat;
use vortex::file::Footer;
use vortex::file::SegmentSpec;
use vortex::file::VortexFile;
use vortex::layout::segments::SegmentId;

/// Cached Vortex file metadata for use with DataFusion's [`FileMetadataCache`].
pub struct CachedVortexMetadata {
    footer: Footer,
}

impl CachedVortexMetadata {
    /// Create a new cached metadata entry from a VortexFile.
    pub fn new(vortex_file: &VortexFile) -> Self {
        Self {
            footer: vortex_file.footer().clone(),
        }
    }

    /// Get the cached footer.
    pub fn footer(&self) -> &Footer {
        &self.footer
    }
}

impl FileMetadata for CachedVortexMetadata {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn memory_size(&self) -> usize {
        estimate_footer_size(&self.footer)
    }

    #[allow(clippy::disallowed_types)]
    fn extra_info(&self) -> std::collections::HashMap<String, String> {
        Default::default()
    }
}

/// Approximate the in-memory size of a footer.
fn estimate_footer_size(footer: &Footer) -> usize {
    let segments_size = footer.segment_map().len() * size_of::<SegmentSpec>();
    let stats_size = footer
        .statistics()
        .map(|stats| {
            stats
                .iter()
                .map(|s| {
                    s.iter().count() * (size_of::<Stat>() + size_of::<Precision<ScalarValue>>())
                })
                .sum::<usize>()
        })
        .unwrap_or(0);

    let root_layout = footer.layout();
    let layout_size = size_of_val(footer.dtype())
        + root_layout.metadata().len()
        + root_layout.segment_ids().len() * size_of::<SegmentId>();

    segments_size + stats_size + layout_size
}
