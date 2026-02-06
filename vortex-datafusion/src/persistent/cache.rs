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
use vortex::layout::Layout;
use vortex::layout::segments::SegmentId;

/// Cached Vortex file metadata for use with DataFusion's [`FileMetadataCache`].
pub struct CachedVortexMetadata {
    footer: Footer,
    memory_size: usize,
}

impl CachedVortexMetadata {
    /// Create a new cached metadata entry from a VortexFile.
    pub fn new(vortex_file: &VortexFile) -> Self {
        let footer = vortex_file.footer();
        let memory_size = estimate_footer_size(footer);
        Self {
            footer: footer.clone(),
            memory_size,
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
        self.memory_size
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

    let layout_size = footer
        .layout()
        .depth_first_traversal()
        .filter_map(|l| l.ok().map(|l| layout_size(l.as_ref())))
        .sum::<usize>();

    segments_size + stats_size + layout_size
}

fn layout_size(layout: &dyn Layout) -> usize {
    size_of_val(layout.dtype())
        + layout.metadata().len()
        + layout.segment_ids().len() * size_of::<SegmentId>()
}
