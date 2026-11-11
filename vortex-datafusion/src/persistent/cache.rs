// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::HashMap;
use datafusion_execution::cache::cache_manager::FileMetadata;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;
use vortex::file::Footer;
use vortex::file::VortexFile;
use vortex::layout::segments::SegmentCache;
use vortex::layout::segments::SegmentId;

/// Cached Vortex file metadata for use with DataFusion's [`FileMetadataCache`].
///
/// [`FileMetadataCache`]: datafusion_execution::cache::cache_manager::FileMetadataCache
pub struct CachedVortexMetadata {
    footer: Footer,
    /// Segments a planner already read, such as the zone maps of the columns a filter reads, which
    /// the file's scan takes from memory instead of reading them again.
    segments: Arc<HashMap<SegmentId, ByteBuffer>>,
}

impl CachedVortexMetadata {
    /// Create a new cached metadata entry from a VortexFile.
    pub fn new(vortex_file: &VortexFile) -> Self {
        let footer = vortex_file.footer();
        Self {
            footer: footer.clone(),
            segments: Arc::default(),
        }
    }

    /// Create a new cached metadata entry from a parsed footer.
    pub fn from_footer(footer: Footer) -> Self {
        Self {
            footer,
            segments: Arc::default(),
        }
    }

    /// The entry with `segments` of the file already read.
    pub fn with_segments(mut self, segments: HashMap<SegmentId, ByteBuffer>) -> Self {
        self.segments = Arc::new(segments);
        self
    }

    /// A segment cache serving the segments already read, if any were.
    pub fn segment_cache(&self) -> Option<Arc<dyn SegmentCache>> {
        (!self.segments.is_empty()).then(|| {
            Arc::new(PrefetchedSegments {
                segments: Arc::clone(&self.segments),
            }) as _
        })
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
        self.footer
            .approx_byte_size()
            // 64KB is not an insane estimate...
            // We just want to avoid returning zero and _never_ being evicted from the cache.
            .unwrap_or(1024 * 64)
            + self.segments.values().map(|s| s.len()).sum::<usize>()
    }

    fn extra_info(&self) -> HashMap<String, String> {
        Default::default()
    }
}

/// Serves the segments a planner read ahead, and stores nothing else.
struct PrefetchedSegments {
    segments: Arc<HashMap<SegmentId, ByteBuffer>>,
}

#[async_trait]
impl SegmentCache for PrefetchedSegments {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.segments.get(&id).cloned())
    }

    async fn put(&self, _id: SegmentId, _buffer: ByteBuffer) -> VortexResult<()> {
        Ok(())
    }
}
