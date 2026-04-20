// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use datafusion_execution::cache::cache_manager::FileMetadata;
use vortex::file::Footer;
use vortex::file::VortexFile;

/// Cached Vortex file metadata for use with DataFusion's [`FileMetadataCache`].
pub struct CachedVortexMetadata {
    footer: Footer,
}

impl CachedVortexMetadata {
    /// Create a new cached metadata entry from a VortexFile.
    pub fn new(vortex_file: &VortexFile) -> Self {
        let footer = vortex_file.footer();
        Self {
            footer: footer.clone(),
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
        self.footer
            .approx_byte_size()
            // 64KB is not an insane estimate...
            // We just want to avoid returning zero and _never_ being evicted from the cache.
            .unwrap_or(1024 * 64)
    }

    #[expect(clippy::disallowed_types)]
    fn extra_info(&self) -> std::collections::HashMap<String, String> {
        Default::default()
    }
}
