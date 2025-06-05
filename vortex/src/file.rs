use std::path::Path;
use std::sync::Arc;

use moka::sync::Cache;
use vortex_array::aliases::DefaultHashBuilder;
use vortex_array::stats::{Precision, Stat};
use vortex_dtype::DType;
use vortex_error::VortexResult;
pub use vortex_file::*;
use vortex_layout::segments::SegmentId;
use vortex_scalar::ScalarValue;

use crate::session::VortexSession;

/// Cache key for a [`VortexFile`].
#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub struct FileKey {
    // TODO: support last modified ts.
    pub location: String,
}

/// Long-lived cache of [`VortexFile`]s.
///
/// The cache is associated with a [`VortexSession`], and all file handles
/// created by a session will be cached.
#[derive(Debug)]
pub struct FileCache {
    inner: Cache<FileKey, Footer, DefaultHashBuilder>,
}

impl FileCache {
    pub fn new() -> Self {
        let inner = Cache::builder()
            .max_capacity(64u64 * (1 << 20))
            .eviction_listener(|k: Arc<FileKey>, _v: Footer, cause| {
                log::trace!("Removed {k:?} due to {cause:?}");
            })
            .weigher(|_k, footer| u32::try_from(estimate_layout_size(footer)).unwrap_or(u32::MAX))
            .build_with_hasher(DefaultHashBuilder::default());

        Self { inner }
    }

    pub fn get_footer(&self, file_key: &FileKey) -> Option<Footer> {
        self.inner.get(file_key)
    }

    pub fn put_footer(&self, file_key: FileKey, footer: Footer) {
        self.inner.insert(file_key, footer)
    }
}

// TODO(joe): unify with the df impl
/// Approximate the in-memory size of a layout
fn estimate_layout_size(footer: &Footer) -> usize {
    let segments_size = footer.segment_map().len() * size_of::<SegmentSpec>();
    let stats_size = footer
        .statistics()
        .iter()
        .map(|v| {
            v.iter()
                .map(|_| size_of::<Stat>() + size_of::<Precision<ScalarValue>>())
                .sum::<usize>()
        })
        .sum::<usize>();

    let root_layout = footer.layout();
    let layout_size = size_of::<DType>()
        + root_layout.metadata().len()
        + root_layout.segment_ids().len() * size_of::<SegmentId>();

    segments_size + stats_size + layout_size
}

// Attach various file IO methods to the session when the `files` feature is enabled
// in compilation.
impl VortexSession {
    /// Open a Vortex file on the local file system, blocking the current thread
    /// until it completes.
    pub fn open_blocking(&self, path: impl AsRef<Path>) -> VortexResult<VortexFile> {
        VortexOpenOptions::file(self.arrays().clone(), self.layouts().clone()).open_blocking(path)
    }

    pub async fn open(&self, path: impl AsRef<Path>) -> VortexResult<VortexFile> {
        VortexOpenOptions::file(self.arrays().clone(), self.layouts().clone())
            .open(path)
            .await
    }

    #[cfg(feature = "object_store")]
    pub async fn open_object_store<Store: object_store::ObjectStore>(
        &self,
        store: &Store,
        path: &str,
    ) -> VortexResult<VortexFile> {
        VortexOpenOptions::file(self.arrays().clone(), self.layouts().clone())
            .open_object_store(path)
    }
}
