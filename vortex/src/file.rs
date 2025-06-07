use std::path::Path;
use std::sync::Arc;

use moka::sync::Cache;
use vortex_array::stats::{Precision, Stat};
use vortex_dtype::DType;
use vortex_error::VortexResult;
pub use vortex_file::*;
use vortex_layout::segments::SegmentId;
use vortex_scalar::ScalarValue;

use crate::session::VortexSession;
use crate::utils::aliases::DefaultHashBuilder;

/// Cache key for a [`VortexFile`].
#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub struct FileKey {
    // TODO: support last modified ts.
    pub location: String,
}

#[cfg(feature = "object_store")]
impl From<object_store::path::Path> for FileKey {
    fn from(path: object_store::path::Path) -> Self {
        Self {
            location: path.to_string(),
        }
    }
}

/// Cache of Vortex file [`Footer`]s. When reading the same file from multiple threads
/// or the same file multiple times, the cache will save footer contents the first time a file
/// is read to avoid unnecessary refetching.
#[derive(Debug)]
pub struct FooterCache {
    inner: Cache<FileKey, Footer, DefaultHashBuilder>,
}

macro_rules! assert_send_sync_static {
    ($typ:ty) => {
        const _: () = {
            const fn check_send_sync_static<T: Send + Sync + 'static>() {}
            check_send_sync_static::<$typ>();
        };
    };
}

assert_send_sync_static!(FooterCache);

/// Default maximum size of the footer cache in bytes.
///
/// Defaults to 64MiB
pub const MAX_FOOTER_CACHE_BYTES: usize = 64 << 20;

impl Default for FooterCache {
    fn default() -> Self {
        Self::new(MAX_FOOTER_CACHE_BYTES)
    }
}

impl FooterCache {
    /// Construct a new empty footer cache with default 64MiB of space reserved for footers
    pub fn new(max_bytes: usize) -> Self {
        let inner = Cache::builder()
            .max_capacity(max_bytes as u64)
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

// Provide an accessor for the footer cache so that spawned files are able to have access to it.
impl VortexSession {
    /// The footer cache for the attached session. The cache can be cloned and owned handles can
    /// be shared across threads safely.
    pub fn footer_cache(&self) -> &FooterCache {
        &self.footer_cache
    }
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
    pub async fn open_object_store(
        &self,
        store: &Arc<dyn object_store::ObjectStore>,
        path: &str,
    ) -> VortexResult<VortexFile> {
        VortexOpenOptions::file(self.arrays().clone(), self.layouts().clone())
            .open_object_store(store, path)
            .await
    }
}
