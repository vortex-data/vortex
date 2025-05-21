use std::sync::Arc;

use moka::sync::Cache;
use vortex::aliases::DefaultHashBuilder;
use vortex::dtype::DType;
use vortex::file::{Footer, SegmentSpec};
use vortex::layout::segments::SegmentId;
use vortex::scalar::ScalarValue;
use vortex::stats::{Precision, Stat};

/// An object that stores registries and caches.
/// This should if possible be reused between queries in ann interactive session.
#[allow(non_camel_case_types)]
pub struct vx_session {
    pub inner: Arc<VortexSession>,
}

/// Create a session to be used for the lifetime of an interactive session.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_session_create() -> *mut vx_session {
    Box::into_raw(Box::new(vx_session {
        inner: Arc::new(VortexSession::new()),
    }))
}

/// Free a session
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_session_free(session: *mut vx_session) {
    drop(unsafe { Box::from_raw(session) })
}

pub struct VortexSession {
    file_cache: Cache<FileKey, Footer, DefaultHashBuilder>,
}

/// Cache key for a [`VortexFile`].
#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub struct FileKey {
    // TODO: support last modified ts.
    pub location: String,
}

impl VortexSession {
    pub fn new() -> Self {
        let file_cache = Cache::builder()
            .max_capacity(64u64 * (1 << 20))
            .eviction_listener(|k: Arc<FileKey>, _v: Footer, cause| {
                log::trace!("Removed {k:?} due to {cause:?}");
            })
            .weigher(|_k, footer| u32::try_from(estimate_layout_size(footer)).unwrap_or(u32::MAX))
            .build_with_hasher(DefaultHashBuilder::default());

        Self { file_cache }
    }

    pub fn get_footer(&self, file_key: &FileKey) -> Option<Footer> {
        self.file_cache.get(file_key)
    }

    pub fn put_footer(&self, file_key: FileKey, footer: Footer) {
        self.file_cache.insert(file_key, footer)
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
