// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use moka::sync::Cache;
use vortex::dtype::DType;
use vortex::file::{Footer, SegmentSpec};
use vortex::layout::segments::SegmentId;
use vortex::scalar::ScalarValue;
use vortex::stats::{Precision, Stat};
use vortex::utils::aliases::DefaultHashBuilder;

use crate::box_wrapper;

box_wrapper!(
    /// A handle to a Vortex session.
    VortexSession,
    vx_session
);

/// Create a new Vortex session.
///
/// The caller is responsible for freeing the session with [`vx_session_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_session_new() -> *mut vx_session {
    vx_session::new(Box::new(VortexSession::new()))
}

/// A Vortex session stores registries of extensible types, various caches, and other
/// top-level configuration.
///
/// Extensible types include array encodings, layouts, extension dtypes, compute functions, etc.
///
/// Multiple sessions may be created in a single process, and individual arrays are not tied to a
/// specific session.
///
/// The session holds a reference to a shared tokio runtime. When the last session is dropped,
/// the runtime may be shut down by calling `crate::try_shutdown_runtime()`.
pub struct VortexSession {
    file_cache: Cache<FileKey, Footer, DefaultHashBuilder>,
    _runtime: Arc<tokio::runtime::Runtime>,
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

        // Get a runtime reference that will be held for the lifetime of this session
        let _runtime = crate::get_runtime();

        Self {
            file_cache,
            _runtime,
        }
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
