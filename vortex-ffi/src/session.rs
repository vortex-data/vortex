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

// Custom session wrapper to handle runtime lifecycle
#[allow(non_camel_case_types)]
pub(crate) struct vx_session(VortexSession);

#[allow(dead_code)]
impl vx_session {
    /// Wrap an owned object into a raw pointer.
    pub(crate) fn new(obj: Box<VortexSession>) -> *mut vx_session {
        Box::into_raw(obj).cast()
    }

    /// Wrap a borrowed object into a raw pointer.
    pub(crate) fn new_ref(obj: &VortexSession) -> *const vx_session {
        obj as *const VortexSession as *const vx_session
    }

    /// Extract a borrowed reference from a const pointer.
    pub(crate) fn as_ref<'a>(ptr: *const vx_session) -> &'a VortexSession {
        use vortex::error::VortexExpect;
        &unsafe { ptr.as_ref() }.vortex_expect("null pointer").0
    }

    /// Extract a borrowed mutable reference from a mut pointer.
    pub(crate) fn as_mut<'a>(ptr: *mut vx_session) -> &'a mut VortexSession {
        use vortex::error::VortexExpect;
        &mut unsafe { ptr.as_mut() }.vortex_expect("null pointer").0
    }

    /// Extract an owned reference.
    pub(crate) fn into_box(ptr: *mut vx_session) -> Box<VortexSession> {
        if ptr.is_null() {
            vortex::error::vortex_panic!("null pointer");
        }
        unsafe { Box::from_raw(ptr.cast::<VortexSession>()) }
    }
}

/// Free an owned [`vx_session`] object and handle runtime lifecycle.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_session_free(ptr: *mut vx_session) {
    if ptr.is_null() {
        vortex::error::vortex_panic!("null pointer");
    }

    // Free the session - the Drop trait will handle runtime lifecycle automatically
    drop(unsafe { Box::from_raw(ptr.cast::<VortexSession>()) });
}

/// Create a new Vortex session.
///
/// The caller is responsible for freeing the session with [`vx_session_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_session_new() -> *mut vx_session {
    vx_session::new(Box::new(VortexSession::new()))
}

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
        let runtime = crate::get_session_runtime();

        Self {
            file_cache,
            _runtime: runtime,
        }
    }

    pub fn get_footer(&self, file_key: &FileKey) -> Option<Footer> {
        self.file_cache.get(file_key)
    }

    pub fn put_footer(&self, file_key: FileKey, footer: Footer) {
        self.file_cache.insert(file_key, footer)
    }
}

impl Drop for VortexSession {
    fn drop(&mut self) {
        // When the session is dropped, try to shutdown the runtime if no other sessions hold references
        crate::try_shutdown_runtime();
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
