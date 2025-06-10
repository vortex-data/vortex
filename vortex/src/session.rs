use std::fmt::Debug;

use vortex_array::{ArrayRegistry, EncodingRef};
use vortex_file::ArrayRegistryExt;
use vortex_layout::{LayoutEncodingRef, LayoutRegistry, LayoutRegistryExt};
use vortex_metrics::VortexMetrics;

/// A Vortex session encapsulates the set of extensible arrays, layouts, compute functions, dtypes,
/// etc. that are available for use in a given context.
///
/// It is also the entry-point passed to dynamic libraries to initialize Vortex plugins.
#[derive(Debug)]
pub struct VortexSession {
    arrays: ArrayRegistry,
    layouts: LayoutRegistry,
    metrics: VortexMetrics,
    #[cfg(feature = "files")]
    pub(crate) footer_cache: crate::file::FooterCache,
}

impl Default for VortexSession {
    fn default() -> Self {
        Self {
            arrays: ArrayRegistry::full(),
            layouts: LayoutRegistry::full(),
            metrics: VortexMetrics::default(),
            #[cfg(feature = "files")]
            footer_cache: crate::file::FooterCache::default(),
        }
    }
}

impl VortexSession {
    /// Create a new VortexSession with all builtin encodings and layouts registered.
    ///
    /// This is also equivalent to the `Default` impl for VortexSession.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new VortexSession with no configured encodings or layouts.
    ///
    /// This is useful for building a session where callers want to selectively enable specific
    /// encodings and layouts.
    pub fn empty() -> Self {
        Self {
            arrays: ArrayRegistry::empty(),
            layouts: LayoutRegistry::empty(),
            metrics: VortexMetrics::default(),
            #[cfg(feature = "files")]
            footer_cache: crate::file::FooterCache::default(),
        }
    }

    /// The array registry for the session.
    pub fn arrays(&self) -> &ArrayRegistry {
        &self.arrays
    }

    /// The layout registry for the session.
    pub fn layouts(&self) -> &LayoutRegistry {
        &self.layouts
    }

    /// The metrics registry for the session.
    pub fn metrics(&self) -> &VortexMetrics {
        &self.metrics
    }

    /// Register a new array encoding in the session.
    ///
    /// The encoding will become available for file reading/writing.
    pub fn register_encoding(&self, encoding: EncodingRef) -> &Self {
        self.arrays.register(encoding);
        self
    }

    /// Register a new layout encoding in the session.
    ///
    /// This allows for runtime-pluggability of new layout refs.s
    pub fn register_layout(&self, layout: LayoutEncodingRef) -> &Self {
        self.layouts.register(layout);
        self
    }
}
