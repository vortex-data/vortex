use std::fmt::Debug;

use vortex_array::{ArrayRegistry, ArrayRegistryBuilder, EncodingRef};
use vortex_file::ArrayRegistryExt;
use vortex_layout::{LayoutEncodingRef, LayoutRegistry, LayoutRegistryBuilder, LayoutRegistryExt};
use vortex_metrics::VortexMetrics;

pub struct VortexSessionBuilder {
    arrays: ArrayRegistryBuilder,
    layouts: LayoutRegistryBuilder,
    metrics: VortexMetrics,
}

impl Default for VortexSessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VortexSessionBuilder {
    /// Make a new builder to configure a [`VortexSession`].
    pub fn new() -> Self {
        Self {
            arrays: ArrayRegistryBuilder::new(),
            layouts: LayoutRegistryBuilder::new(),
            metrics: VortexMetrics::default(),
        }
    }

    /// Configure the encodings builder as we see fit
    pub fn with_encodings(mut self, arrays: ArrayRegistryBuilder) -> Self {
        self.arrays = arrays;
        self
    }

    pub fn with_layouts(mut self, layouts: LayoutRegistryBuilder) -> Self {
        self.layouts = layouts;
        self
    }

    pub fn with_encoding(mut self, encoding: EncodingRef) -> Self {
        self.arrays = self.arrays.register(encoding);
        self
    }

    pub fn with_layout(mut self, layout: LayoutEncodingRef) -> Self {
        self.layouts = self.layouts.register(layout);
        self
    }

    pub fn with_metrics(mut self, metrics: VortexMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    // TODO(aduffy): with_extension_type once the ExtVTable stuff merges.
    pub fn build(self) -> VortexSession {
        VortexSession {
            arrays: self.arrays.build(),
            layouts: self.layouts.build(),
            metrics: self.metrics,
            // TODO(aduffy): add config to setup the cache size.
            #[cfg(feature = "files")]
            footer_cache: crate::file::FooterCache::default(),
        }
    }
}

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
}
