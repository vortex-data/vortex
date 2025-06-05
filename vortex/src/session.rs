use std::fmt::Debug;

use vortex_array::ArrayRegistry;
use vortex_file::ArrayRegistryExt;
use vortex_layout::{LayoutRegistry, LayoutRegistryExt};
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
            footer_cache: crate::file::FooterCache::new(),
        }
    }
}

impl VortexSession {
    /// Returns the array registry for this session.
    pub fn arrays(&self) -> &ArrayRegistry {
        &self.arrays
    }

    /// Returns the layout registry for this session.
    pub fn layouts(&self) -> &LayoutRegistry {
        &self.layouts
    }

    /// Returns the metrics for this session.
    pub fn metrics(&self) -> &VortexMetrics {
        &self.metrics
    }
}
