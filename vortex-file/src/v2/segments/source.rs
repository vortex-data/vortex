use std::sync::Arc;

use vortex_layout::segments::AsyncSegmentReader;

/// A source of segment data within a contiguous file.
pub trait FileSegmentSource {
    /// Returns an [`AsyncSegmentReader`] for the segment source that will be used to construct
    /// a [`vortex_layout::LayoutReader`].
    fn reader(&self) -> Arc<dyn AsyncSegmentReader + 'static>;
}
