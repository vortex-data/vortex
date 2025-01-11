use async_trait::async_trait;

/// A source of segment data.
#[async_trait]
pub trait SegmentSource {}
