//! ID types local to the v2 (lowering / pipeline) plan API.
//!
//! `DomainId` and `RelationId` are reused from the existing
//! [`crate::graph::ids`] module; only the v2-specific identifiers
//! live here.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineId(usize);

impl PipelineId {
    pub const fn from_index(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineBarrier(usize);

impl PipelineBarrier {
    pub(crate) const fn from_index(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}
