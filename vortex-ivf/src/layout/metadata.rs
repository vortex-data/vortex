// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialized metadata for [`IvfLayout`](crate::layout::IvfLayout).

/// Serialized metadata for an IVF layout.
///
/// The chunked data child already records cluster boundaries via its chunk row offsets,
/// so we only need to store the high-level index parameters here.
#[derive(prost::Message)]
pub struct IvfLayoutMetadata {
    /// Vector dimension.
    #[prost(uint32, tag = "1")]
    pub dim: u32,
    /// Default number of clusters to probe.
    #[prost(uint32, tag = "2")]
    pub nprobes: u32,
    /// Number of clusters (K).
    #[prost(uint32, tag = "3")]
    pub num_clusters: u32,
}

impl IvfLayoutMetadata {
    pub fn new(dim: u32, nprobes: u32, num_clusters: u32) -> Self {
        Self {
            dim,
            nprobes,
            num_clusters,
        }
    }
}
