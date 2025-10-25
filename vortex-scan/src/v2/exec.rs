// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_mask::Mask;

pub type StreamExecRef = Box<dyn StreamExec>;

/// A physical node in the Vortex stream processing graph.
///
/// Streams emit arrays of data in response to execution requests from upstream nodes that provide
/// a selection mask.
#[async_trait]
pub trait StreamExec: 'static + Send {
    /// Returns the size of the next batch that this node can efficiently emit.
    ///
    /// This is a hint to upstream nodes to help with batching. It is typically expected that
    /// the mask passed to `execute` will have a length equal to this size, but not guaranteed.
    fn next_batch_size(&self) -> usize;

    /// Emits the next batch of rows covered by the input mask. The length of the returned array
    /// must be equal to the number of true values in the input mask.
    async fn next_batch(&mut self, mask: &Mask) -> VortexResult<ArrayRef>;
}
