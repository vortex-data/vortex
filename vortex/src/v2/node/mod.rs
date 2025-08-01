// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod primitive;

use crate::experiment::vector::Vector;
use std::pin::Pin;
use std::task::{Context, Poll};
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

/// A node represents a unit of computation in the Vortex system that is capable of
/// exporting canonicalized data into a mutable vector.
pub trait Node {
    fn bind(&self, ctx: BindContext) -> VortexResult<Box<dyn Evaluation>>;
}

pub struct BindContext {
    pub metadata: Box<dyn MetadataSource>,
    pub buffers: Box<dyn BufferSource>,
}

pub trait MetadataSource {
    /// Retrieves metadata for a given node ID.
    fn get(&self, node_id: NodeId) -> VortexResult<ByteBuffer>;
}

pub trait BufferSource {
    /// Retrieves a buffer for a given node ID.
    ///
    /// Returns `None` if the buffer is not yet available.
    fn get(&self, node_id: NodeId, buffer_id: BufferId) -> VortexResult<Option<ByteBuffer>>;
}

/// A unique identifier for a node.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct NodeId(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct BufferId(usize);

/// An instantiated evaluation of a pipeline.
pub trait Evaluation {
    /// Attempts to perform a single step of the evaluation, writing data to the output vector.
    /// Returns `Poll::Done` if the evaluation is complete, or `Poll::Pending` if buffers are
    /// required to continue.
    fn step(
        &mut self,
        ctx: &dyn EvaluationContext,
        selected: &Mask,
        defined: &Mask,
        out: &mut Vector,
    ) -> VortexResult<Poll<()>>;
}

pub trait EvaluationContext {
    fn get_buffer(&self, buffer_id: BufferId) -> VortexResult<Option<ByteBuffer>>;
}
