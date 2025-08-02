// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bitpacked;
mod primitive;

use crate::experiment::vector::Vector;
use std::ops::Deref;
use std::pin::Pin;
use std::task::{Context, Poll};
use vortex_array::stats::StatsSet;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

/// An operator represents a unit of computation in the Vortex system that is capable of
/// exporting canonicalized data into a mutable vector.
pub trait Operator {
    /// [`DType`] and length of the node are passed down in the bind context.
    fn bind(&self, ctx: &BindContext) -> VortexResult<Box<dyn Evaluation>>;
}

/// Context required for binding a node.
///
/// During the bind phase, context is passed down from parent nodes to child nodes.
pub struct BindContext<'a> {
    pub len: usize,
    pub dtype: &'a DType,
    pub metadata: &'a [u8],
    pub stats: Option<&'a StatsSet>,
}

pub trait MetadataSource {
    /// Retrieves metadata for a given node ID.
    fn get(&self, node_id: NodeId) -> VortexResult<ByteBuffer>;
}

pub trait BufferSource {
    /// Retrieves the requested buffer.
    fn get(&self, buffer_id: BufferId) -> VortexResult<Option<ByteBuffer>>;
}

/// A unique identifier for a node.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct NodeId(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct BufferId(usize);

impl Deref for BufferId {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// An instantiated evaluation of a pipeline.
pub trait Evaluation {
    /// Seek the evaluation to a specific chunk offset.
    /// The resulting row offset should be `idx * N`, where `N` is the number of elements in
    /// a chunk.
    ///
    // NOTE(ngates): we have a separate seek function since it can often be more efficient for
    //  arrays to assume they will be evaluated in order, e.g. run-length would have to do a full
    //  binary search of the ends in each step if we passed an offset that way.
    fn seek(&mut self, idx: usize) -> VortexResult<()>;

    /// Attempts to perform a single step of the evaluation, writing data to the output vector.
    /// Returns `Poll::Done` if the evaluation is complete, or `Poll::Pending` if buffers are
    /// required to continue.
    fn step(
        &mut self,
        ctx: &dyn EvaluationContext,
        selected: &Mask,
        defined: &Mask,
        out: &mut Vector,
    ) -> Poll<VortexResult<()>>;
}

pub trait EvaluationContext {
    fn buffer(&self, buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>>;
}
