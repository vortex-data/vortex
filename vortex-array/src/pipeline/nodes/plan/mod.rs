// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan nodes represent the logical structure of a pipeline.

pub mod source;

use crate::pipeline::Operator;
use crate::pipeline::types::VType;
use crate::pipeline::vector::VectorId;
use dyn_hash::DynHash;
use vortex_error::VortexResult;

pub trait PlanNode: DynHash {
    /// The output type of this node
    fn output_type(&self) -> VType;

    /// The child nodes of this plan node
    fn children(&self) -> &[Box<dyn PlanNode>];

    /// Whether the node operates by mutating its first child in-place.
    ///
    /// If `true`, the node is invoked with the first child's input data passed via the mutable
    /// output view. The node is expected to mutate this data in-place.
    fn in_place(&self) -> bool {
        false
    }

    /// Create the execution operator for this node
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Operator>>;
}

dyn_hash::hash_trait_object!(PlanNode);

/// The context used when binding a node for execution.
pub trait BindContext {
    fn input_ids(&self) -> &[VectorId];
}
