// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan nodes represent the logical structure of a pipeline.

pub mod compare;
pub mod primitive;

use crate::pipeline::Kernel;
use crate::pipeline::types::VType;
use crate::pipeline::vector::VectorId;
use dyn_hash::DynHash;
use std::fmt::Debug;
use vortex_error::VortexResult;

/// An operator represents a node in a logical query plan.
pub trait Operator: Debug + DynHash {
    /// The output [`VType`] of this operator.
    fn vtype(&self) -> VType;

    /// The children of this operator.
    fn children(&self) -> &[Box<dyn Operator>];

    /// Whether this operator works by mutating its first child in-place.
    ///
    /// If `true`, the operator is invoked with the first child's input data passed via the
    /// mutable output view. The node is expected to mutate this data in-place.
    fn in_place(&self) -> bool {
        false
    }

    /// Create a kernel for this operator
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>>;
}

dyn_hash::hash_trait_object!(Operator);

/// The context used when binding an operator for execution.
pub trait BindContext {
    fn children(&self) -> &[VectorId];
}
