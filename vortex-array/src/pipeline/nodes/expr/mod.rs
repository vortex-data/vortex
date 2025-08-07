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

pub trait Expression: Debug + DynHash {
    /// The output [`VType`] of this expression.
    fn vtype(&self) -> VType;

    /// The child nodes of this expression.
    fn children(&self) -> &[Box<dyn Expression>];

    /// Whether the expression operates by mutating its first child in-place.
    ///
    /// If `true`, the expression is invoked with the first child's input data passed via the
    /// mutable output view. The node is expected to mutate this data in-place.
    fn in_place(&self) -> bool {
        false
    }

    /// Create a kernel for this expression
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>>;
}

dyn_hash::hash_trait_object!(Expression);

/// The context used when binding a node for execution.
pub trait BindContext {
    fn children(&self) -> &[VectorId];
}
