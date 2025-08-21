// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan nodes represent the logical structure of a pipeline.

pub mod binary_bool;
pub mod compare;
pub mod constant;
pub mod primitive;
pub mod scalar_compare;

use std::any::Any;
use std::fmt::Debug;
use std::rc::Rc;

use dyn_hash::DynHash;
use vortex_error::VortexResult;

use crate::Kernel;
use crate::types::VType;
use crate::vector::VectorId;

// TODO: clean up this diagram
// compare(_, _) <-  for <- bitpacked
//               <- 12

// !--> reduce_child(compare(_, _), for <- bitpacked) -->
// --> reduce_child(compare(_, _), 12) -->

// compare_single[12](_) <- for(10) <- bitpacked

// compare_single[2](_) <- bitpacked

/// An operator represents a node in a logical query plan.
pub trait Operator: Debug + DynHash + 'static {
    fn as_any(&self) -> &dyn Any;

    /// The output [`VType`] of this operator.
    fn vtype(&self) -> VType;

    /// The children of this operator.
    fn children(&self) -> &[Rc<dyn Operator>];

    fn with_children(&self, children: Vec<Rc<dyn Operator>>) -> Rc<dyn Operator>;

    /// Whether this operator works by mutating its first child in-place.
    ///
    /// If `true`, the operator is invoked with the first child's input data passed via the
    /// mutable output view. The node is expected to mutate this data in-place.
    fn in_place(&self) -> bool {
        false
    }

    /// Create a kernel for this operator
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>>;

    //TODO: fixme
    /// Given a set of reduced children, try and reduce the current node.
    /// If Keep is returned then the children of this node as still updated.
    fn reduce_children(&self, children: &[Rc<dyn Operator>]) -> Option<Rc<dyn Operator>> {
        None
    }

    /// Given a reduced parent, try and reduce the current node.
    /// If `Replace` is returned then  the parent node and this node and replaced by the returned node.
    fn reduce_parent(&self, parent: Rc<dyn Operator>) -> Option<Rc<dyn Operator>> {
        None
    }
}

dyn_hash::hash_trait_object!(Operator);

/// The context used when binding an operator for execution.
pub trait BindContext {
    fn children(&self) -> &[VectorId];
}
