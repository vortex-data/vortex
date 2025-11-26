// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod funcs;
mod session;

use arcref::ArcRef;
use std::fmt::Debug;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::Vector;

pub type FunctionId = ArcRef<str>;
pub type ChildName = ArcRef<str>;

pub type ScalarFunctionRef = Arc<dyn ScalarFunction>;

/// Dynamic trait for defining scalar functions in Vortex.
pub trait ScalarFunction: 'static + Send + Sync + Debug {
    /// Returns the unique identifier for this function.
    fn id(&self) -> FunctionId;

    /// Returns the child name for this function instance.
    fn child_name(&self, child_idx: usize) -> ChildName;

    /// Returns the arity (number of arguments) for this function instance.
    // TODO(ngates): evolve this API to include more information about the function signature
    fn arity(&self) -> usize;

    /// Computes the return [`DType`] given the argument types.
    fn return_dtype(&self, arg_types: &[DType]) -> VortexResult<DType>;

    /// Executes the function with the given options.
    // TODO(ngates): this should take an execution context and likely return some sort of physical
    //  plan node in the future?
    fn execute(&self, inputs: &[Vector]) -> VortexResult<Vector>;
}
