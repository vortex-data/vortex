// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::functions::{ExecutionCtx, FunctionId, Signature};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_utils::dyn_traits::{DynEq, DynHash};
use vortex_vector::Vector;

/// A reference-counted pointer to a scalar function.
pub type ScalarFnRef = Arc<dyn ScalarFn>;

/// An instance of a scalar function, including any options required for its execution.
pub trait ScalarFn: 'static + Send + Sync + Debug + DynEq + DynHash {
    /// Returns the unique identifier for this scalar function.
    fn id(&self) -> FunctionId;

    /// Returns signature information about this function.
    fn signature(&self) -> &dyn Signature;

    /// Computes the return [`DType`] given the argument types and function options.
    fn return_dtype(&self, arg_types: &[DType]) -> VortexResult<DType>;

    /// Binds the function for execution over a specific set of inputs.
    // TODO(ngates): in the future, we should return a kernel as a node in a physical plan and
    //  continue to run further cost-based optimizations prior to execution.
    fn execute(&self, _ctx: &ExecutionCtx) -> VortexResult<Vector> {
        vortex_bail!("Execution is not supported for {}", self.id())
    }
}

impl Hash for dyn ScalarFn + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}
impl PartialEq for dyn ScalarFn + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other)
    }
}
impl Eq for dyn ScalarFn + '_ {}

/// A reference-counted pointer to a scalar function codec.
pub type ScalarFnCodecRef = Arc<dyn ScalarFnCodec>;

/// A codec for serializing and deserializing scalar functions.
pub trait ScalarFnCodec: 'static + Send + Sync + Debug {
    /// The identifier for this scalar function.
    fn id(&self) -> FunctionId;

    /// Serialize the given scalar function into a byte vector.
    ///
    /// The `id` of the function should not be serialized as part of this method.
    ///
    /// If the function does not support serialization, return `Ok(None)`.
    fn serialize(&self, function: &ScalarFnRef) -> VortexResult<Option<Vec<u8>>> {
        _ = function;
        Ok(None)
    }

    /// Deserialize a scalar function from the given byte slice.
    fn deserialize(&self, bytes: &[u8]) -> VortexResult<ScalarFnRef>;
}
