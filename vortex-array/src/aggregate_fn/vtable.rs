// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::scalar::Scalar;

/// Defines the interface for aggregate function vtables.
///
/// This trait is non-object-safe and allows the implementer to make use of associated types
/// for improved type safety, while allowing Vortex to enforce runtime checks on the inputs and
/// outputs of each function.
///
/// The [`AggregateFnVTable`] trait should be implemented for a struct that holds global data across
/// all instances of the aggregate. In almost all cases, this struct will be an empty unit
/// struct, since most aggregates do not require any global state.
pub trait AggregateFnVTable: 'static + Sized + Clone + Send + Sync {
    /// Options for this aggregate function.
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// The accumulator state for a group.
    type GroupState: 'static + Send;

    /// Returns the ID of the aggregate function vtable.
    fn id(&self) -> AggregateFnId;

    /// Serialize the options for this aggregate function.
    ///
    /// Should return `Ok(None)` if the function is not serializable, and `Ok(vec![])` if it is
    /// serializable but has no metadata.
    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        _ = options;
        Ok(None)
    }

    /// Deserialize the options of this aggregate function.
    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_bail!("Aggregate function {} is not deserializable", self.id());
    }

    /// The return [`DType`] of the aggregate.
    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType>;

    /// DType of the intermediate accumulator state.
    ///
    /// Use a struct dtype when multiple fields are needed
    /// (e.g., Mean: `Struct { sum: f64, count: u64 }`).
    fn state_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType>;

    /// Return accumulator state for an empty group.
    fn state_new(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::GroupState>;

    /// Merge partial scalar state into the accumulator.
    fn state_merge(&self, state: &mut Self::GroupState, partial: Scalar) -> VortexResult<()>;

    /// Flush the partial aggregate for the given accumulator state.
    ///
    /// The returned scalar must have the same DType as specified by `state_dtype` for the
    /// options and input dtype used to construct the state.
    ///
    /// The internal state of the accumulator is reset to the empty state after flushing.
    fn state_flush(&self, state: &mut Self::GroupState) -> VortexResult<Scalar>;

    /// Is the accumulator state "saturated", i.e. has it reached a state where the final result
    /// is fully determined.
    fn state_is_saturated(&self, state: &Self::GroupState) -> bool;

    /// Accumulate a new canonical array into the accumulator state.
    fn state_accumulate(
        &self,
        state: &mut Self::GroupState,
        batch: &Canonical,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    /// Finalize an array of accumulator states into an array of aggregate results.
    ///
    /// The provides `states` array has dtype as specified by `state_dtype`, the result array
    /// must have dtype as specified by `return_dtype`.
    fn finalize(&self, states: ArrayRef) -> VortexResult<ArrayRef>;

    /// Finalize a scalar accumulator state into an aggregate result.
    ///
    /// The provided `state` has dtype as specified by `state_dtype`, the result scalar must have
    /// dtype as specified by `return_dtype`.
    fn finalize_scalar(&self, state: Scalar) -> VortexResult<Scalar> {
        let array = ConstantArray::new(state, 1).into_array();
        let result = self.finalize(array)?;
        result.scalar_at(0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EmptyOptions;
impl Display for EmptyOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

/// Factory functions for aggregate vtables.
pub trait AggregateFnVTableExt: AggregateFnVTable {
    /// Bind this vtable with the given options into an [`AggregateFnRef`].
    fn bind(&self, options: Self::Options) -> AggregateFnRef {
        AggregateFn::new(self.clone(), options).erased()
    }
}
impl<V: AggregateFnVTable> AggregateFnVTableExt for V {}
