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
use crate::Columnar;
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

    /// The partial accumulator state for a single group.
    type Partial: 'static + Send;

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

    /// DType of the intermediate partial accumulator state.
    ///
    /// Use a struct dtype when multiple fields are needed
    /// (e.g., Mean: `Struct { sum: f64, count: u64 }`).
    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType>;

    /// Return the partial accumulator state for an empty group.
    fn empty_partial(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial>;

    /// Combine partial scalar state into the accumulator.
    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()>;

    /// Flush the partial aggregate for the given accumulator state.
    ///
    /// The returned scalar must have the same DType as specified by `state_dtype` for the
    /// options and input dtype used to construct the state.
    ///
    /// The internal state of the accumulator is reset to the empty state after flushing.
    fn flush(&self, partial: &mut Self::Partial) -> VortexResult<Scalar>;

    /// Is the partial accumulator state is "saturated", i.e. has it reached a state where the
    /// final result is fully determined.
    fn is_saturated(&self, state: &Self::Partial) -> bool;

    /// Accumulate a new canonical array into the accumulator state.
    fn accumulate(
        &self,
        state: &mut Self::Partial,
        batch: &Columnar,
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
