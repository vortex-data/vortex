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
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
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

    /// Coerce the input type for this aggregate function.
    ///
    /// This is optionally used by Vortex users when performing type coercion over a Vortex
    /// expression. The default implementation returns the input type unchanged.
    fn coerce_args(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        let _ = options;
        Ok(input_dtype.clone())
    }

    /// The return [`DType`] of the aggregate.
    ///
    /// Returns `None` if the aggregate function cannot be applied to the input dtype.
    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType>;

    /// DType of the intermediate partial accumulator state.
    ///
    /// Use a struct dtype when multiple fields are needed
    /// (e.g., Mean: `Struct { sum: f64, count: u64 }`).
    ///
    /// Returns `None` if the aggregate function cannot be applied to the input dtype.
    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType>;

    /// Return the partial accumulator state for an empty group.
    fn empty_partial(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial>;

    /// Combine partial scalar state into the accumulator.
    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()>;

    /// Convert the partial state into a partial scalar.
    ///
    /// The returned scalar must have the same DType as specified by `partial_dtype` for the
    /// options and input dtype used to construct the state.
    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar>;

    /// Reset the state of the accumulator to an empty group.
    fn reset(&self, partial: &mut Self::Partial);

    /// Is the partial accumulator state is "saturated", i.e. has it reached a state where the
    /// final result is fully determined.
    fn is_saturated(&self, state: &Self::Partial) -> bool;

    /// Try to derive a partial scalar from the batch's cached statistics, before any
    /// kernel dispatch or canonicalization.
    ///
    /// Returns `Some(partial_scalar)` if the answer can be read directly from `batch.statistics()`,
    /// otherwise `Ok(None)` to fall through to the rest of dispatch. The returned scalar must
    /// have the dtype reported by `partial_dtype`.
    ///
    /// This is the single place stats-based shortcuts live; encoding kernels must not consult
    /// stats themselves. Runs first so that an upstream producer who pre-populates the relevant
    /// stat (e.g. a layout reader hydrating `Stat::UncompressedSizeInBytes` from file metadata)
    /// can skip both kernel dispatch and decode.
    ///
    /// TODO: this hook may be removed once `ArrayStats` stores aggregate partials internally —
    /// at that point stat-driven shortcuts can be resolved automatically by the dispatch layer
    /// without each aggregate vtable opting in.
    fn try_partial_from_stats(&self, _batch: &ArrayRef) -> VortexResult<Option<Scalar>> {
        Ok(None)
    }

    /// Try to accumulate the raw array before decompression.
    ///
    /// Returns `true` if the array was handled, `false` to fall through to
    /// the default kernel dispatch and canonicalization path.
    ///
    /// This is useful for aggregates that only depend on array metadata (e.g., validity)
    /// rather than the encoded data, avoiding unnecessary decompression.
    fn try_accumulate(
        &self,
        _state: &mut Self::Partial,
        _batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        Ok(false)
    }

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
    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar>;
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
