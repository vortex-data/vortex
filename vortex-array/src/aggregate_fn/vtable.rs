// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnSatisfaction;
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

    /// Return whether this stored aggregate can satisfy `requested`.
    ///
    /// The default implementation only treats exactly equal aggregate functions as satisfying the
    /// request. Approximate pruning aggregates can override this to expose looser-but-sound bounds.
    fn can_satisfy(
        &self,
        options: &Self::Options,
        requested: &AggregateFnRef,
    ) -> AggregateFnSatisfaction {
        if requested
            .as_opt::<Self>()
            .is_some_and(|other| other == options)
        {
            AggregateFnSatisfaction::Exact
        } else {
            AggregateFnSatisfaction::No
        }
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

/// Options for aggregate functions over primitive numeric inputs, controlling how NaN values in
/// floating-point arrays are handled.
///
/// When `skip_nans` is `true` (the default), NaN values are treated as missing: they contribute
/// nothing to `sum`/`min`/`max`/`mean` and are excluded from `count`.
///
/// When `skip_nans` is `false`, NaN values participate in the aggregate: `count` includes them,
/// while any NaN value poisons the result of `sum`/`min`/`max`/`mean` to NaN.
///
/// The option has no effect on non-float inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumericalAggregateOpts {
    /// Whether NaN values are skipped (treated as missing) during aggregation.
    pub skip_nans: bool,
}

impl NumericalAggregateOpts {
    /// Options that skip NaN values, treating them as missing during aggregation.
    ///
    /// This is the default configuration; see [`NumericalAggregateOpts::include_nans`] for the
    /// NaN-including variant.
    pub const fn skip_nans() -> Self {
        Self { skip_nans: true }
    }

    /// Options that include NaN values in the aggregate: `count` counts them, while any NaN
    /// poisons the result of `sum`/`min`/`max`/`mean` to NaN.
    ///
    /// See [`NumericalAggregateOpts::skip_nans`] for the default NaN-skipping variant.
    pub const fn include_nans() -> Self {
        Self { skip_nans: false }
    }

    /// Serialize these options to protobuf-encoded metadata bytes.
    pub fn serialize(&self) -> Vec<u8> {
        pb::NumericalAggregateOpts {
            skip_nans: self.skip_nans,
        }
        .encode_to_vec()
    }

    /// Deserialize these options from protobuf-encoded metadata bytes.
    pub fn deserialize(metadata: &[u8]) -> VortexResult<Self> {
        let opts = pb::NumericalAggregateOpts::decode(metadata)?;
        Ok(Self {
            skip_nans: opts.skip_nans,
        })
    }
}

impl Default for NumericalAggregateOpts {
    fn default() -> Self {
        Self::skip_nans()
    }
}

impl Display for NumericalAggregateOpts {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Only the non-default configuration is displayed, so that aggregates with default
        // options render identically to their pre-options form, e.g. `vortex.sum()`.
        if !self.skip_nans {
            write!(f, "skip_nans=false")?;
        }
        Ok(())
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
