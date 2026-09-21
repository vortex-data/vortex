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
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnSatisfaction;
use crate::dtype::DType;
use crate::proto::expr as pb;
use crate::scalar::Scalar;

/// Resolved dtypes of one aggregate function bound to its options and input.
///
/// Accumulators resolve these once and pair them with the options as [`AggregateArgs`] for every
/// execution method, so partial states only hold accumulated values.
#[derive(Clone, Debug)]
pub struct AggregateDTypes {
    /// The DType of the input.
    pub dtype: DType,
    /// The DType of the aggregate, as reported by [`AggregateFnVTable::return_dtype`].
    pub return_dtype: DType,
    /// The DType of the partial accumulator state, as reported by
    /// [`AggregateFnVTable::partial_dtype`].
    pub partial_dtype: DType,
}

impl AggregateDTypes {
    /// Resolve the return and partial dtypes of `vtable` bound to `options` over `dtype`.
    ///
    /// Fails if the aggregate cannot be applied to `dtype`.
    pub fn try_new<V: AggregateFnVTable>(
        vtable: &V,
        options: &V::Options,
        dtype: DType,
    ) -> VortexResult<Self> {
        let return_dtype = vtable.return_dtype(options, &dtype).ok_or_else(|| {
            vortex_err!(
                "Aggregate function {} cannot be applied to dtype {}",
                vtable.id(),
                dtype
            )
        })?;
        let partial_dtype = vtable.partial_dtype(options, &dtype).ok_or_else(|| {
            vortex_err!(
                "Aggregate function {} cannot be applied to dtype {}",
                vtable.id(),
                dtype
            )
        })?;
        Ok(Self {
            dtype,
            return_dtype,
            partial_dtype,
        })
    }

    /// Pair these dtypes with `options` for one aggregate execution call.
    pub fn args<'a, O>(&'a self, options: &'a O) -> AggregateArgs<'a, O> {
        AggregateArgs {
            options,
            dtype: &self.dtype,
            return_dtype: &self.return_dtype,
            partial_dtype: &self.partial_dtype,
        }
    }
}

/// The options and resolved dtypes passed to every aggregate execution method.
#[derive(Debug)]
pub struct AggregateArgs<'a, O> {
    /// The options the aggregate function is bound to.
    pub options: &'a O,
    /// The DType of the input.
    pub dtype: &'a DType,
    /// The DType of the aggregate, as reported by [`AggregateFnVTable::return_dtype`].
    pub return_dtype: &'a DType,
    /// The DType of the partial accumulator state, as reported by
    /// [`AggregateFnVTable::partial_dtype`].
    pub partial_dtype: &'a DType,
}

// Derived impls would bound `O: Clone + Copy`, which the borrowed fields do not need.
impl<O> Clone for AggregateArgs<'_, O> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<O> Copy for AggregateArgs<'_, O> {}

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

    /// If this aggregate should be computed as a default zone statistic for `input_dtype`, return
    /// the bound aggregate to store. Default: not a zone-map default.
    fn zone_stat_default(&self, _input_dtype: &DType) -> Option<AggregateFnRef> {
        None
    }

    /// DType of the intermediate partial accumulator state.
    ///
    /// Use a struct dtype when multiple fields are needed
    /// (e.g., Mean: `Struct { sum: f64, count: u64 }`).
    ///
    /// Returns `None` if the aggregate function cannot be applied to the input dtype.
    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType>;

    /// The partial state of a group with no accumulated values.
    ///
    /// The identity of [`merge_partials`].
    ///
    /// [`merge_partials`]: AggregateFnVTable::merge_partials
    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial>;

    /// Parse a partial scalar into the typed partial state.
    ///
    /// The scalar must have dtype `args.partial_dtype`; this is the inverse of [`to_scalar`].
    ///
    /// [`to_scalar`]: AggregateFnVTable::to_scalar
    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial>;

    /// Merge two partial states into one.
    ///
    /// `first` accumulated the input preceding `second`'s. Order-dependent aggregates (e.g.
    /// first/last or is_sorted) rely on this, so the merge need not be commutative. Merging with
    /// [`empty_partial`] on either side is the identity.
    ///
    /// [`empty_partial`]: AggregateFnVTable::empty_partial
    fn merge_partials(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial>;

    /// Convert the partial state into a partial scalar of dtype `args.partial_dtype`.
    ///
    /// This is the inverse of [`partial_from_scalar`]: parsing the returned scalar must
    /// reconstruct an equivalent state.
    ///
    /// [`partial_from_scalar`]: AggregateFnVTable::partial_from_scalar
    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar>;

    /// Is the partial state "saturated", i.e. has it reached a state where the final result is
    /// fully determined.
    fn is_saturated(&self, args: AggregateArgs<'_, Self::Options>, partial: &Self::Partial)
    -> bool;

    /// Try to accumulate the raw array before decompression.
    ///
    /// Returns `true` if the array was handled, `false` to fall through to the default kernel
    /// dispatch and canonicalization path. When returning `false`, the partial state must be
    /// unchanged, since the same batch is then accumulated by the fallback path.
    ///
    /// This is useful for aggregates that only depend on array metadata (e.g., validity)
    /// rather than the encoded data, avoiding unnecessary decompression.
    fn try_accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _state: &mut Self::Partial,
        _batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        Ok(false)
    }

    /// Accumulate a new canonical array into the partial state.
    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    /// Finalize an array of partial states into an array of aggregate results.
    ///
    /// The `states` array has dtype `args.partial_dtype`; the result must have dtype
    /// `args.return_dtype`.
    fn finalize(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        states: ArrayRef,
    ) -> VortexResult<ArrayRef>;

    /// Finalize a partial state into an aggregate result of dtype `args.return_dtype`.
    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar>;
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
