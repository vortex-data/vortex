// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The byte-bounded maximum: an upper bound of a column's maximum that fits in a byte limit.

use std::cmp::Ordering;

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnSatisfaction;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::bound::BOUND_VALUE;
use crate::aggregate_fn::fns::bound::BoundOptions;
use crate::aggregate_fn::fns::bound::bound_partial_dtype;
use crate::aggregate_fn::fns::bound::bound_partial_scalar;
use crate::aggregate_fn::fns::bound::byte_limit_satisfaction;
use crate::aggregate_fn::fns::bound::parse_bound_partial;
use crate::aggregate_fn::fns::bound::supported_dtype;
use crate::aggregate_fn::fns::bounded_max::BoundedMax;
use crate::aggregate_fn::fns::bounded_max::BoundedMaxPartial;
use crate::aggregate_fn::fns::bounded_max::BoundedMaxState;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min_max::columnar_min_max;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::partial_ord::partial_max;
use crate::scalar::Scalar;
use crate::scalar::ScalarTruncation;
use crate::scalar::upper_bound;

/// An upper bound of the maximum non-null value of an array, at most `max_bytes` bytes long.
///
/// Fixed-width values are never truncated, so for them this is the exact NaN-skipping maximum.
/// A `Utf8`/`Binary` maximum longer than `max_bytes` is replaced by the smallest value of at most
/// `max_bytes` bytes that is greater than it; when no such value exists (every byte of the
/// truncated prefix is already at its limit), the maximum has no representable upper bound.
///
/// The partial is `{ value: T?, is_exact: bool }?`, see [`MaxBoundState`] for its four states.
/// The result is the nullable `value` alone: null for an empty input and for an unrepresentable
/// bound, which pruning treats alike. Readers that need to tell the states apart, or to know
/// whether the value is exact, read the partial through [`AggregateFnVTable::to_scalar`].
///
/// This supersedes [`BoundedMax`], which neither records exactness nor distinguishes an empty
/// input from an unrepresentable bound in its result. [`MaxBoundPartial::from`] maps its partials.
#[derive(Clone, Copy, Debug)]
pub struct MaxBound;

/// The accumulated state of [`MaxBound`].
///
/// Every state but `Empty` describes the maximum `m` of the non-null values seen so far.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaxBoundState {
    /// No non-null values have been seen.
    Empty,
    /// The value is `m` itself.
    Exact(Scalar),
    /// The value is a bound `b >= m`, which `m` may not attain.
    UpperBound(Scalar),
    /// `m` exceeds every value of at most `max_bytes` bytes, so no bound can be recorded.
    Unrepresentable,
}

/// Partial accumulator state for [`MaxBound`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxBoundPartial {
    state: MaxBoundState,
}

impl MaxBoundPartial {
    /// A partial in the given state.
    pub fn new(state: MaxBoundState) -> Self {
        Self { state }
    }

    /// The accumulated state.
    pub fn state(&self) -> &MaxBoundState {
        &self.state
    }

    /// The accumulated state.
    pub fn into_state(self) -> MaxBoundState {
        self.state
    }

    /// Merge the state of another input into this one.
    ///
    /// An exact maximum at or above a bound is still exact, since it is attained and exceeds
    /// everything under the bound. Below the bound the true maximum is unknown, so the bound wins.
    fn merge(&mut self, incoming: MaxBoundState) {
        use MaxBoundState::*;

        let current = std::mem::replace(&mut self.state, Empty);
        self.state = match (current, incoming) {
            (Empty, state) | (state, Empty) => state,
            (Unrepresentable, _) | (_, Unrepresentable) => Unrepresentable,
            (Exact(a), Exact(b)) => Exact(max_scalar(a, b)),
            (Exact(exact), UpperBound(bound)) | (UpperBound(bound), Exact(exact)) => {
                if compare(&exact, &bound) != Ordering::Less {
                    Exact(exact)
                } else {
                    UpperBound(bound)
                }
            }
            (UpperBound(a), UpperBound(b)) => UpperBound(max_scalar(a, b)),
        };
    }
}

/// Map a [`BoundedMax`] partial to its [`MaxBound`] equivalent.
///
/// `vortex.bounded_max` never recorded whether its value was truncated, so a `Utf8`/`Binary` value
/// can only be read as an upper bound. Fixed-width values were never truncated and stay exact.
impl From<BoundedMaxPartial> for MaxBoundPartial {
    fn from(partial: BoundedMaxPartial) -> Self {
        Self::new(match partial.into_state() {
            BoundedMaxState::Empty => MaxBoundState::Empty,
            BoundedMaxState::Value(value) => {
                if matches!(value.dtype(), DType::Utf8(_) | DType::Binary(_)) {
                    MaxBoundState::UpperBound(value)
                } else {
                    MaxBoundState::Exact(value)
                }
            }
            BoundedMaxState::Unknown => MaxBoundState::Unrepresentable,
        })
    }
}

impl MaxBound {
    /// Parse a serialized [`BoundedMax`] partial into a [`MaxBound`] partial.
    ///
    /// See [`MaxBoundPartial::from`] for how the states map.
    pub fn partial_from_bounded_max(&self, scalar: &Scalar) -> VortexResult<MaxBoundPartial> {
        BoundedMaxPartial::from_scalar(scalar).map(MaxBoundPartial::from)
    }
}

impl AggregateFnVTable for MaxBound {
    type Options = BoundOptions;
    type Partial = MaxBoundPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.max_bound");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        options.serialize().map(Some)
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        BoundOptions::deserialize(metadata, "MaxBound")
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        supported_dtype(input_dtype).map(DType::as_nullable)
    }

    fn can_satisfy(
        &self,
        options: &Self::Options,
        requested: &AggregateFnRef,
    ) -> AggregateFnSatisfaction {
        if let Some(other) = requested.as_opt::<Self>() {
            return byte_limit_satisfaction(options, other);
        }

        // The stored value is a sound upper bound for the superseded aggregate and, as it skips
        // NaNs, for the NaN-skipping maximum. A NaN-including maximum may be NaN, which no bound
        // stands in for.
        if requested.is::<BoundedMax>()
            || requested
                .as_opt::<Max>()
                .is_some_and(|options| options.skip_nans)
        {
            AggregateFnSatisfaction::Approximate
        } else {
            AggregateFnSatisfaction::No
        }
    }

    fn partial_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        supported_dtype(input_dtype).map(|dtype| bound_partial_dtype(dtype, Nullability::Nullable))
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(MaxBoundPartial::new(MaxBoundState::Empty))
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        let state = match parse_bound_partial(scalar, "MaxBound")? {
            None => MaxBoundState::Empty,
            Some((value, true)) if value.is_null() => {
                vortex_bail!("MaxBound partial claims an exact maximum but has no value")
            }
            Some((value, true)) => MaxBoundState::Exact(value),
            Some((value, false)) if value.is_null() => MaxBoundState::Unrepresentable,
            Some((value, false)) => MaxBoundState::UpperBound(value),
        };
        Ok(MaxBoundPartial::new(state))
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        mut first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        first.merge(second.state);
        Ok(first)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        let value_dtype = args.dtype.as_nullable();
        Ok(match &partial.state {
            MaxBoundState::Empty => Scalar::null(args.partial_dtype.clone()),
            MaxBoundState::Exact(value) => {
                bound_partial_scalar(args.partial_dtype, value.cast(&value_dtype)?, true)
            }
            MaxBoundState::UpperBound(value) => {
                bound_partial_scalar(args.partial_dtype, value.cast(&value_dtype)?, false)
            }
            MaxBoundState::Unrepresentable => {
                bound_partial_scalar(args.partial_dtype, Scalar::null(value_dtype), false)
            }
        })
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> bool {
        matches!(partial.state, MaxBoundState::Unrepresentable)
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if self.is_saturated(args, partial) {
            return Ok(());
        }
        // Delegate to the existing min_max implementation for now. A dedicated max aggregate
        // would avoid computing min when only max is needed.
        let Some(result) = columnar_min_max(batch, NumericalAggregateOpts::default(), ctx)? else {
            return Ok(());
        };
        partial.merge(upper_bound_state(result.max, args.options.max_bytes.get())?);
        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        partials.get_item(BOUND_VALUE)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        let dtype = args.return_dtype.clone();
        match &partial.state {
            MaxBoundState::Exact(value) | MaxBoundState::UpperBound(value) => value.cast(&dtype),
            MaxBoundState::Empty | MaxBoundState::Unrepresentable => Ok(Scalar::null(dtype)),
        }
    }
}

/// The state recording a batch maximum of `value` under the `max_bytes` limit.
fn upper_bound_state(value: Scalar, max_bytes: usize) -> VortexResult<MaxBoundState> {
    if value.is_null() {
        return Ok(MaxBoundState::Empty);
    }
    let nullability = value.dtype().nullability();
    let bound = match value.dtype() {
        DType::Utf8(_) => upper_bound(BufferString::from_scalar(value)?, max_bytes, nullability),
        DType::Binary(_) => upper_bound(ByteBuffer::from_scalar(value)?, max_bytes, nullability),
        _ => return Ok(MaxBoundState::Exact(value)),
    };
    Ok(match bound {
        None => MaxBoundState::Unrepresentable,
        Some((bound, true)) => MaxBoundState::UpperBound(bound),
        Some((value, false)) => MaxBoundState::Exact(value),
    })
}

fn compare(a: &Scalar, b: &Scalar) -> Ordering {
    a.partial_cmp(b)
        .vortex_expect("incomparable max_bound scalars")
}

fn max_scalar(a: Scalar, b: Scalar) -> Scalar {
    partial_max(a, b).vortex_expect("incomparable max_bound scalars")
}

#[cfg(test)]
mod tests;
