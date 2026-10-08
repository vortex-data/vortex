// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The byte-bounded minimum: a lower bound of a column's minimum that fits in a byte limit.

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
use crate::aggregate_fn::fns::bounded_min::BoundedMin;
use crate::aggregate_fn::fns::bounded_min::BoundedMinPartial;
use crate::aggregate_fn::fns::bounded_min::BoundedMinState;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::columnar_min_max;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::partial_ord::partial_min;
use crate::scalar::Scalar;
use crate::scalar::ScalarTruncation;
use crate::scalar::lower_bound;

/// A lower bound of the minimum non-null value of an array, at most `max_bytes` bytes long.
///
/// Fixed-width values are never truncated, so for them this is the exact NaN-skipping minimum. A
/// `Utf8`/`Binary` minimum longer than `max_bytes` is replaced by its `max_bytes`-byte prefix,
/// which is always a smaller value, so unlike [`MaxBound`] a bound always exists.
///
/// The partial is `{ value: T, is_exact: bool }?`, see [`MinBoundState`] for its three states.
/// The result is the nullable `value` alone, null for an empty input. Readers that need to know
/// whether the value is exact read the partial through [`AggregateFnVTable::to_scalar`].
///
/// This supersedes [`BoundedMin`], which does not record exactness. [`MinBoundPartial::from`] maps
/// its partials.
///
/// [`MaxBound`]: crate::aggregate_fn::fns::max_bound::MaxBound
#[derive(Clone, Copy, Debug)]
pub struct MinBound;

/// The accumulated state of [`MinBound`].
///
/// Every state but `Empty` describes the minimum `m` of the non-null values seen so far.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MinBoundState {
    /// No non-null values have been seen.
    Empty,
    /// The value is `m` itself.
    Exact(Scalar),
    /// The value is a bound `b <= m`, which `m` may not attain.
    LowerBound(Scalar),
}

/// Partial accumulator state for [`MinBound`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinBoundPartial {
    state: MinBoundState,
}

impl MinBoundPartial {
    /// A partial in the given state.
    pub fn new(state: MinBoundState) -> Self {
        Self { state }
    }

    /// The accumulated state.
    pub fn state(&self) -> &MinBoundState {
        &self.state
    }

    /// The accumulated state.
    pub fn into_state(self) -> MinBoundState {
        self.state
    }

    /// Merge the state of another input into this one.
    ///
    /// An exact minimum at or below a bound is still exact, since it is attained and undercuts
    /// everything above the bound. Above the bound the true minimum is unknown, so the bound wins.
    fn merge(&mut self, incoming: MinBoundState) {
        use MinBoundState::*;

        let current = std::mem::replace(&mut self.state, Empty);
        self.state = match (current, incoming) {
            (Empty, state) | (state, Empty) => state,
            (Exact(a), Exact(b)) => Exact(min_scalar(a, b)),
            (Exact(exact), LowerBound(bound)) | (LowerBound(bound), Exact(exact)) => {
                if compare(&exact, &bound) != Ordering::Greater {
                    Exact(exact)
                } else {
                    LowerBound(bound)
                }
            }
            (LowerBound(a), LowerBound(b)) => LowerBound(min_scalar(a, b)),
        };
    }
}

/// Map a [`BoundedMin`] partial to its [`MinBound`] equivalent.
///
/// `vortex.bounded_min` never recorded whether its value was truncated, so a `Utf8`/`Binary` value
/// can only be read as a lower bound. Fixed-width values were never truncated and stay exact.
impl From<BoundedMinPartial> for MinBoundPartial {
    fn from(partial: BoundedMinPartial) -> Self {
        Self::new(match partial.into_state() {
            BoundedMinState::Empty => MinBoundState::Empty,
            BoundedMinState::Value(value) => {
                if matches!(value.dtype(), DType::Utf8(_) | DType::Binary(_)) {
                    MinBoundState::LowerBound(value)
                } else {
                    MinBoundState::Exact(value)
                }
            }
        })
    }
}

impl MinBound {
    /// Parse a serialized [`BoundedMin`] partial into a [`MinBound`] partial.
    ///
    /// See [`MinBoundPartial::from`] for how the states map.
    pub fn partial_from_bounded_min(&self, scalar: &Scalar) -> VortexResult<MinBoundPartial> {
        BoundedMinPartial::from_scalar(scalar).map(MinBoundPartial::from)
    }
}

impl AggregateFnVTable for MinBound {
    type Options = BoundOptions;
    type Partial = MinBoundPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.min_bound");
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
        BoundOptions::deserialize(metadata, "MinBound")
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

        // The stored value is a sound lower bound for the superseded aggregate and, as it skips
        // NaNs, for the NaN-skipping minimum. A NaN-including minimum may be NaN, which no bound
        // stands in for.
        if requested.is::<BoundedMin>()
            || requested
                .as_opt::<Min>()
                .is_some_and(|options| options.skip_nans)
        {
            AggregateFnSatisfaction::Approximate
        } else {
            AggregateFnSatisfaction::No
        }
    }

    fn partial_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        supported_dtype(input_dtype)
            .map(|dtype| bound_partial_dtype(dtype, Nullability::NonNullable))
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(MinBoundPartial::new(MinBoundState::Empty))
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        let state = match parse_bound_partial(scalar, "MinBound")? {
            None => MinBoundState::Empty,
            Some((value, _)) if value.is_null() => {
                vortex_bail!("MinBound partial has no value")
            }
            Some((value, true)) => MinBoundState::Exact(value),
            Some((value, false)) => MinBoundState::LowerBound(value),
        };
        Ok(MinBoundPartial::new(state))
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
        let value_dtype = args.dtype.as_nonnullable();
        Ok(match &partial.state {
            MinBoundState::Empty => Scalar::null(args.partial_dtype.clone()),
            MinBoundState::Exact(value) => {
                bound_partial_scalar(args.partial_dtype, value.cast(&value_dtype)?, true)
            }
            MinBoundState::LowerBound(value) => {
                bound_partial_scalar(args.partial_dtype, value.cast(&value_dtype)?, false)
            }
        })
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        false
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Delegate to the existing min_max implementation for now. A dedicated min aggregate
        // would avoid computing max when only min is needed.
        let Some(result) = columnar_min_max(batch, NumericalAggregateOpts::default(), ctx)? else {
            return Ok(());
        };
        partial.merge(lower_bound_state(result.min, args.options.max_bytes.get())?);
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
            MinBoundState::Exact(value) | MinBoundState::LowerBound(value) => value.cast(&dtype),
            MinBoundState::Empty => Ok(Scalar::null(dtype)),
        }
    }
}

/// The state recording a batch minimum of `value` under the `max_bytes` limit.
fn lower_bound_state(value: Scalar, max_bytes: usize) -> VortexResult<MinBoundState> {
    if value.is_null() {
        return Ok(MinBoundState::Empty);
    }
    let nullability = value.dtype().nullability();
    let bound = match value.dtype() {
        DType::Utf8(_) => lower_bound(BufferString::from_scalar(value)?, max_bytes, nullability),
        DType::Binary(_) => lower_bound(ByteBuffer::from_scalar(value)?, max_bytes, nullability),
        _ => return Ok(MinBoundState::Exact(value)),
    };
    Ok(match bound {
        // `lower_bound` only returns `None` for a null value, which was handled above.
        None => MinBoundState::Empty,
        Some((bound, true)) => MinBoundState::LowerBound(bound),
        Some((value, false)) => MinBoundState::Exact(value),
    })
}

fn compare(a: &Scalar, b: &Scalar) -> Ordering {
    a.partial_cmp(b)
        .vortex_expect("incomparable min_bound scalars")
}

fn min_scalar(a: Scalar, b: Scalar) -> Scalar {
    partial_min(a, b).vortex_expect("incomparable min_bound scalars")
}

#[cfg(test)]
mod tests;
