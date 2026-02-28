// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool_accumulator;
mod primitive_accumulator;

use num_traits::Bounded;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use self::bool_accumulator::BoolExtremumAccumulator;
use self::primitive_accumulator::PrimitiveExtremumAccumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::accumulator::Accumulator;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::match_each_native_ptype;
use crate::scalar_fn::EmptyOptions;

/// Compile-time direction for extremum accumulators.
pub(crate) trait Direction: Send + Sync + 'static {
    /// Returns `true` if `candidate` should replace `current`.
    fn should_replace<T: NativePType>(current: T, candidate: T) -> bool;

    /// Returns `true` if `value` is at the type's extreme and no further improvement is possible.
    fn is_saturated<T: NativePType + Bounded>(value: T) -> bool;

    /// Returns `true` if `candidate` should replace `current` for booleans.
    fn should_replace_bool(current: bool, candidate: bool) -> bool;

    /// Returns `true` if the boolean value is saturated.
    fn is_saturated_bool(value: bool) -> bool;
}

/// Seek the minimum value.
pub(crate) struct FindMin;

impl Direction for FindMin {
    #[inline]
    fn should_replace<T: NativePType>(current: T, candidate: T) -> bool {
        candidate.is_lt(current)
    }

    #[inline]
    fn is_saturated<T: NativePType + Bounded>(value: T) -> bool {
        value.total_compare(T::min_value()).is_eq()
    }

    #[inline]
    fn should_replace_bool(_current: bool, candidate: bool) -> bool {
        !candidate
    }

    #[inline]
    fn is_saturated_bool(value: bool) -> bool {
        !value
    }
}

/// Seek the maximum value.
pub(crate) struct FindMax;

impl Direction for FindMax {
    #[inline]
    fn should_replace<T: NativePType>(current: T, candidate: T) -> bool {
        candidate.is_gt(current)
    }

    #[inline]
    fn is_saturated<T: NativePType + Bounded>(value: T) -> bool {
        value.total_compare(T::max_value()).is_eq()
    }

    #[inline]
    fn should_replace_bool(_current: bool, candidate: bool) -> bool {
        candidate
    }

    #[inline]
    fn is_saturated_bool(value: bool) -> bool {
        value
    }
}

/// Computes the minimum of numeric or boolean values.
///
/// Nulls and NaN values are skipped. The output dtype matches the input dtype but is always
/// nullable.
///
/// # Flush semantics
///
/// - **Empty group** (no accumulate/merge calls): produces **null**.
/// - **All-null group**: produces **null**.
/// - `is_saturated()` returns true once the type's minimum value is seen.
#[derive(Clone)]
pub struct Min;

/// Computes the maximum of numeric or boolean values.
///
/// Nulls and NaN values are skipped. The output dtype matches the input dtype but is always
/// nullable.
///
/// # Flush semantics
///
/// - **Empty group** (no accumulate/merge calls): produces **null**.
/// - **All-null group**: produces **null**.
/// - `is_saturated()` returns true once the type's maximum value is seen.
#[derive(Clone)]
pub struct Max;

fn return_dtype(input_dtype: &DType) -> VortexResult<DType> {
    match input_dtype {
        DType::Bool(_) => Ok(DType::Bool(Nullability::Nullable)),
        DType::Primitive(p, _) => Ok(DType::Primitive(*p, Nullability::Nullable)),
        _ => vortex_bail!(
            "Min/Max requires numeric or boolean input, got {}",
            input_dtype
        ),
    }
}

fn make_accumulator<D: Direction>(input_dtype: &DType) -> VortexResult<Box<dyn Accumulator>> {
    match input_dtype {
        DType::Bool(_) => Ok(Box::new(BoolExtremumAccumulator::<D>::new())),
        DType::Primitive(p, _) => Ok(match_each_native_ptype!(*p, |T| {
            Box::new(PrimitiveExtremumAccumulator::<T, D>::new()) as Box<dyn Accumulator>
        })),
        _ => vortex_bail!(
            "Min/Max requires numeric or boolean input, got {}",
            input_dtype
        ),
    }
}

impl AggregateFnVTable for Min {
    type Options = EmptyOptions;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new_ref("vortex.min")
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        return_dtype(input_dtype)
    }

    fn state_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn accumulator(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Box<dyn Accumulator>> {
        make_accumulator::<FindMin>(input_dtype)
    }
}

impl AggregateFnVTable for Max {
    type Options = EmptyOptions;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new_ref("vortex.max")
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        return_dtype(input_dtype)
    }

    fn state_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn accumulator(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Box<dyn Accumulator>> {
        make_accumulator::<FindMax>(input_dtype)
    }
}

#[cfg(test)]
mod tests;
