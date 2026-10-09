// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::PrimInt;
use num_traits::WrappingSub;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::BetweenReduce;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::FoR;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;

impl BetweenReduce for FoR {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower), Some(upper), Some(reference)) = (
            lower.as_constant(),
            upper.as_constant(),
            array.constant_reference(),
        ) else {
            return Ok(None);
        };
        let nullability =
            array.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();
        match_each_integer_ptype!(array.ptype(), |T| {
            between_constant::<T>(
                array,
                lower
                    .as_primitive()
                    .as_::<T>()
                    .vortex_expect("non-null bound"),
                upper
                    .as_primitive()
                    .as_::<T>()
                    .vortex_expect("non-null bound"),
                reference
                    .as_primitive()
                    .as_::<T>()
                    .vortex_expect("non-null reference"),
                nullability,
                options,
            )
            .map(Some)
        })
    }
}

fn between_constant<T: NativePType + PrimInt + WrappingSub>(
    array: ArrayView<'_, FoR>,
    lower: T,
    upper: T,
    reference: T,
    nullability: Nullability,
    options: &BetweenOptions,
) -> VortexResult<ArrayRef>
where
    PValue: From<T>,
{
    let between = |lower, upper, options| {
        array.encoded().clone().between(
            ConstantArray::new(Scalar::primitive(lower, nullability), array.len()).into_array(),
            ConstantArray::new(Scalar::primitive(upper, nullability), array.len()).into_array(),
            options,
        )
    };
    let no_matches = || {
        between(
            T::zero(),
            T::zero(),
            BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::Strict,
            },
        )
    };
    let lower = if options.lower_strict.is_strict() {
        let Some(lower) = lower.checked_add(&T::one()) else {
            return no_matches();
        };
        lower
    } else {
        lower
    };
    let upper = if options.upper_strict.is_strict() {
        let Some(upper) = upper.checked_sub(&T::one()) else {
            return no_matches();
        };
        upper
    } else {
        upper
    };
    if lower > upper {
        return no_matches();
    }
    let lower = lower.wrapping_sub(&reference);
    let upper = upper.wrapping_sub(&reference);
    let inclusive = BetweenOptions {
        lower_strict: StrictComparison::NonStrict,
        upper_strict: StrictComparison::NonStrict,
    };
    if lower <= upper {
        return between(lower, upper, inclusive);
    }
    // Subtracting the reference can move an interval across the encoded type's wrap point.
    // Its preimage is then two disjoint intervals, with validity preserved by Kleene OR.
    between(T::min_value(), upper, inclusive.clone())?
        .binary(between(lower, T::max_value(), inclusive)?, Operator::Or)
}

#[cfg(test)]
mod tests;
