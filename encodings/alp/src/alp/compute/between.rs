// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativeDType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::BetweenReduce;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_error::VortexResult;

use crate::ALP;
use crate::ALPFloat;
use crate::alp::array::ALPArrayExt;
use crate::alp::array::ALPArraySlotsExt;
use crate::match_each_alp_float_ptype;

impl BetweenReduce for ALP {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        if array.patches().is_some() {
            return Ok(None);
        }

        let nullability =
            array.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();
        match_each_alp_float_ptype!(array.dtype().as_ptype(), |F| {
            between_impl::<F>(
                array,
                F::try_from(&lower)?,
                F::try_from(&upper)?,
                nullability,
                options,
            )
        })
        .map(Some)
    }
}

fn between_impl<T: NativePType + ALPFloat>(
    array: ArrayView<'_, ALP>,
    lower: T,
    upper: T,
    nullability: Nullability,
    options: &BetweenOptions,
) -> VortexResult<ArrayRef>
where
    Scalar: From<T::ALPInt>,
    <T as ALPFloat>::ALPInt: NativeDType + Debug,
{
    let exponents = array.exponents();

    // There are always compared
    // the below bound is `value {< | <=} x`, either value encodes into the ALPInt domain
    // in which case we can leave the comparison unchanged `enc(value) {< | <=} x` or it doesn't
    // and we encode into value below enc_below(value) < value < x, in which case the comparison
    // becomes enc(value) < x. See `alp_scalar_compare` for more details.
    // note that if the value doesn't encode than value != x, so must use strict comparison.
    let (lower_enc, lower_strict) = T::encode_single(lower, exponents)
        .map(|x| (x, options.lower_strict))
        .unwrap_or_else(|| (T::encode_below(lower, exponents), StrictComparison::Strict));

    // the upper value `x { < | <= } value` similarly encodes or `x < value < enc_above(value())`
    let (upper_enc, upper_strict) = T::encode_single(upper, exponents)
        .map(|x| (x, options.upper_strict))
        .unwrap_or_else(|| (T::encode_above(upper, exponents), StrictComparison::Strict));

    let options = BetweenOptions {
        lower_strict,
        upper_strict,
    };

    array.encoded().clone().between(
        ConstantArray::new(Scalar::primitive(lower_enc, nullability), array.len()).into_array(),
        ConstantArray::new(Scalar::primitive(upper_enc, nullability), array.len()).into_array(),
        options,
    )
}

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison;

    use crate::ALPArray;
    use crate::alp::array::ALPArrayExt;
    use crate::alp::compute::between::between_impl;
    use crate::alp_encode;

    fn assert_between(
        arr: &ALPArray,
        lower: f32,
        upper: f32,
        options: &BetweenOptions,
        expected: bool,
    ) {
        let res =
            between_impl(arr.as_view(), lower, upper, Nullability::Nullable, options).unwrap();
        assert_arrays_eq!(res, BoolArray::from_iter([Some(expected)]));
    }

    #[test]
    fn comparison_range() {
        let value = 0.0605_f32;
        let array = PrimitiveArray::from_iter([value; 1]);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert!(encoded.patches().is_none());

        assert_between(
            &encoded,
            0.0605_f32,
            0.0605,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
            true,
        );

        assert_between(
            &encoded,
            0.0605_f32,
            0.0605,
            &BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::NonStrict,
            },
            false,
        );

        assert_between(
            &encoded,
            0.0605_f32,
            0.0605,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
            false,
        );

        assert_between(
            &encoded,
            0.060499_f32,
            0.06051,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
            true,
        );

        assert_between(
            &encoded,
            0.06_f32,
            0.06051,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
            true,
        );
    }
}
