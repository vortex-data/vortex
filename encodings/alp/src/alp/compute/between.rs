// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::patches::Patches;
use vortex_array::scalar::PValue;
use vortex_error::VortexResult;
use vortex_functions::between::BetweenKernel;
use vortex_functions::between::BetweenOptions;
use vortex_functions::between::BetweenReduce;

use crate::ALP;
use crate::ALPFloat;
use crate::alp::array::ALPArrayExt;
use crate::alp::compute::predicate::apply_patch_predicate;
use crate::alp::compute::predicate::encoded_bound;
use crate::alp::compute::predicate::encoded_range;
use crate::match_each_alp_float_ptype;

/// Rewrites a between over a patch-free array into a between over its encoded integers.
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

        // Patched rows need their exact values, which the execute kernel below supplies.
        if array.patches().is_some() {
            return Ok(None);
        }

        let nullability =
            array.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();
        match_each_alp_float_ptype!(array.dtype().as_ptype(), |F| {
            between_impl::<F>(
                array,
                F::try_from(lower)?,
                F::try_from(upper)?,
                nullability,
                options,
            )
        })
        .map(Some)
    }
}

/// Answers a between over a patched array from its encoded integers, then re-evaluates the
/// patched rows on their exact values.
impl BetweenKernel for ALP {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(patches) = array.patches() else {
            return Ok(None);
        };
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        let nullability =
            array.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();
        match_each_alp_float_ptype!(array.dtype().as_ptype(), |F| {
            between_patched_impl::<F>(
                array,
                F::try_from(lower)?,
                F::try_from(upper)?,
                &patches,
                nullability,
                options,
                ctx,
            )
        })
        .map(Some)
    }
}

/// Answers a between from the encoded integers, then re-evaluates the patched rows exactly.
fn between_patched_impl<T: NativePType + ALPFloat>(
    array: ArrayView<'_, ALP>,
    lower: T,
    upper: T,
    patches: &Patches,
    nullability: Nullability,
    options: &BetweenOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T::ALPInt: NativePType + Into<PValue>,
{
    let encoded = between_impl::<T>(array, lower, upper, nullability, options)?;
    let lower_fn = options.lower_strict.to_predicate::<T>();
    let upper_fn = options.upper_strict.to_predicate::<T>();
    apply_patch_predicate(
        encoded,
        patches,
        |value: T| lower_fn(lower, value) & upper_fn(value, upper),
        ctx,
    )
}

fn between_impl<T: NativePType + ALPFloat>(
    array: ArrayView<'_, ALP>,
    lower: T,
    upper: T,
    nullability: Nullability,
    options: &BetweenOptions,
) -> VortexResult<ArrayRef>
where
    T::ALPInt: NativePType + Into<PValue>,
{
    let exponents = array.exponents();

    let start = encoded_bound(lower, exponents, options.lower_strict.is_strict());
    let end = encoded_bound(upper, exponents, !options.upper_strict.is_strict());
    encoded_range(array, start, end, false, nullability)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_functions::between::BetweenOptions;
    use vortex_functions::between::StrictComparison;
    use vortex_functions::builtins::ArrayFunctions;
    use vortex_session::VortexSession;

    use crate::ALP;
    use crate::ALPArray;
    use crate::Exponents;
    use crate::alp::array::ALPArrayExt;
    use crate::alp::compute::between::between_impl;
    use crate::alp_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn assert_between(
        arr: &ALPArray,
        lower: f32,
        upper: f32,
        options: &BetweenOptions,
        expected: bool,
    ) {
        let res =
            between_impl(arr.as_view(), lower, upper, Nullability::Nullable, options).unwrap();
        assert_arrays_eq!(
            res,
            ConstantArray::new(Scalar::bool(expected, Nullability::Nullable), arr.len())
                .into_array(),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn between_outside_encoded_range() -> VortexResult<()> {
        let ints = PrimitiveArray::from_iter([i32::MIN, 0, i32::MAX]);
        let array = ALP::try_new(ints.into_array(), Exponents { e: 0, f: 0 }, None)?;
        let strict = BetweenOptions {
            lower_strict: StrictComparison::Strict,
            upper_strict: StrictComparison::Strict,
        };
        assert_between(&array, -1e10, 1e10, &strict, true);
        Ok(())
    }

    #[test]
    fn comparison_range() {
        let value = 0.0605_f32;
        let array = PrimitiveArray::from_iter([value; 1]);
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();
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

    #[test]
    fn non_finite_bounds_use_total_order() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter([1.234f32; 10]);
        let encoded = alp_encode(array.as_view(), None, &mut ctx).unwrap();
        assert!(encoded.patches().is_none());

        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::Strict,
        };

        assert_between(&encoded, f32::from_bits(0xffffff5e), 2.0, &options, true);
        assert_between(&encoded, f32::NAN, 2.0, &options, false);
        assert_between(&encoded, f32::NEG_INFINITY, 2.0, &options, true);
        assert_between(&encoded, f32::INFINITY, 2.0, &options, false);

        assert_between(&encoded, 0.0, f32::NAN, &options, true);
        assert_between(&encoded, 0.0, f32::from_bits(0xffffff5e), &options, false);
        assert_between(&encoded, 0.0, f32::INFINITY, &options, true);
        assert_between(&encoded, 0.0, f32::NEG_INFINITY, &options, false);
    }

    #[rstest]
    #[case::negative_zero_lower(-0.0f32, 1.0f32, true)]
    #[case::negative_zero_upper(-1.0f32, -0.0f32, false)]
    #[case::subnormal_lower(f32::from_bits(1), 1.0f32, false)]
    #[case::subnormal_upper(-1.0f32, -f32::from_bits(1), false)]
    fn bounds_near_zero_use_total_order(
        #[case] lower: f32,
        #[case] upper: f32,
        #[case] expected: bool,
    ) {
        let array = PrimitiveArray::from_iter([0.0f32; 10]);
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();
        assert!(encoded.patches().is_none());

        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        };
        assert_between(&encoded, lower, upper, &options, expected);
    }

    #[test]
    fn nullable_constant_answer_keeps_validity() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_option_iter([Some(1.234f32), None, Some(1.234)]);
        let encoded = alp_encode(array.as_view(), None, &mut ctx)?;
        assert!(encoded.patches().is_none());

        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        };
        let res = between_impl(
            encoded.as_view(),
            f32::NAN,
            2.0,
            Nullability::Nullable,
            &options,
        )?;
        let expected = BoolArray::from_iter([Some(false), None, Some(false)]);
        assert_arrays_eq!(res, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(StrictComparison::NonStrict, StrictComparison::NonStrict)]
    #[case(StrictComparison::Strict, StrictComparison::NonStrict)]
    #[case(StrictComparison::NonStrict, StrictComparison::Strict)]
    #[case(StrictComparison::Strict, StrictComparison::Strict)]
    fn patched_matches_primitive(
        #[case] lower_strict: StrictComparison,
        #[case] upper_strict: StrictComparison,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<Option<f32>> = (0..3000)
            .map(|i| {
                if i % 11 == 0 {
                    None
                } else if i % 97 == 0 {
                    Some(PI * i as f32)
                } else {
                    Some(i as f32 / 100.0)
                }
            })
            .collect();
        let array = PrimitiveArray::from_option_iter(values);
        let encoded = alp_encode(array.as_view(), None, &mut ctx)?;
        assert!(encoded.patches().is_some());
        let options = BetweenOptions {
            lower_strict,
            upper_strict,
        };

        for (lower, upper) in [
            (10.0f32, 20.0f32),
            (PI * 97.0, PI * 1_067.0),
            (f32::NEG_INFINITY, 5.0),
            (f32::NAN, 5.0),
        ] {
            let lower = ConstantArray::new(lower, array.len()).into_array();
            let upper = ConstantArray::new(upper, array.len()).into_array();
            let actual = encoded.clone().into_array().between(
                lower.clone(),
                upper.clone(),
                options.clone(),
            )?;
            let expected = array
                .clone()
                .into_array()
                .between(lower, upper, options.clone())?;
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }
}
