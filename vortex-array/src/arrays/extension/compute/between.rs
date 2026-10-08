// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Extension;
use crate::arrays::extension::ExtensionArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::BetweenReduce;

impl BetweenReduce for Extension {
    /// Evaluates between on the storage array, so a date or timestamp keeps its compressed
    /// storage and reaches that encoding's between kernel instead of being decompressed.
    fn between(
        array: ArrayView<'_, Extension>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        // Storage values are only ordered alike when the bounds share the extension dtype
        // (a timestamp in milliseconds must not be compared against raw seconds).
        if !array.dtype().eq_ignore_nullability(lower.dtype())
            || !array.dtype().eq_ignore_nullability(upper.dtype())
        {
            return Ok(None);
        }

        let (Some(lower), Some(upper)) = (storage_of(lower), storage_of(upper)) else {
            return Ok(None);
        };

        array
            .storage_array()
            .clone()
            .between(lower, upper, options.clone())
            .map(Some)
    }
}

/// The storage of a bound: a constant extension scalar becomes a constant storage scalar, and an
/// extension array its storage array.
fn storage_of(bound: &ArrayRef) -> Option<ArrayRef> {
    if let Some(scalar) = bound.as_constant() {
        let storage = scalar.as_extension().to_storage_scalar();
        return Some(ConstantArray::new(storage, bound.len()).into_array());
    }
    bound
        .as_opt::<Extension>()
        .map(|ext| ext.storage_array().clone())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::Nullability;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;
    use crate::scalar_fn::fns::operators::Operator;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

    fn dates(storage: ArrayRef) -> ArrayRef {
        let nullability = storage.dtype().nullability();
        ExtensionArray::new(Date::new(TimeUnit::Days, nullability).erased(), storage).into_array()
    }

    fn date(days: i32) -> Scalar {
        Scalar::extension::<Date>(
            TimeUnit::Days,
            Scalar::primitive(days, Nullability::NonNullable),
        )
    }

    /// The pushdown must agree with the two compares that between stands for.
    fn assert_matches_compares(
        array: ArrayRef,
        lower: ArrayRef,
        upper: ArrayRef,
        options: BetweenOptions,
    ) -> VortexResult<()> {
        let ctx = &mut SESSION.create_execution_ctx();
        let expected = lower
            .clone()
            .binary(array.clone(), options.lower_strict.to_operator())?
            .binary(
                array.clone().binary(upper.clone(), options.upper_strict.to_operator())?,
                Operator::And,
            )?
            .execute::<Canonical>(ctx)?
            .into_array();
        let actual = array
            .between(lower, upper, options)?
            .execute::<Canonical>(ctx)?
            .into_array();
        assert_arrays_eq!(actual, expected, ctx);
        Ok(())
    }

    #[rstest]
    #[case(StrictComparison::NonStrict, StrictComparison::NonStrict)]
    #[case(StrictComparison::NonStrict, StrictComparison::Strict)]
    #[case(StrictComparison::Strict, StrictComparison::NonStrict)]
    #[case(StrictComparison::Strict, StrictComparison::Strict)]
    fn constant_bounds(
        #[case] lower_strict: StrictComparison,
        #[case] upper_strict: StrictComparison,
    ) -> VortexResult<()> {
        let array = dates(buffer![1i32, 2, 3, 4, 5].into_array());
        assert_matches_compares(
            array,
            ConstantArray::new(date(2), 5).into_array(),
            ConstantArray::new(date(4), 5).into_array(),
            BetweenOptions {
                lower_strict,
                upper_strict,
            },
        )
    }

    #[test]
    fn nullable_storage() -> VortexResult<()> {
        let array = dates(
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None])
                .into_array(),
        );
        assert_matches_compares(
            array,
            ConstantArray::new(date(1), 5).into_array(),
            ConstantArray::new(date(3), 5).into_array(),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )
    }

    #[test]
    fn array_bounds() -> VortexResult<()> {
        let array = dates(buffer![1i32, 2, 3, 4, 5].into_array());
        assert_matches_compares(
            array,
            dates(buffer![0i32, 3, 3, 5, 0].into_array()),
            dates(buffer![2i32, 2, 9, 9, 4].into_array()),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
        )
    }

    #[test]
    fn timestamps() -> VortexResult<()> {
        let dtype = Timestamp::new(TimeUnit::Microseconds, Nullability::NonNullable).erased();
        let ts = |v: i64| {
            Scalar::extension_ref(dtype.clone(), Scalar::primitive(v, Nullability::NonNullable))
        };
        let array =
            ExtensionArray::new(dtype.clone(), buffer![10i64, 20, 30, 40].into_array()).into_array();
        assert_matches_compares(
            array,
            ConstantArray::new(ts(15), 4).into_array(),
            ConstantArray::new(ts(40), 4).into_array(),
            BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::NonStrict,
            },
        )
    }
}
