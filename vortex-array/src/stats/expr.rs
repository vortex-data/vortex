// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression constructors for statistics backed by aggregate functions.

use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::all_nan::AllNan;
use crate::aggregate_fn::fns::all_non_nan::AllNonNan;
use crate::aggregate_fn::fns::all_non_null::AllNonNull;
use crate::aggregate_fn::fns::all_null::AllNull;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::nan_count::NanCount;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::expr::Expression;
use crate::scalar_fn::ScalarFnVTableExt;
pub use crate::scalar_fn::fns::stat::StatFn;
pub use crate::scalar_fn::fns::stat::StatOptions;

/// Creates an expression that reads a stored aggregate statistic for `expr`.
///
/// If the statistic is not available in the current stats scope, evaluating the expression returns
/// a nullable all-null array with the aggregate return type.
pub fn stat(expr: Expression, aggregate_fn: AggregateFnRef) -> Expression {
    StatFn.new_expr(StatOptions::new(aggregate_fn), [expr])
}

/// Creates `stat(expr, min_max)`, returning a nullable `{ min, max }` struct statistic.
pub fn min_max(expr: Expression) -> Expression {
    // Statistics follow NaN-skipping semantics; request it explicitly rather than via the default.
    stat(expr, MinMax.bind(NumericalAggregateOpts::skip_nans()))
}

/// Creates `stat(expr, sum)`, returning a nullable sum statistic.
pub fn sum(expr: Expression) -> Expression {
    // Statistics follow NaN-skipping semantics; request it explicitly rather than via the default.
    stat(expr, Sum.bind(NumericalAggregateOpts::skip_nans()))
}

/// Creates `stat(expr, null_count)`, returning a nullable null-count statistic.
pub fn null_count(expr: Expression) -> Expression {
    stat(expr, NullCount.bind(EmptyOptions))
}

/// Creates `stat(expr, all_null)`, returning a nullable all-null statistic.
pub fn all_null(expr: Expression) -> Expression {
    stat(expr, AllNull.bind(EmptyOptions))
}

/// Creates `stat(expr, all_nan)`, returning a nullable all-NaN statistic.
pub fn all_nan(expr: Expression) -> Expression {
    stat(expr, AllNan.bind(EmptyOptions))
}

/// Creates `stat(expr, all_non_null)`, returning a nullable all-non-null statistic.
pub fn all_non_null(expr: Expression) -> Expression {
    stat(expr, AllNonNull.bind(EmptyOptions))
}

/// Creates `stat(expr, all_non_nan)`, returning a nullable all-non-NaN statistic.
pub fn all_non_nan(expr: Expression) -> Expression {
    stat(expr, AllNonNan.bind(EmptyOptions))
}

/// Creates `stat(expr, nan_count)`, returning a nullable NaN-count statistic.
pub fn nan_count(expr: Expression) -> Expression {
    stat(expr, NanCount.bind(EmptyOptions))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::all_nan;
    use super::all_non_nan;
    use super::all_non_null;
    use super::all_null;
    use super::null_count;
    use super::stat;
    use super::sum;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Chunked;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::root;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

    #[test]
    fn stat_expr_reads_cached_sum() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        let sum_scalar = Scalar::primitive(6i64, Nullability::Nullable);
        array.statistics().set(
            Stat::Sum,
            Precision::exact(sum_scalar.into_value().vortex_expect("non-null sum")),
        );

        let result = array
            .apply(&sum(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::primitive(6i64, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_returns_null_when_sum_is_missing() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();

        let result = array
            .apply(&sum(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
            3,
        )
        .into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_sum_per_chunk() -> VortexResult<()> {
        let chunk0 = buffer![1i32, 2].into_array();
        let sum_scalar = Scalar::primitive(3i64, Nullability::Nullable);
        chunk0.statistics().set(
            Stat::Sum,
            Precision::exact(sum_scalar.into_value().vortex_expect("non-null sum")),
        );
        let chunk1 = buffer![4i32, 5, 6].into_array();
        let chunked = ChunkedArray::try_new(
            vec![chunk0, chunk1],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?
        .into_array();

        let result = chunked.apply(&sum(root()))?;

        let chunked_result = result
            .as_opt::<Chunked>()
            .vortex_expect("stat expression should preserve chunked alignment");
        assert_eq!(chunked_result.nchunks(), 2);

        let result = result
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();
        let expected = PrimitiveArray::new(
            buffer![3i64, 3, 0, 0, 0],
            Validity::from_iter([true, true, false, false, false]),
        )
        .into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_null_count() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None]).into_array();
        let null_count_scalar = Scalar::primitive(2u64, Nullability::NonNullable);
        array.statistics().set(
            Stat::NullCount,
            Precision::exact(
                null_count_scalar
                    .into_value()
                    .vortex_expect("non-null null_count"),
            ),
        );

        let result = array
            .apply(&null_count(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::primitive(2u64, Nullability::Nullable), 4).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_null_from_null_count() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]).into_array();
        array
            .statistics()
            .set(Stat::NullCount, Precision::exact(ScalarValue::from(3u64)));

        let result = array
            .apply(&all_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_null_false_from_inexact_low_null_count() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, Some(2), None]).into_array();
        array
            .statistics()
            .set(Stat::NullCount, Precision::inexact(ScalarValue::from(2u64)));

        let result = array
            .apply(&all_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(false, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_returns_null_for_inexact_full_null_count_as_all_null() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, Some(2), None]).into_array();
        array
            .statistics()
            .set(Stat::NullCount, Precision::inexact(ScalarValue::from(3u64)));

        let result = array
            .apply(&all_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 3).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_non_null_from_null_count() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        array
            .statistics()
            .set(Stat::NullCount, Precision::exact(ScalarValue::from(0u64)));

        let result = array
            .apply(&all_non_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_non_null_true_from_inexact_zero_null_count() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        array
            .statistics()
            .set(Stat::NullCount, Precision::inexact(ScalarValue::from(0u64)));

        let result = array
            .apply(&all_non_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_returns_null_for_inexact_nonzero_null_count_as_all_non_null() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None]).into_array();
        array
            .statistics()
            .set(Stat::NullCount, Precision::inexact(ScalarValue::from(2u64)));

        let result = array
            .apply(&all_non_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 4).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_rejects_all_nan_for_non_float() -> VortexResult<()> {
        let array = PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let result = array
            .apply(&all_nan(root()))
            .and_then(|array| array.execute::<Canonical>(&mut ctx));

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_nan_from_nan_count() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(f32::NAN), Some(f32::NAN), Some(f32::NAN)])
                .into_array();
        array
            .statistics()
            .set(Stat::NaNCount, Precision::exact(ScalarValue::from(3u64)));

        let result = array
            .apply(&all_nan(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_nan_false_from_inexact_low_nan_count() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(f32::NAN), Some(1.0f32), Some(f32::NAN)])
                .into_array();
        array
            .statistics()
            .set(Stat::NaNCount, Precision::inexact(ScalarValue::from(2u64)));

        let result = array
            .apply(&all_nan(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(false, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_returns_null_for_inexact_full_nan_count_as_all_nan() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(f32::NAN), Some(1.0f32), Some(f32::NAN)])
                .into_array();
        array
            .statistics()
            .set(Stat::NaNCount, Precision::inexact(ScalarValue::from(3u64)));

        let result = array
            .apply(&all_nan(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 3).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_non_nan_true_from_inexact_zero_nan_count() -> VortexResult<()> {
        let array = buffer![1.0f32, 2.0, 3.0].into_array();
        array
            .statistics()
            .set(Stat::NaNCount, Precision::inexact(ScalarValue::from(0u64)));

        let result = array
            .apply(&all_non_nan(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_returns_null_for_inexact_nonzero_nan_count_as_all_non_nan() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(1.0f32), Some(f32::NAN), Some(3.0)])
            .into_array();
        array
            .statistics()
            .set(Stat::NaNCount, Precision::inexact(ScalarValue::from(1u64)));

        let result = array
            .apply(&all_non_nan(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 3).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_min_and_max() -> VortexResult<()> {
        let array = buffer![3i32, 1, 2].into_array();
        array
            .statistics()
            .set(Stat::Min, Precision::exact(ScalarValue::from(1i32)));
        array
            .statistics()
            .set(Stat::Max, Precision::exact(ScalarValue::from(3i32)));

        let min_result = array
            .clone()
            .apply(&stat(
                root(),
                Stat::Min
                    .aggregate_fn()
                    .vortex_expect("min should have an aggregate function"),
            ))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();
        let expected_min =
            ConstantArray::new(Scalar::primitive(1i32, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(min_result, expected_min);

        let max_result = array
            .apply(&stat(
                root(),
                Stat::Max
                    .aggregate_fn()
                    .vortex_expect("max should have an aggregate function"),
            ))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();
        let expected_max =
            ConstantArray::new(Scalar::primitive(3i32, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(max_result, expected_max);

        Ok(())
    }
}
