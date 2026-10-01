// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression constructors for statistics backed by aggregate functions.

use vortex_error::VortexExpect;

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
use crate::expr::BoundExpression;
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

fn bound_stat(expr: BoundExpression, aggregate_fn: AggregateFnRef) -> BoundExpression {
    StatFn
        .try_new_bound_expr(StatOptions::new(aggregate_fn), [expr])
        .vortex_expect("stat expressions must use an aggregate supported by the child dtype")
}

/// Creates `stat(expr, min_max)`, returning a nullable `{ min, max }` struct statistic.
pub fn min_max(expr: Expression) -> Expression {
    // Statistics follow NaN-skipping semantics; request it explicitly rather than via the default.
    stat(expr, MinMax.bind(NumericalAggregateOpts::skip_nans()))
}

fn bound_min_max(expr: BoundExpression) -> BoundExpression {
    bound_stat(expr, MinMax.bind(NumericalAggregateOpts::skip_nans()))
}

/// Creates `stat(expr, sum)`, returning a nullable sum statistic.
pub fn sum(expr: Expression) -> Expression {
    // Statistics follow NaN-skipping semantics; request it explicitly rather than via the default.
    stat(expr, Sum.bind(NumericalAggregateOpts::skip_nans()))
}

fn bound_sum(expr: BoundExpression) -> BoundExpression {
    bound_stat(expr, Sum.bind(NumericalAggregateOpts::skip_nans()))
}

/// Creates `stat(expr, null_count)`, returning a nullable null-count statistic.
pub fn null_count(expr: Expression) -> Expression {
    stat(expr, NullCount.bind(EmptyOptions))
}

fn bound_null_count(expr: BoundExpression) -> BoundExpression {
    bound_stat(expr, NullCount.bind(EmptyOptions))
}

/// Creates `stat(expr, all_null)`, returning a nullable all-null statistic.
pub fn all_null(expr: Expression) -> Expression {
    stat(expr, AllNull.bind(EmptyOptions))
}

fn bound_all_null(expr: BoundExpression) -> BoundExpression {
    bound_stat(expr, AllNull.bind(EmptyOptions))
}

/// Creates `stat(expr, all_nan)`, returning a nullable all-NaN statistic.
pub fn all_nan(expr: Expression) -> Expression {
    stat(expr, AllNan.bind(EmptyOptions))
}

fn bound_all_nan(expr: BoundExpression) -> BoundExpression {
    bound_stat(expr, AllNan.bind(EmptyOptions))
}

/// Creates `stat(expr, all_non_null)`, returning a nullable all-non-null statistic.
pub fn all_non_null(expr: Expression) -> Expression {
    stat(expr, AllNonNull.bind(EmptyOptions))
}

fn bound_all_non_null(expr: BoundExpression) -> BoundExpression {
    bound_stat(expr, AllNonNull.bind(EmptyOptions))
}

/// Creates `stat(expr, all_non_nan)`, returning a nullable all-non-NaN statistic.
pub fn all_non_nan(expr: Expression) -> Expression {
    stat(expr, AllNonNan.bind(EmptyOptions))
}

fn bound_all_non_nan(expr: BoundExpression) -> BoundExpression {
    bound_stat(expr, AllNonNan.bind(EmptyOptions))
}

/// Creates `stat(expr, nan_count)`, returning a nullable NaN-count statistic.
pub fn nan_count(expr: Expression) -> Expression {
    stat(expr, NanCount.bind(EmptyOptions))
}

fn bound_nan_count(expr: BoundExpression) -> BoundExpression {
    bound_stat(expr, NanCount.bind(EmptyOptions))
}

/// Constructors for statistic expressions whose input has already been bound.
///
/// These mirror the constructors in [`crate::stats`] and panic when the aggregate does not support
/// the input dtype.
pub mod bound {
    use crate::aggregate_fn::AggregateFnRef;
    use crate::expr::BoundExpression;

    /// Creates a bound expression that reads a stored aggregate statistic.
    pub fn stat(expr: BoundExpression, aggregate_fn: AggregateFnRef) -> BoundExpression {
        super::bound_stat(expr, aggregate_fn)
    }

    /// Creates a bound nullable `{ min, max }` statistic expression.
    pub fn min_max(expr: BoundExpression) -> BoundExpression {
        super::bound_min_max(expr)
    }

    /// Creates a bound nullable sum statistic expression.
    pub fn sum(expr: BoundExpression) -> BoundExpression {
        super::bound_sum(expr)
    }

    /// Creates a bound nullable null-count statistic expression.
    pub fn null_count(expr: BoundExpression) -> BoundExpression {
        super::bound_null_count(expr)
    }

    /// Creates a bound nullable all-null statistic expression.
    pub fn all_null(expr: BoundExpression) -> BoundExpression {
        super::bound_all_null(expr)
    }

    /// Creates a bound nullable all-NaN statistic expression.
    pub fn all_nan(expr: BoundExpression) -> BoundExpression {
        super::bound_all_nan(expr)
    }

    /// Creates a bound nullable all-non-null statistic expression.
    pub fn all_non_null(expr: BoundExpression) -> BoundExpression {
        super::bound_all_non_null(expr)
    }

    /// Creates a bound nullable all-non-NaN statistic expression.
    pub fn all_non_nan(expr: BoundExpression) -> BoundExpression {
        super::bound_all_non_nan(expr)
    }

    /// Creates a bound nullable NaN-count statistic expression.
    pub fn nan_count(expr: BoundExpression) -> BoundExpression {
        super::bound_nan_count(expr)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::StatFn;
    use super::StatOptions;
    use super::all_nan;
    use super::all_non_nan;
    use super::all_non_null;
    use super::all_null;
    use super::bound as bound_stats;
    use super::null_count;
    use super::stat;
    use super::sum;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::all_nan::AllNan;
    use crate::aggregate_fn::fns::all_null::AllNull;
    use crate::aggregate_fn::fns::is_sorted::IsSorted;
    use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use crate::aggregate_fn::fns::max::Max;
    use crate::aggregate_fn::fns::min::Min;
    use crate::aggregate_fn::fns::nan_count::NanCount;
    use crate::aggregate_fn::fns::null_count::NullCount;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::array_session;
    use crate::arrays::Chunked;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::bound as bound_expr;
    use crate::expr::root;
    use crate::expr::stats::Precision;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::VecExecutionArgs;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

    #[test]
    fn bound_stats_constructor_preserves_child_and_dtype() -> VortexResult<()> {
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let root = bound_expr::root(input_dtype.clone());
        let bound = bound_stats::sum(root.clone());

        assert_eq!(bound.children(), &[root]);
        assert_eq!(
            bound.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
        assert_eq!(bound, sum(crate::expr::root()).bind(&input_dtype)?);
        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_sum() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        let sum_scalar = Scalar::primitive(6i64, Nullability::Nullable);
        array.aggregations().insert_result(
            Sum.bind(NumericalAggregateOpts::skip_nans()),
            Precision::exact(sum_scalar),
        )?;

        let result = array
            .apply(&sum(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::primitive(6i64, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

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
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_sum_per_chunk() -> VortexResult<()> {
        let chunk0 = buffer![1i32, 2].into_array();
        let sum_scalar = Scalar::primitive(3i64, Nullability::Nullable);
        chunk0.aggregations().insert_result(
            Sum.bind(NumericalAggregateOpts::skip_nans()),
            Precision::exact(sum_scalar),
        )?;
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
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_null_count() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None]).into_array();
        let null_count_scalar = Scalar::primitive(2u64, Nullability::NonNullable);
        array.aggregations().insert_result(
            NullCount.bind(EmptyOptions),
            Precision::exact(null_count_scalar),
        )?;

        let result = array
            .apply(&null_count(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::primitive(2u64, Nullability::Nullable), 4).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_null_from_null_count() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]).into_array();
        array.aggregations().insert_result(
            NullCount.bind(EmptyOptions),
            Precision::exact(Scalar::from(3u64)),
        )?;

        let result = array
            .apply(&all_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_null_false_from_inexact_low_null_count() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, Some(2), None]).into_array();
        array.aggregations().insert_result(
            NullCount.bind(EmptyOptions),
            Precision::inexact(Scalar::from(2u64)),
        )?;

        let result = array
            .apply(&all_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(false, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_returns_null_for_inexact_full_null_count_as_all_null() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, Some(2), None]).into_array();
        array.aggregations().insert_result(
            NullCount.bind(EmptyOptions),
            Precision::inexact(Scalar::from(3u64)),
        )?;

        let result = array
            .apply(&all_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_non_null_from_null_count() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        array.aggregations().insert_result(
            NullCount.bind(EmptyOptions),
            Precision::exact(Scalar::from(0u64)),
        )?;

        let result = array
            .apply(&all_non_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_non_null_true_from_inexact_zero_null_count() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        array.aggregations().insert_result(
            NullCount.bind(EmptyOptions),
            Precision::inexact(Scalar::from(0u64)),
        )?;

        let result = array
            .apply(&all_non_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_returns_null_for_inexact_nonzero_null_count_as_all_non_null() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None]).into_array();
        array.aggregations().insert_result(
            NullCount.bind(EmptyOptions),
            Precision::inexact(Scalar::from(2u64)),
        )?;

        let result = array
            .apply(&all_non_null(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 4).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

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
        array.aggregations().insert_result(
            NanCount.bind(EmptyOptions),
            Precision::exact(Scalar::from(3u64)),
        )?;

        let result = array
            .apply(&all_nan(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_nan_false_from_inexact_low_nan_count() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(f32::NAN), Some(1.0f32), Some(f32::NAN)])
                .into_array();
        array.aggregations().insert_result(
            NanCount.bind(EmptyOptions),
            Precision::inexact(Scalar::from(2u64)),
        )?;

        let result = array
            .apply(&all_nan(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(false, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_returns_null_for_inexact_full_nan_count_as_all_nan() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(f32::NAN), Some(1.0f32), Some(f32::NAN)])
                .into_array();
        array.aggregations().insert_result(
            NanCount.bind(EmptyOptions),
            Precision::inexact(Scalar::from(3u64)),
        )?;

        let result = array
            .apply(&all_nan(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_all_non_nan_true_from_inexact_zero_nan_count() -> VortexResult<()> {
        let array = buffer![1.0f32, 2.0, 3.0].into_array();
        array.aggregations().insert_result(
            NanCount.bind(EmptyOptions),
            Precision::inexact(Scalar::from(0u64)),
        )?;

        let result = array
            .apply(&all_non_nan(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_returns_null_for_inexact_nonzero_nan_count_as_all_non_nan() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(1.0f32), Some(f32::NAN), Some(3.0)])
            .into_array();
        array.aggregations().insert_result(
            NanCount.bind(EmptyOptions),
            Precision::inexact(Scalar::from(1u64)),
        )?;

        let result = array
            .apply(&all_non_nan(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[rstest]
    #[case::all_null(false)]
    #[case::all_nan(true)]
    fn stat_expr_uses_input_length_for_count_metadata(#[case] nan: bool) -> VortexResult<()> {
        let (array, count_fn, all_fn) = if nan {
            (
                buffer![f64::NAN, f64::NAN].into_array(),
                NanCount.bind(EmptyOptions),
                AllNan.bind(EmptyOptions),
            )
        } else {
            (
                PrimitiveArray::from_option_iter([None::<f64>, None]).into_array(),
                NullCount.bind(EmptyOptions),
                AllNull.bind(EmptyOptions),
            )
        };
        array
            .aggregations()
            .insert_result(count_fn, Precision::Exact(Scalar::from(2u64)))?;

        let args = VecExecutionArgs::new(vec![array], 5);
        let mut ctx = SESSION.create_execution_ctx();
        let result = StatFn.execute(&StatOptions::new(all_fn), &args, &mut ctx)?;
        let expected =
            ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 5).into_array();
        assert_arrays_eq!(result, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn stat_expr_does_not_reconstruct_sorted_partial_from_boolean() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        let options = IsSortedOptions { strict: false };
        array.aggregations().insert_result(
            IsSorted.bind(options.clone()),
            Precision::Exact(Scalar::from(true)),
        )?;
        let partial_dtype = IsSorted
            .partial_dtype(&options, array.dtype())
            .expect("sortedness supports primitive inputs");

        let result = array
            .apply(&stat(root(), IsSorted.bind(options)))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();
        let expected = ConstantArray::new(Scalar::null(partial_dtype), 3).into_array();
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn stat_expr_reads_cached_min_and_max() -> VortexResult<()> {
        let array = buffer![3i32, 1, 2].into_array();
        array.aggregations().insert_result(
            Min.bind(NumericalAggregateOpts::skip_nans()),
            Precision::exact(Scalar::primitive(1i32, Nullability::Nullable)),
        )?;
        array.aggregations().insert_result(
            Max.bind(NumericalAggregateOpts::skip_nans()),
            Precision::exact(Scalar::primitive(3i32, Nullability::Nullable)),
        )?;

        let min_result = array
            .clone()
            .apply(&stat(root(), Min.bind(NumericalAggregateOpts::skip_nans())))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();
        let expected_min =
            ConstantArray::new(Scalar::primitive(1i32, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(
            min_result,
            expected_min,
            &mut SESSION.create_execution_ctx()
        );

        let max_result = array
            .apply(&stat(root(), Max.bind(NumericalAggregateOpts::skip_nans())))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_array();
        let expected_max =
            ConstantArray::new(Scalar::primitive(3i32, Nullability::Nullable), 3).into_array();
        assert_arrays_eq!(
            max_result,
            expected_max,
            &mut SESSION.create_execution_ctx()
        );

        Ok(())
    }
}
