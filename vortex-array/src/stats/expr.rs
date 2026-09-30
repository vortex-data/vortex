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
/// a nullable all-null array with the aggregate state type.
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
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::is_constant::IsConstant;
    use crate::aggregate_fn::fns::is_sorted::IsSorted;
    use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use crate::aggregate_fn::fns::max::Max;
    use crate::aggregate_fn::fns::min::Min;
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
    use crate::expr::Expression;
    use crate::expr::bound as bound_expr;
    use crate::expr::root;
    use crate::scalar::Scalar;

    #[test]
    fn bound_stats_constructor_preserves_child_and_dtype() -> VortexResult<()> {
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let root = bound_expr::root(input_dtype.clone());
        let bound = bound::sum(root.clone());
        assert_eq!(bound.children(), &[root]);
        assert_eq!(
            bound.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
        assert_eq!(bound, sum(crate::expr::root()).bind(&input_dtype)?);
        Ok(())
    }

    #[rstest]
    #[case::sum(sum(root()))]
    #[case::null_count(null_count(root()))]
    #[case::nan_count(nan_count(root()))]
    #[case::all_null(all_null(root()))]
    #[case::all_non_null(all_non_null(root()))]
    #[case::all_nan(all_nan(root()))]
    #[case::all_non_nan(all_non_nan(root()))]
    #[case::min(stat(root(), Min.bind(NumericalAggregateOpts::skip_nans())))]
    #[case::max(stat(root(), Max.bind(NumericalAggregateOpts::skip_nans())))]
    #[case::min_max(min_max(root()))]
    #[case::is_constant(stat(root(), IsConstant.bind(EmptyOptions)))]
    #[case::is_sorted(stat(root(), IsSorted.bind(IsSortedOptions { strict: false })))]
    fn unresolved_stats_are_typed_nulls(#[case] expression: Expression) -> VortexResult<()> {
        let array = buffer![1.0f64, f64::NAN, 3.0].into_array();
        let expression = expression.bind(array.dtype())?;
        let expected = ConstantArray::new(Scalar::null(expression.dtype().clone()), array.len());
        let mut ctx = array_session().create_execution_ctx();
        let result = array
            .apply_bound(&expression)?
            .execute::<Canonical>(&mut ctx)?
            .into_array();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn unresolved_stats_preserve_chunk_alignment() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let array = ChunkedArray::try_new(
            vec![
                buffer![1i32, 2].into_array(),
                buffer![3i32, 4, 5].into_array(),
            ],
            dtype,
        )?
        .into_array();
        let result = array.apply(&sum(root()))?;
        assert_eq!(result.as_::<Chunked>().nchunks(), 2);
        let expected = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
            5,
        );
        assert_arrays_eq!(
            result,
            expected,
            &mut array_session().create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn all_nan_rejects_non_float_input() {
        let array = PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array();
        assert!(array.apply(&all_nan(root())).is_err());
    }
}
