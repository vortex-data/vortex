// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression constructors for statistics backed by aggregate functions.

use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
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
    stat(expr, MinMax.bind(EmptyOptions))
}

/// Creates `stat(expr, sum)`, returning a nullable sum statistic.
pub fn sum(expr: Expression) -> Expression {
    stat(expr, Sum.bind(EmptyOptions))
}

/// Creates `stat(expr, null_count)`, returning a nullable null-count statistic.
pub fn null_count(expr: Expression) -> Expression {
    stat(expr, NullCount.bind(EmptyOptions))
}

/// Creates `stat(expr, nan_count)`, returning a nullable NaN-count statistic.
pub fn nan_count(expr: Expression) -> Expression {
    stat(expr, NanCount.bind(EmptyOptions))
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::stat;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFn;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::sum::Sum;
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
    use crate::validity::Validity;

    #[test]
    fn stat_expr_reads_cached_sum() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        let sum_scalar = Scalar::primitive(6i64, Nullability::Nullable);
        array.statistics().set(
            Stat::Sum,
            Precision::exact(sum_scalar.into_value().vortex_expect("non-null sum")),
        );

        let result = array
            .apply(&stat(root(), AggregateFn::new(Sum, EmptyOptions).erased()))?
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())?
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
            .apply(&stat(root(), AggregateFn::new(Sum, EmptyOptions).erased()))?
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())?
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

        let result = chunked.apply(&stat(root(), AggregateFn::new(Sum, EmptyOptions).erased()))?;

        let chunked_result = result
            .as_opt::<Chunked>()
            .vortex_expect("stat expression should preserve chunked alignment");
        assert_eq!(chunked_result.nchunks(), 2);

        let result = result
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())?
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
            .apply(&super::null_count(root()))?
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())?
            .into_array();

        let expected =
            ConstantArray::new(Scalar::primitive(2u64, Nullability::Nullable), 4).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }
}
