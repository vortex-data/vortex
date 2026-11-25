// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-registered derivations of a scalar function's result statistics from the statistics
//! of its inputs.
//!
//! Stats rewrite rules bound sub-expressions of a predicate, such as the min and max of
//! `cast(col)` in `cast(col) > 5`. Statistics are stored only for columns, so a statistic of
//! `cast(col)` must be expressed in terms of the statistics of `col`. A [`StatDerivation`]
//! registered for a scalar function says how.

use std::fmt::Debug;
use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_session::VortexSession;

use crate::stats::bound;
use crate::stats::session::StatsSessionExt;

/// Shared reference to a statistic derivation.
pub type StatDerivationRef = Arc<dyn StatDerivation>;

/// Expresses the statistics of a scalar function's result in terms of the statistics of its
/// inputs.
///
/// At most one derivation is registered per scalar function. When one is registered, it decides
/// every statistic of expressions rooted at that function: stats rewrites never read a statistic
/// stored for such an expression itself.
pub trait StatDerivation: Debug + Send + Sync + 'static {
    /// Returns the scalar function id whose statistics this derivation describes.
    fn scalar_fn_id(&self) -> ScalarFnId;

    /// Expresses the statistic `aggregate_fn` of `expr`, whose root scalar function id is
    /// [`Self::scalar_fn_id`], as an expression over statistics of other expressions.
    ///
    /// Returns `None` when the statistic cannot be known from statistics.
    fn derive_stat(
        &self,
        expr: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        session: &VortexSession,
    ) -> Option<BoundExpression>;
}

/// Expresses the statistic `aggregate_fn` of `expr` as an expression over the statistics that
/// can be read for the current scope.
///
/// If a [`StatDerivation`] is registered for the root scalar function of `expr`, it decides the
/// statistic. Otherwise this is a reference to `aggregate_fn` of `expr` itself. Returns `None`
/// when the statistic cannot be known, for example when the aggregate does not support the dtype
/// of `expr`.
pub fn stat_expr(
    expr: &BoundExpression,
    aggregate_fn: &AggregateFnRef,
    session: &VortexSession,
) -> Option<BoundExpression> {
    if let Some(scalar_fn) = expr.as_scalar()
        && let Some(derivation) = session.stats().stat_derivation_for(scalar_fn.id())
    {
        return derivation.derive_stat(expr, aggregate_fn, session);
    }

    // The aggregate may not support the expression's dtype, e.g. min/max over structs,
    // even when the predicate itself is well-typed. Such stats cannot be lowered later,
    // so do not reference them in the rewrite.
    aggregate_fn
        .return_dtype(expr.dtype())
        .is_some()
        .then(|| bound::stat(expr.clone(), aggregate_fn.clone()))
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::all_nan::AllNan;
    use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
    use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
    use vortex_array::aggregate_fn::fns::all_null::AllNull;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::nan_count::NanCount;
    use vortex_array::aggregate_fn::fns::null_count::NullCount;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtId;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayExprExt;
    use crate::bound::lit;
    use crate::lambda::row_count::substitute_row_count;
    use crate::stats::derive::stat_expr;

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct EmptyMeta;

    impl fmt::Display for EmptyMeta {
        fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
            Ok(())
        }
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct FloatExt;

    impl ExtVTable for FloatExt {
        type Metadata = EmptyMeta;
        type StoredMetadata = Self::Metadata;
        type NativeValue<'a> = &'a ScalarValue;

        #[expect(clippy::disallowed_methods, reason = "test-only id")]
        fn id(&self) -> ExtId {
            ExtId::new("test.float_ext")
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            Ok(vec![])
        }

        fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::StoredMetadata> {
            Ok(EmptyMeta)
        }

        fn metadata_from_storage(
            &self,
            stored_metadata: Self::StoredMetadata,
            _storage_dtype: &DType,
        ) -> VortexResult<Self::Metadata> {
            Ok(stored_metadata)
        }

        fn storage_dtype(_metadata: &Self::Metadata) -> DType {
            DType::Primitive(PType::F64, Nullability::NonNullable)
        }

        fn unpack_native<'a>(
            _extension_dtype: &'a ExtDType<Self>,
            storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok(storage_value)
        }
    }

    #[test]
    fn literal_stat_derivation_matches_direct_aggregation() -> VortexResult<()> {
        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();
        let float_ext =
            ExtDType::<FloatExt>::try_new(EmptyMeta, Nullability::NonNullable)?.erased();

        let literals = [
            Scalar::from(5.0f64),
            Scalar::from(f64::NAN),
            Scalar::null(DType::Primitive(PType::F64, Nullability::Nullable)),
            Scalar::extension_ref(float_ext, Scalar::from(f64::NAN)),
        ];
        let aggregates = [
            ("null_count", NullCount.bind(EmptyOptions)),
            ("nan_count", NanCount.bind(EmptyOptions)),
            ("all_null", AllNull.bind(EmptyOptions)),
            ("all_non_null", AllNonNull.bind(EmptyOptions)),
            ("all_nan", AllNan.bind(EmptyOptions)),
            ("all_non_nan", AllNonNan.bind(EmptyOptions)),
            ("min_skip", Min.bind(NumericalAggregateOpts::skip_nans())),
            ("max_skip", Max.bind(NumericalAggregateOpts::skip_nans())),
            (
                "min_include",
                Min.bind(NumericalAggregateOpts::include_nans()),
            ),
            (
                "max_include",
                Max.bind(NumericalAggregateOpts::include_nans()),
            ),
        ];

        for n in [0usize, 1, 3] {
            for value in &literals {
                for (label, aggregate) in &aggregates {
                    let Some(expr) = stat_expr(&lit(value.clone()), aggregate, &session) else {
                        continue;
                    };

                    let values = ConstantArray::new(value.clone(), n).into_array();
                    let mut acc = aggregate.accumulator(values.dtype())?;
                    acc.accumulate(&values, &mut ctx)?;
                    let direct = acc.finish()?;

                    let group_len = ConstantArray::new(Scalar::from(n as u64), 1).into_array();
                    let projected = group_len.clone().apply_bound(&expr)?;
                    let derived =
                        substitute_row_count(projected, &group_len)?.execute_scalar(0, &mut ctx)?;

                    assert_eq!(
                        direct.value(),
                        derived.value(),
                        "mismatch for rows={n}, literal={value}, agg={label}: direct={direct}, derived={derived}"
                    );
                }
            }
        }
        Ok(())
    }
}
