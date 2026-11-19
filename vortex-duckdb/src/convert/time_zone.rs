// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Time zones of the timestamp values that DuckDB compares with a zoned Vortex column.
//!
//! A DuckDB zoned timestamp is a UTC instant without a time zone, so its values convert to Vortex
//! timestamps in `UTC`. A Vortex column keeps the time zone it was written with, and Vortex only
//! compares timestamps of the same time zone. Both sides hold UTC instants, so a value takes the
//! time zone of the column it is compared with.

use std::sync::Arc;

use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::dynamic;
use vortex::expr::lit;
use vortex::expr::traversal::NodeExt;
use vortex::expr::traversal::Transformed;
use vortex::extension::datetime::AnyTemporal;
use vortex::extension::datetime::TemporalMetadata;
use vortex::scalar::Scalar;
use vortex::scalar_fn::fns::dynamic::DynamicComparison;
use vortex::scalar_fn::fns::literal::Literal;

/// Gives the timestamp literals and dynamic values in `expr` the time zone of the zoned timestamp
/// that they are compared with. `scope` is the dtype that `expr` is evaluated over.
pub(super) fn align_time_zones(expr: Expression, scope: &DType) -> VortexResult<Expression> {
    let aligned = expr.transform_down(|node| {
        // An operand that is not over `scope`, such as a hive partition column, has no dtype
        // here. DuckDB evaluates it, or binding reports it.
        let Some(zoned) = node
            .children()
            .iter()
            .filter(|child| !child.is::<Literal>())
            .filter_map(|child| child.return_dtype(scope).ok())
            .find(|dtype| time_zone(dtype).is_some())
        else {
            return Ok(Transformed::no(node));
        };

        if let Some(options) = node.as_opt::<DynamicComparison>() {
            // The dtype of the dynamic value cannot be read here. DuckDB's values convert in
            // `UTC`, so a column in `UTC` needs no new comparison.
            if time_zone(&zoned) == Some("UTC") {
                return Ok(Transformed::no(node));
            }
            let (operator, default) = (options.operator(), options.default());
            let options = options.clone();
            let aligned = dynamic(
                operator,
                move || options.scalar().and_then(Scalar::into_value),
                zoned.as_nullable(),
                default,
                node.child(0).clone(),
            );
            return Ok(Transformed::yes(aligned));
        }

        let mut changed = false;
        let children = node
            .children()
            .iter()
            .map(|child| {
                let Some(literal) = child.as_opt::<Literal>() else {
                    return Ok(child.clone());
                };
                let Some(dtype) = aligned_dtype(literal.dtype(), &zoned) else {
                    return Ok(child.clone());
                };
                changed = true;
                Scalar::try_new(dtype, literal.value().cloned()).map(lit)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        if !changed {
            return Ok(Transformed::no(node));
        }
        node.with_children(children).map(Transformed::yes)
    })?;
    Ok(aligned.value)
}

/// The time zone of a zoned timestamp dtype.
fn time_zone(dtype: &DType) -> Option<&str> {
    match dtype.as_extension_opt()?.metadata_opt::<AnyTemporal>()? {
        TemporalMetadata::Timestamp(_, tz) => tz.as_deref(),
        _ => None,
    }
}

/// The dtype of a literal with its time zone replaced by that of `zoned`, or `None` if the literal
/// is not a timestamp, or a list of timestamps, that differs from `zoned` in its time zone alone.
fn aligned_dtype(literal: &DType, zoned: &DType) -> Option<DType> {
    if let DType::List(element, nullability) = literal {
        let element = aligned_dtype(element, zoned)?;
        return Some(DType::List(Arc::new(element), *nullability));
    }

    let (literal_ext, zoned_ext) = (literal.as_extension_opt()?, zoned.as_extension_opt()?);
    let TemporalMetadata::Timestamp(literal_unit, Some(literal_tz)) =
        literal_ext.metadata_opt::<AnyTemporal>()?
    else {
        return None;
    };
    let TemporalMetadata::Timestamp(zoned_unit, Some(zoned_tz)) =
        zoned_ext.metadata_opt::<AnyTemporal>()?
    else {
        return None;
    };
    (literal_unit == zoned_unit && literal_tz != zoned_tz)
        .then(|| zoned.with_nullability(literal.nullability()))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::dtype::StructFields;
    use vortex::error::VortexResult;
    use vortex::expr::Expression;
    use vortex::expr::between;
    use vortex::expr::col;
    use vortex::expr::gt_eq;
    use vortex::expr::in_list_of;
    use vortex::expr::lit;
    use vortex::extension::datetime::TimeUnit;
    use vortex::extension::datetime::Timestamp;
    use vortex::extension::datetime::TimestampOptions;
    use vortex::scalar::Scalar;
    use vortex::scalar::ScalarValue;
    use vortex::scalar_fn::fns::between::BetweenOptions;
    use vortex::scalar_fn::fns::between::StrictComparison;

    use super::align_time_zones;

    fn timestamp(unit: TimeUnit, tz: Option<&str>, value: i64) -> VortexResult<Scalar> {
        Ok(Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit,
                tz: tz.map(Into::into),
            },
            Scalar::try_new(
                DType::Primitive(PType::I64, Nullability::Nullable),
                Some(ScalarValue::from(value)),
            )?,
        ))
    }

    /// A scope with the single column `t` of the dtype of `like`.
    fn scope_of(like: &Scalar) -> DType {
        DType::Struct(
            StructFields::from_iter([("t", like.dtype().clone())]),
            Nullability::NonNullable,
        )
    }

    fn comparison(value: Scalar) -> VortexResult<Expression> {
        Ok(gt_eq(col("t"), lit(value)))
    }

    fn range(value: Scalar) -> VortexResult<Expression> {
        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        };
        Ok(between(col("t"), lit(value.clone()), lit(value), options))
    }

    fn membership(value: Scalar) -> VortexResult<Expression> {
        in_list_of(col("t"), value.dtype().clone(), vec![value])
    }

    #[rstest]
    #[case::comparison(comparison)]
    #[case::between(range)]
    #[case::in_list(membership)]
    fn test_align_time_zones_literal_takes_column_zone(
        #[case] filter: fn(Scalar) -> VortexResult<Expression>,
    ) -> VortexResult<()> {
        let utc = timestamp(TimeUnit::Microseconds, Some("UTC"), 1)?;
        let new_york = timestamp(TimeUnit::Microseconds, Some("America/New_York"), 1)?;
        let scope = scope_of(&new_york);

        assert!(filter(utc.clone())?.bind(&scope).is_err());

        let aligned = align_time_zones(filter(utc)?, &scope)?;
        assert_eq!(aligned, filter(new_york)?);
        aligned.bind(&scope)?;

        Ok(())
    }

    #[rstest]
    #[case::same_zone(Some("UTC"), TimeUnit::Microseconds, Some("UTC"))]
    #[case::other_unit(Some("America/New_York"), TimeUnit::Nanoseconds, Some("UTC"))]
    #[case::literal_without_zone(Some("America/New_York"), TimeUnit::Microseconds, None)]
    #[case::column_without_zone(None, TimeUnit::Microseconds, Some("UTC"))]
    fn test_align_time_zones_leaves_other_literals(
        #[case] column_tz: Option<&str>,
        #[case] literal_unit: TimeUnit,
        #[case] literal_tz: Option<&str>,
    ) -> VortexResult<()> {
        let scope = scope_of(&timestamp(TimeUnit::Microseconds, column_tz, 1)?);
        let filter = comparison(timestamp(literal_unit, literal_tz, 1)?)?;

        assert_eq!(align_time_zones(filter.clone(), &scope)?, filter);

        Ok(())
    }
}
