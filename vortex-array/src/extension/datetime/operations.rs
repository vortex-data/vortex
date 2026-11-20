// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Named temporal conversions.
//!
//! These operations select a rounding policy or interpret a time zone explicitly. Unit conversion
//! preserves the source zone, and calendar extraction uses that zone when present.

use std::fmt;
use std::sync::Arc;

use jiff::tz::TimeZone;
use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::Date;
use super::TemporalMetadata;
use super::Time;
use super::TimeUnit;
use super::Timestamp;
use super::convert::check_time;
use super::convert::nanos_per_unit;
use super::convert::temporal_metadata;
use super::convert::transform_temporal;
use super::convert::units_per_day;
use super::resolve_time_zone;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::proto::expr::TemporalConvertOpts;
use crate::proto::expr::temporal_convert_opts as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;

/// How to handle fractional counts in a coarser temporal unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TemporalRounding {
    /// Require the value to be exactly representable in the target unit.
    Exact,
    /// Discard the remainder toward zero.
    TowardZero,
    /// Round toward negative infinity.
    Floor,
    /// Round to the nearest count, resolving ties away from zero.
    NearestTiesAwayFromZero,
}

impl fmt::Display for TemporalRounding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Exact => "exact",
            Self::TowardZero => "toward_zero",
            Self::Floor => "floor",
            Self::NearestTiesAwayFromZero => "nearest_ties_away_from_zero",
        })
    }
}

impl TemporalRounding {
    /// Divide a signed count by the positive factor between two temporal units.
    fn divide(self, value: i64, divisor: i64) -> VortexResult<i64> {
        let quotient = value / divisor;
        let remainder = value % divisor;

        Ok(match self {
            Self::Exact => {
                vortex_ensure_eq!(
                    remainder,
                    0,
                    "Expected an exact temporal conversion, got {value}"
                );
                quotient
            }
            Self::TowardZero => quotient,
            Self::Floor => value.div_euclid(divisor),
            Self::NearestTiesAwayFromZero if remainder.abs() * 2 >= divisor => {
                quotient + remainder.signum()
            }
            Self::NearestTiesAwayFromZero => quotient,
        })
    }
}

/// Convert a [`Date`], [`Time`], or [`Timestamp`] to a coarser `unit` with an explicit
/// [`TemporalRounding`].
///
/// `unit` **must** be coarser than the input unit and valid for the input type (a [`Date`]
/// supports only days and milliseconds), or binding fails. At execution,
/// [`TemporalRounding::Exact`] fails on a count with a remainder, and a [`Time`] fails when it
/// rounds up to midnight, because the result must stay within the same day. A [`Timestamp`] keeps
/// its time zone, and nulls stay null.
pub fn change_time_unit(
    child: Expression,
    unit: TimeUnit,
    rounding: TemporalRounding,
) -> Expression {
    TemporalConvert.new_expr(TemporalOperation::ChangeUnit(unit, rounding), [child])
}

/// Extract the local calendar date of a [`Timestamp`] as a [`Date`] in days.
///
/// A timestamp with a time zone uses the offset of that zone at its instant, and a timestamp
/// without a time zone already counts local time. The input **must** be a [`Timestamp`], or
/// binding fails. A date outside the `i32` range of days fails at execution.
pub fn timestamp_to_date(child: Expression) -> Expression {
    TemporalConvert.new_expr(TemporalOperation::TimestampToDate, [child])
}

/// Extract the local time of day of a [`Timestamp`] as a [`Time`] in the same unit.
///
/// The time zone applies as in [`timestamp_to_date`]. The input **must** be a [`Timestamp`], or
/// binding fails.
pub fn timestamp_to_time(child: Expression) -> Expression {
    TemporalConvert.new_expr(TemporalOperation::TimestampToTime, [child])
}

/// Change the time zone of a [`Timestamp`] and keep its instant, so the stored count does not
/// change.
///
/// The input **must** be a [`Timestamp`] with a time zone, or binding fails. Use
/// [`replace_time_zone`](crate::expr::replace_time_zone) to keep the local clock time in another
/// zone instead.
pub fn convert_time_zone(child: Expression, zone: impl Into<Arc<str>>) -> Expression {
    TemporalConvert.new_expr(TemporalOperation::ConvertZone(zone.into()), [child])
}

/// Operation-specific options encoded with the temporal conversion expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TemporalOperation {
    /// Convert to a coarser unit using the supplied rounding mode.
    ChangeUnit(TimeUnit, TemporalRounding),
    /// Extract the local calendar date.
    TimestampToDate,
    /// Extract the local time of day.
    TimestampToTime,
    /// Change the display zone without changing the instant.
    ConvertZone(Arc<str>),
}

impl fmt::Display for TemporalOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChangeUnit(unit, rounding) => write!(f, "change_time_unit({unit}, {rounding})"),
            Self::TimestampToDate => f.write_str("timestamp_to_date"),
            Self::TimestampToTime => f.write_str("timestamp_to_time"),
            Self::ConvertZone(zone) => write!(f, "convert_time_zone({zone})"),
        }
    }
}

/// Executes the named operations over built-in temporal dtypes.
#[derive(Clone)]
pub(crate) struct TemporalConvert;

impl ScalarFnVTable for TemporalConvert {
    type Options = TemporalOperation;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.temporal_convert");
        *ID
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _: &Self::Options, _: usize) -> ChildName {
        ChildName::from("input")
    }

    fn serialize(&self, operation: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let operation = match operation {
            TemporalOperation::ChangeUnit(unit, rounding) => {
                let rounding = match rounding {
                    TemporalRounding::Exact => pb::Rounding::Exact,
                    TemporalRounding::TowardZero => pb::Rounding::TowardZero,
                    TemporalRounding::Floor => pb::Rounding::Floor,
                    TemporalRounding::NearestTiesAwayFromZero => {
                        pb::Rounding::NearestTiesAwayFromZero
                    }
                };
                pb::Operation::ChangeUnit(pb::ChangeUnit {
                    unit: u32::from(u8::from(*unit)),
                    rounding: rounding.into(),
                })
            }
            TemporalOperation::TimestampToDate => {
                pb::Operation::TimestampToDate(pb::TimestampToDate {})
            }
            TemporalOperation::TimestampToTime => {
                pb::Operation::TimestampToTime(pb::TimestampToTime {})
            }
            TemporalOperation::ConvertZone(time_zone) => {
                pb::Operation::ConvertZone(pb::ConvertZone {
                    time_zone: time_zone.to_string(),
                })
            }
        };

        Ok(Some(
            TemporalConvertOpts {
                operation: Some(operation),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, bytes: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        Ok(match TemporalConvertOpts::decode(bytes)?.operation {
            Some(pb::Operation::ChangeUnit(opts)) => {
                let unit = TimeUnit::try_from(u8::try_from(opts.unit)?)?;
                let rounding = match pb::Rounding::try_from(opts.rounding) {
                    Ok(pb::Rounding::Exact) => TemporalRounding::Exact,
                    Ok(pb::Rounding::TowardZero) => TemporalRounding::TowardZero,
                    Ok(pb::Rounding::Floor) => TemporalRounding::Floor,
                    Ok(pb::Rounding::NearestTiesAwayFromZero) => {
                        TemporalRounding::NearestTiesAwayFromZero
                    }
                    Err(_) => vortex_bail!(
                        "Expected a known temporal rounding mode, got {}",
                        opts.rounding
                    ),
                };
                TemporalOperation::ChangeUnit(unit, rounding)
            }
            Some(pb::Operation::TimestampToDate(_)) => TemporalOperation::TimestampToDate,
            Some(pb::Operation::TimestampToTime(_)) => TemporalOperation::TimestampToTime,
            Some(pb::Operation::ConvertZone(opts)) => {
                TemporalOperation::ConvertZone(Arc::from(opts.time_zone))
            }
            None => vortex_bail!("Missing temporal conversion operation"),
        })
    }

    fn return_dtype(&self, operation: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        let input = &args[0];
        let nullable = input.nullability();
        let metadata = temporal_metadata(input)
            .ok_or_else(|| vortex_err!("Expected a temporal dtype, got {input}"))?;

        let ext = match (operation, metadata) {
            (TemporalOperation::ChangeUnit(unit, _), metadata) => {
                vortex_ensure!(
                    nanos_per_unit(*unit) > nanos_per_unit(metadata.time_unit()),
                    "Expected a temporal unit coarser than {}, got {unit}",
                    metadata.time_unit()
                );
                match metadata {
                    TemporalMetadata::Timestamp(_, zone) => {
                        Timestamp::try_new_with_tz(*unit, zone.clone(), nullable)?.erased()
                    }
                    TemporalMetadata::Time(_) => Time::try_new(*unit, nullable)?.erased(),
                    TemporalMetadata::Date(_) => Date::try_new(*unit, nullable)?.erased(),
                }
            }
            (TemporalOperation::TimestampToDate, TemporalMetadata::Timestamp(..)) => {
                Date::new(TimeUnit::Days, nullable).erased()
            }
            (TemporalOperation::TimestampToTime, TemporalMetadata::Timestamp(unit, _)) => {
                Time::new(*unit, nullable).erased()
            }
            (TemporalOperation::ConvertZone(zone), TemporalMetadata::Timestamp(unit, Some(_))) => {
                Timestamp::try_new_with_tz(*unit, Some(Arc::clone(zone)), nullable)?.erased()
            }
            _ => vortex_bail!("Expected an input dtype that supports {operation}, got {input}"),
        };

        Ok(DType::Extension(ext))
    }

    fn execute(
        &self,
        operation: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let target = self.return_dtype(operation, std::slice::from_ref(input.dtype()))?;
        let metadata = temporal_metadata(input.dtype())
            .vortex_expect("`return_dtype` above accepts only temporal inputs");
        let source_unit = metadata.time_unit();

        match operation {
            TemporalOperation::ChangeUnit(unit, rounding) => {
                let is_time = matches!(metadata, TemporalMetadata::Time(_));
                let divisor = nanos_per_unit(*unit) / nanos_per_unit(source_unit);
                transform_temporal(input, &target, ctx, |value| {
                    if is_time {
                        check_time(value, source_unit)?;
                    }
                    let result = rounding.divide(value, divisor)?;
                    if is_time {
                        check_time(result, *unit)?;
                    }
                    Ok(result)
                })
            }
            TemporalOperation::ConvertZone(_) => {
                let target = target
                    .as_extension_opt()
                    .vortex_expect("`return_dtype` returns a timestamp for `ConvertZone`");
                if let Some(scalar) = input.as_constant() {
                    let output = Scalar::try_extension_ref(
                        target.clone(),
                        scalar.as_extension().to_storage_scalar(),
                    )?;
                    return Ok(ConstantArray::new(output, input.len()).into_array());
                }

                let input = input.execute::<ExtensionArray>(ctx)?;
                Ok(
                    ExtensionArray::try_new(target.clone(), input.storage_array().clone())?
                        .into_array(),
                )
            }
            TemporalOperation::TimestampToDate | TemporalOperation::TimestampToTime => {
                let TemporalMetadata::Timestamp(_, zone) = metadata else {
                    vortex_panic!("`return_dtype` accepts only a timestamp input for {operation}");
                };

                let zone = zone
                    .as_deref()
                    .map(|zone| resolve_time_zone(Some(zone)))
                    .transpose()?;
                let day = i128::from(units_per_day(source_unit));
                transform_temporal(input, &target, ctx, |value| {
                    let local = local_count(value, source_unit, zone.as_ref())?;
                    let result = if matches!(operation, TemporalOperation::TimestampToDate) {
                        local.div_euclid(day)
                    } else {
                        local.rem_euclid(day)
                    };
                    Ok(i64::try_from(result)?)
                })
            }
        }
    }

    fn is_strict(&self, _: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, operation: &Self::Options) -> bool {
        // Changing the zone keeps every stored instant. The other operations can overflow or reject
        // an out-of-range `Time`.
        matches!(operation, TemporalOperation::ConvertZone(_))
    }
}

/// Offset a count into local time before splitting it into a day and a time of day.
fn local_count(value: i64, unit: TimeUnit, zone: Option<&TimeZone>) -> VortexResult<i128> {
    let Some(zone) = zone else {
        return Ok(i128::from(value));
    };
    let units_per_second = nanos_per_unit(TimeUnit::Seconds) / nanos_per_unit(unit);
    let offset = if let Ok(offset) = zone.to_fixed_offset() {
        offset
    } else {
        let second = value.div_euclid(units_per_second);
        let nanos = value.rem_euclid(units_per_second) * nanos_per_unit(unit);
        zone.to_offset(jiff::Timestamp::new(second, i32::try_from(nanos)?)?)
    };
    Ok(i128::from(value) + i128::from(offset.seconds()) * i128::from(units_per_second))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::dtype::extension::ExtDTypeRef;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::test_harness::ExpectedOutcome;
    use crate::test_harness::assert_execution_paths;

    const NON_NULLABLE: Nullability = Nullability::NonNullable;

    #[rstest]
    #[case::toward_zero(TemporalRounding::TowardZero, [Some(0), Some(0), None])]
    #[case::floor(TemporalRounding::Floor, [Some(-1), Some(0), None])]
    #[case::nearest(TemporalRounding::NearestTiesAwayFromZero, [Some(-1), Some(1), None])]
    fn test_change_timestamp_unit_rounding(
        #[case] mode: TemporalRounding,
        #[case] values: [Option<i64>; 3],
    ) -> VortexResult<()> {
        let source = Timestamp::new(TimeUnit::Nanoseconds, Nullability::Nullable).erased();
        let target = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        let input = ExtensionArray::try_new(
            source,
            PrimitiveArray::from_option_iter([Some(-750_000i64), Some(500_000), None]).into_array(),
        )?
        .into_array();
        let expected = ExtensionArray::try_new(
            target,
            PrimitiveArray::from_option_iter(values).into_array(),
        )?;
        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();
        let operation = change_time_unit(root(), TimeUnit::Milliseconds, mode);
        let restored = Expression::from_proto(&operation.serialize_proto()?, &session)?;
        assert_execution_paths(
            &[input],
            |inputs| inputs[0].clone().apply(&restored),
            ExpectedOutcome::Array(expected.into_array()),
            &mut ctx,
        );
        Ok(())
    }

    #[rstest]
    #[case::time_rounds_up_to_midnight(
        Time::new(TimeUnit::Milliseconds, NON_NULLABLE).erased(),
        buffer![86_399_500i32].into_array(),
        TimeUnit::Seconds,
        TemporalRounding::NearestTiesAwayFromZero,
    )]
    #[case::exact_with_remainder(
        Timestamp::new(TimeUnit::Nanoseconds, NON_NULLABLE).erased(),
        buffer![1_500_000i64].into_array(),
        TimeUnit::Milliseconds,
        TemporalRounding::Exact,
    )]
    fn test_change_time_unit_errors(
        #[case] source: ExtDTypeRef,
        #[case] storage: ArrayRef,
        #[case] unit: TimeUnit,
        #[case] rounding: TemporalRounding,
    ) -> VortexResult<()> {
        let input = ExtensionArray::try_new(source, storage)?;
        let operation = change_time_unit(root(), unit, rounding);
        assert_execution_paths(
            &[input.into_array()],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Error,
            &mut VortexSession::empty().create_execution_ctx(),
        );
        Ok(())
    }

    #[rstest]
    #[case::date(
        timestamp_to_date(root()),
        Date::new(TimeUnit::Days, NON_NULLABLE).erased(),
        buffer![-1i32, 19_792, 19_792].into_array(),
    )]
    #[case::time(
        timestamp_to_time(root()),
        Time::new(TimeUnit::Seconds, NON_NULLABLE).erased(),
        buffer![68_400i32, 7_199, 10_800].into_array(),
    )]
    #[case::zone(
        convert_time_zone(root(), "UTC"),
        Timestamp::new_with_tz(TimeUnit::Seconds, Some("UTC".into()), NON_NULLABLE).erased(),
        buffer![0i64, 1_710_053_999, 1_710_054_000].into_array(),
    )]
    fn test_timestamp_calendar_and_display_zone(
        #[case] operation: Expression,
        #[case] target: ExtDTypeRef,
        #[case] expected_storage: ArrayRef,
    ) -> VortexResult<()> {
        let input = ExtensionArray::try_new(
            Timestamp::new_with_tz(
                TimeUnit::Seconds,
                Some("America/New_York".into()),
                NON_NULLABLE,
            )
            .erased(),
            buffer![
                0i64,          // The epoch, which is 1969-12-31 19:00:00 EST in New York.
                1_710_053_999, // 2024-03-10 01:59:59 EST, the last second before DST starts.
                1_710_054_000, // 2024-03-10 03:00:00 EDT, after the offset moves from -5h to -4h.
            ]
            .into_array(),
        )?;
        let expected = ExtensionArray::try_new(target, expected_storage)?;
        let session = VortexSession::empty();
        let restored = Expression::from_proto(&operation.serialize_proto()?, &session)?;

        assert_execution_paths(
            &[input.into_array()],
            |inputs| inputs[0].clone().apply(&restored),
            ExpectedOutcome::Array(expected.into_array()),
            &mut session.create_execution_ctx(),
        );
        Ok(())
    }

    #[test]
    fn test_convert_time_zone_pushes_into_dictionary_values() -> VortexResult<()> {
        let zoned = |zone: &str, nullability: Nullability| {
            Timestamp::new_with_tz(TimeUnit::Seconds, Some(zone.into()), nullability).erased()
        };

        // The null code needs a strict function, and the unused value `1` needs an infallible one.
        let input = DictArray::try_new(
            PrimitiveArray::from_option_iter([Some(0u8), None, Some(0)]).into_array(),
            ExtensionArray::try_new(
                zoned("America/New_York", NON_NULLABLE),
                buffer![0i64, 1].into_array(),
            )?
            .into_array(),
        )?;
        let mut ctx = VortexSession::empty().create_execution_ctx();

        let result = input
            .into_array()
            .apply(&convert_time_zone(root(), "UTC"))?
            .execute_until::<Dict>(&mut ctx)?;

        assert!(result.is::<Dict>());
        let expected = ExtensionArray::try_new(
            zoned("UTC", Nullability::Nullable),
            PrimitiveArray::from_option_iter([Some(0i64), None, Some(0)]).into_array(),
        )?;
        assert_arrays_eq!(result, expected.into_array(), &mut ctx);
        Ok(())
    }
}
