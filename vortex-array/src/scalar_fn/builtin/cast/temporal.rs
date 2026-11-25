// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The exact conversions that [`Cast`](super::Cast) performs between built-in temporal types.
//!
//! `cast` treats extension types as opaque, except for the pairs of temporal types in which every
//! valid source value has exactly one target value. A conversion that can lose precision or that
//! depends on a time zone rule is a named function instead, and [`temporal_cast_hint`] names that
//! function when `cast` rejects the pair.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::extension::datetime::AnyTemporal;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::check_time_of_day;
use crate::extension::datetime::map_counts;
use crate::extension::datetime::map_scalar_count;
use crate::scalar::Scalar;

/// An exact conversion between two built-in temporal dtypes.
///
/// Every allowed pair converts by multiplication alone: the cast reads a count from each source
/// value and multiplies it by the number of target units in one unit of that count. The cast fails
/// when the product does not fit in `i64` or in the target storage, or when a source value is not
/// valid for its type.
pub(crate) struct TemporalCast {
    source_unit: TimeUnit,
    target_unit: TimeUnit,
    source_count: SourceCount,
    /// The number of target units in one unit of the source count.
    factor: i64,
}

/// How a [`TemporalCast`] reads the count that it multiplies from a source value.
enum SourceCount {
    /// A timestamp count, which can hold any `i64`.
    Timestamp,
    /// A time of day, which must lie within one day.
    TimeOfDay,
    /// The whole days of a date. A date in milliseconds must not hold a partial day.
    WholeDays,
}

impl TemporalCast {
    /// Select the exact conversion between two temporal dtypes, or return `None` when `cast` does
    /// not convert the pair.
    ///
    /// `cast` converts a timestamp or a time to a finer unit, keeping the time zone of a
    /// timestamp. It also converts a date in days to a date in milliseconds, and a date to the
    /// start of that day as a timestamp without a time zone. A date has no such conversion to a
    /// zoned timestamp, because the instant at which a day starts depends on the zone.
    pub(crate) fn classify(source: &DType, target: &DType) -> Option<Self> {
        let source = source.as_extension_opt()?.metadata_opt::<AnyTemporal>()?;
        let target = target.as_extension_opt()?.metadata_opt::<AnyTemporal>()?;
        let source_unit = source.time_unit();
        let target_unit = target.time_unit();
        let source_nanos = source_unit.nanos_per_unit();
        let target_nanos = target_unit.nanos_per_unit();

        let (source_count, factor) = match (source, target) {
            (
                TemporalMetadata::Timestamp(_, source_zone),
                TemporalMetadata::Timestamp(_, target_zone),
            ) if target_nanos < source_nanos && source_zone == target_zone => {
                (SourceCount::Timestamp, source_nanos / target_nanos)
            }
            (TemporalMetadata::Time(_), TemporalMetadata::Time(_))
                if target_nanos < source_nanos =>
            {
                (SourceCount::TimeOfDay, source_nanos / target_nanos)
            }
            (
                TemporalMetadata::Date(TimeUnit::Days),
                TemporalMetadata::Date(TimeUnit::Milliseconds),
            )
            | (TemporalMetadata::Date(_), TemporalMetadata::Timestamp(_, None)) => {
                (SourceCount::WholeDays, target_unit.units_per_day())
            }
            _ => return None,
        };

        Some(Self {
            source_unit,
            target_unit,
            source_count,
            factor,
        })
    }

    /// Convert a scalar to the target dtype that selected this cast.
    pub(crate) fn cast_scalar(&self, input: &Scalar, target: &DType) -> VortexResult<Scalar> {
        map_scalar_count(input, target, |value| self.convert(value))
    }

    /// Convert each valid row of an array to the target dtype that selected this cast.
    pub(crate) fn cast_array(
        &self,
        input: ArrayRef,
        target: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        map_counts(input, target, ctx, |value| self.convert(value))
    }

    fn convert(&self, value: i64) -> VortexResult<i64> {
        let count = match self.source_count {
            SourceCount::Timestamp => value,
            SourceCount::TimeOfDay => {
                check_time_of_day(value, self.source_unit)?;
                value
            }
            SourceCount::WholeDays => whole_days(value, self.source_unit)?,
        };

        count.checked_mul(self.factor).ok_or_else(|| {
            vortex_err!(
                "Expected a {} count that fits in i64 as {}, got {value}",
                self.source_unit,
                self.target_unit,
            )
        })
    }
}

/// Return the number of whole days in a date count of `unit`.
///
/// Arrow requires a date in milliseconds to hold a whole number of days, so a partial day makes
/// the date invalid. A date in days always holds whole days.
fn whole_days(value: i64, unit: TimeUnit) -> VortexResult<i64> {
    let units_per_day = unit.units_per_day();
    vortex_ensure_eq!(
        value % units_per_day,
        0,
        "Expected a date of whole days in {unit}, got {value}"
    );
    Ok(value / units_per_day)
}

/// Name the function that performs a temporal conversion that `cast` rejects, if one exists.
pub(super) fn temporal_cast_hint(source: &DType, target: &DType) -> Option<&'static str> {
    let source = source.as_extension_opt()?.metadata_opt::<AnyTemporal>()?;
    let target = target.as_extension_opt()?.metadata_opt::<AnyTemporal>()?;

    match (source, target) {
        (
            TemporalMetadata::Timestamp(_, Some(source_zone)),
            TemporalMetadata::Timestamp(_, Some(target_zone)),
        ) if source_zone != target_zone => {
            Some("Use convert_time_zone to change the time zone while preserving the instant.")
        }
        (
            TemporalMetadata::Timestamp(_, source_zone),
            TemporalMetadata::Timestamp(_, target_zone),
        ) if source_zone.is_some() != target_zone.is_some() => {
            Some("Use replace_time_zone to add or remove a time zone while preserving wall time.")
        }
        (TemporalMetadata::Timestamp(..), TemporalMetadata::Date(_)) => {
            Some("Use timestamp_to_date to extract the local calendar date in days.")
        }
        (TemporalMetadata::Timestamp(..), TemporalMetadata::Time(_)) => {
            Some("Use timestamp_to_time to extract the local time of day in the timestamp's unit.")
        }
        (TemporalMetadata::Date(from), TemporalMetadata::Date(to))
        | (TemporalMetadata::Time(from), TemporalMetadata::Time(to))
        | (TemporalMetadata::Timestamp(from, _), TemporalMetadata::Timestamp(to, _))
            if from.nanos_per_unit() < to.nanos_per_unit() =>
        {
            Some("Use change_time_unit to select an explicit rounding mode.")
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::extension::ExtDTypeRef;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;
    use crate::test_harness::ExpectedOutcome;
    use crate::test_harness::assert_execution_paths;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    const NON_NULLABLE: Nullability = Nullability::NonNullable;

    /// Cast `storage` of the `source` dtype to `target` on every execution path.
    #[track_caller]
    fn assert_cast(
        source: ExtDTypeRef,
        storage: ArrayRef,
        target: &ExtDTypeRef,
        expected: ExpectedOutcome,
    ) -> VortexResult<()> {
        let input = ExtensionArray::try_new(source, storage)?.into_array();
        let target = DType::Extension(target.clone());
        assert_execution_paths(
            &[input],
            |inputs| inputs[0].cast(target.clone()),
            expected,
            &mut SESSION.create_execution_ctx(),
        );
        Ok(())
    }

    #[test]
    fn test_widen_timestamp_skips_null_payload() -> VortexResult<()> {
        let utc =
            |unit| Timestamp::new_with_tz(unit, Some("UTC".into()), Nullability::Nullable).erased();
        let target = utc(TimeUnit::Nanoseconds);
        let storage = PrimitiveArray::new(
            buffer![
                1_000i64, //
                i64::MAX, // A null payload that overflows when scaled.
            ],
            Validity::from_iter([true, false]),
        );
        let expected = ExtensionArray::try_new(
            target.clone(),
            PrimitiveArray::from_option_iter([Some(1_000_000_000i64), None]).into_array(),
        )?;

        assert_cast(
            utc(TimeUnit::Milliseconds),
            storage.into_array(),
            &target,
            ExpectedOutcome::Array(expected.into_array()),
        )
    }

    #[rstest]
    #[case::date_to_milliseconds(
        Date::new(TimeUnit::Days, NON_NULLABLE).erased(),
        buffer![-1i32, 2].into_array(),
        Date::new(TimeUnit::Milliseconds, NON_NULLABLE).erased(),
        buffer![-86_400_000i64, 172_800_000].into_array(),
    )]
    #[case::date_to_timestamp(
        Date::new(TimeUnit::Days, NON_NULLABLE).erased(),
        buffer![-1i32, 2].into_array(),
        Timestamp::new(TimeUnit::Milliseconds, NON_NULLABLE).erased(),
        buffer![-86_400_000i64, 172_800_000].into_array(),
    )]
    #[case::millisecond_date_to_seconds(
        Date::new(TimeUnit::Milliseconds, NON_NULLABLE).erased(),
        buffer![-86_400_000i64, 172_800_000].into_array(),
        Timestamp::new(TimeUnit::Seconds, NON_NULLABLE).erased(),
        buffer![-86_400i64, 172_800].into_array(),
    )]
    #[case::time_to_nanoseconds(
        Time::new(TimeUnit::Seconds, NON_NULLABLE).erased(),
        buffer![0i32, 86_399].into_array(),
        Time::new(TimeUnit::Nanoseconds, NON_NULLABLE).erased(),
        buffer![0i64, 86_399_000_000_000].into_array(),
    )]
    fn test_exact_date_and_time_casts(
        #[case] source: ExtDTypeRef,
        #[case] storage: ArrayRef,
        #[case] target: ExtDTypeRef,
        #[case] expected_storage: ArrayRef,
    ) -> VortexResult<()> {
        let expected = ExtensionArray::try_new(target.clone(), expected_storage)?.into_array();
        assert_cast(source, storage, &target, ExpectedOutcome::Array(expected))
    }

    #[rstest]
    #[case::timestamp_overflow(
        Timestamp::new(TimeUnit::Milliseconds, NON_NULLABLE).erased(),
        buffer![i64::MAX].into_array(),
        Timestamp::new(TimeUnit::Nanoseconds, NON_NULLABLE).erased(),
    )]
    #[case::unaligned_date(
        Date::new(TimeUnit::Milliseconds, NON_NULLABLE).erased(),
        buffer![1i64].into_array(),
        Timestamp::new(TimeUnit::Milliseconds, NON_NULLABLE).erased(),
    )]
    fn test_exact_cast_errors(
        #[case] source: ExtDTypeRef,
        #[case] storage: ArrayRef,
        #[case] target: ExtDTypeRef,
    ) -> VortexResult<()> {
        assert_cast(source, storage, &target, ExpectedOutcome::Error)
    }

    #[test]
    fn test_null_date_casts_to_null_naive_timestamp() -> VortexResult<()> {
        let target =
            DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullability::Nullable).erased());
        let date = DType::Extension(Date::new(TimeUnit::Days, Nullability::Nullable).erased());

        let result = Scalar::null(date).cast(&target)?;

        assert_eq!(result, Scalar::null(target));
        Ok(())
    }
}
